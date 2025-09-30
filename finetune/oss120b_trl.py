#!/usr/bin/env python
# finetune/oss120b_trl_bf16.py
#
# Dequantize MXFP4 -> bf16 (trainable), no ZeRO-3. LoRA on attention/MLP.
# Shards across available GPUs via accelerate device_map/max_memory.

import os, re, torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    Mxfp4Config,            # <- needed for MXFP4 -> bf16 dequantize
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# ---------------- Env / caches ----------------
os.environ.setdefault("HF_HOME", "/mnt/custom-file-systems/efs/_hf_cache")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("TMPDIR", "/mnt/custom-file-systems/efs/_tmp")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.8")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

# ---------------- Config ----------------
MODEL_NAME = "openai/gpt-oss-120b"
SEED = 3407
MAX_LEN = 1024  # reduce to 768 if you OOM

torch.manual_seed(SEED)

# ---------------- Tokenizer ----------------
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.model_max_length = MAX_LEN  # TRL will truncate to this

# (optional) sanity-load model config
_ = AutoConfig.from_pretrained(MODEL_NAME)

# ---------------- Device map & memory ----------------
# Use integer keys for GPUs; allow a CPU bucket for spill if needed.
n_gpus = torch.cuda.device_count()
max_memory = {i: "70GiB" for i in range(n_gpus)}   # leave ~10GB headroom per H100-80
max_memory["cpu"] = os.environ.get("CPU_MAX_MEM", "400GiB")  # adjust to your host RAM

# Prefer a balanced map; fall back to auto if unavailable.
device_map = "balanced_low_0"
# Some accelerate versions may not like "balanced_low_0"; we'll catch and retry below.

# ---------------- Model (MXFP4 -> bf16 dequantized) ----------------
def load_bf16_model():
    try:
        return AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            quantization_config=Mxfp4Config(dequantize=True),  # <- key: dequantize MXFP4
            dtype=torch.bfloat16,                               # use dtype= (not torch_dtype)
            device_map=device_map,
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
    except Exception as e:
        # retry with device_map="auto"
        print(f"[warn] device_map='{device_map}' failed ({e}); retrying with device_map='auto'")
        return AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            quantization_config=Mxfp4Config(dequantize=True),
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )

model = load_bf16_model()
model.gradient_checkpointing_enable()
model.config.use_cache = False  # required with gradient checkpointing

# ---------------- LoRA ----------------
def guess_lora_targets(m):
    names = {n for n, _ in m.named_modules()}
    if any(re.search(r"\bc_attn\b", n) for n in names):  # GPT-2 style
        return ["c_attn", "c_proj", "mlp.c_fc", "mlp.c_proj"]
    # LLaMA/Qwen style
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

peft_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    target_modules=guess_lora_targets(model),
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_cfg)

# ---------------- Dataset ----------------
ds = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")

def to_text(batch):
    out = []
    for convo in batch["messages"]:
        if getattr(tok, "chat_template", None):
            s = tok.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        else:
            s = "".join(f"<|{m.get('role','user')}|>: {m.get('content','')}\n" for m in convo)
        out.append(s)
    return {"text": out}

ds = ds.map(
    to_text,
    batched=True,
    remove_columns=[c for c in ds.column_names if c != "text"],
    num_proc=1,  # avoid AF_UNIX path-too-long issues on some platforms
)

def formatting_func(example):
    return example["text"]  # TRL expects str or list[str]

# ---------------- TRL config ----------------
train_cfg = SFTConfig(
    output_dir="outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,   # increase if you OOM
    max_steps=30,
    learning_rate=2e-4,
    lr_scheduler_type="linear",
    warmup_steps=5,
    weight_decay=0.01,
    logging_steps=1,
    bf16=True, fp16=False,
    optim="adamw_torch",
    gradient_checkpointing=True,
    group_by_length=True,
    seed=SEED,
    report_to="none",
    packing=False,
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    model=model,
    args=train_cfg,
    train_dataset=ds,
    formatting_func=formatting_func,
    processing_class=tok,  # TRL >=0.23 uses processing_class (tokenizer= also works on some versions)
)

if __name__ == "__main__":
    trainer.train()
    trainer.model.save_pretrained("outputs/lora_adapter")
    tok.save_pretrained("outputs/lora_adapter")

#!/usr/bin/env python
# finetune/oss120b_trl_lenfilter_wandb.py
# - Single-load 120B (MXFP4 -> bf16), attention-only LoRA by default.
# - Filters out over-length examples (no truncation).
# - Logs every step; W&B enabled; saves every 100 steps (keep last 10).
# - Fixes numpy.int64 indexing for datasets.

import os, math, json
import numpy as np
import torch, multiprocessing as mp
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
from accelerate import init_empty_weights
try:
    from accelerate.utils import get_balanced_device_map
except Exception:
    get_balanced_device_map = None
try:
    from accelerate.utils import infer_auto_device_map
except Exception:
    infer_auto_device_map = None
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# ---------------- env early ----------------
os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.8,expandable_segments:True")
os.environ.setdefault("HF_HOME", "/mnt/custom-file-systems/efs/_hf_cache")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# ---------------- user/config knobs ----------------
DATA_PATH = "train_cot_long.jsonl"      # each line: {"messages":[...]}
MODEL_NAME = "openai/gpt-oss-120b"
SEED = 3407
MAX_LEN = 2048
EXPERTS = False                          # keep False unless you add MoE ParamWrapper targets
# LoRA hyperparams (tunable)
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05                      # <--- tunable dropout
# training hyperparams
PER_DEVICE_BATCH = 2
GRAD_ACCUM = 4
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
SAVE_STEPS = 75
SAVE_TOTAL_LIMIT = 10                    # <--- per your request
LOGGING_STEPS = 1                        # <--- log every step
PROJECT = os.environ.get("WANDB_PROJECT", "oss120b-finetune")
ENTITY = os.environ.get("WANDB_ENTITY", None)  # optional

torch.manual_seed(SEED)

# ---- spawn start method ----
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# ---------------- tokenizer ----------------
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tok.pad_token = tok.pad_token or tok.eos_token
tok.model_max_length = MAX_LEN  # we filter so we shouldn't truncate, but keep this aligned

# ---------------- device_map (single-load path) ----------------
assert torch.cuda.device_count() >= 8, "Need >= 8 GPUs for this script"
max_memory = {i: "64GiB" for i in range(torch.cuda.device_count())}
max_memory["cpu"] = "0GiB"

cfg = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
with init_empty_weights():
    empty = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

no_split = ["GptOssDecoderLayer", "DecoderLayer", "GPTBlock", "TransformerLayer"]
if get_balanced_device_map is not None:
    device_map = get_balanced_device_map(empty, max_memory=max_memory, no_split_module_classes=no_split)
elif infer_auto_device_map is not None:
    device_map = infer_auto_device_map(empty, max_memory=max_memory, no_split_module_classes=no_split)
else:
    device_map = "auto"

# ---------------- real load (ONE call; MXFP4 -> bf16) ----------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    quantization_config=Mxfp4Config(dequantize=True),
    dtype=torch.bfloat16,
    device_map=device_map,
    low_cpu_mem_usage=True,
    attn_implementation="eager",  # padding-free/packing disabled below, so eager is safe
)
model.gradient_checkpointing_enable()
model.config.use_cache = False
print("Device map:", getattr(model, "hf_device_map", device_map))

# ---------------- LoRA ----------------
attn_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj"]
if EXPERTS:
    raise NotImplementedError("EXPERTS=False is required here to avoid ParamWrapper OOMs.")

peft_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=attn_targets,
    target_parameters=None,
)
model = get_peft_model(model, peft_cfg)

# ===== Inspect LoRA targets =====
def collect_lora_targets(m):
    lora_modules = set()
    lora_params = set()
    for name, mod in m.named_modules():
        mod_name = mod.__class__.__name__.lower()
        if hasattr(mod, "lora_A") or hasattr(mod, "lora_B") or "lora" in mod_name:
            lora_modules.add(name)
        if hasattr(mod, "parametrizations"):
            for p_name in list(getattr(mod, "_parameters", {}).keys()):
                if p_name in mod.parametrizations:
                    plist = mod.parametrizations[p_name]
                    if any("lora" in p.__class__.__name__.lower() for p in plist):
                        lora_params.add(f"{name}.{p_name}")
    return sorted(lora_modules), sorted(lora_params)

lora_mods, lora_param_targets = collect_lora_targets(model)
print("\n=== LoRA MODULE TARGETS (module-wrapped) ===")
print(f"count={len(lora_mods)}")
for n in lora_mods:
    print(n)
print("\n=== LoRA PARAMETER TARGETS (ParamWrapper / experts) ===")
print(f"count={len(lora_param_targets)}")
for n in lora_param_targets:
    print(n)
print("\n=== Active adapters ===", getattr(model, "active_adapter", None))
model.print_trainable_parameters()

# ---------------- dataset: load, to_text, length stats, filter ----------------
ds = load_dataset("json", data_files={"train": DATA_PATH}, split="train")

def to_text(batch):
    out = []
    for convo in batch["messages"]:
        if getattr(tok, "chat_template", None):
            s = tok.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        else:
            s = "".join(f"<|{m.get('role','user')}|>: {m.get('content','')}\n" for m in convo)
        out.append(s)
    return {"text": out}

# keep both 'messages' and 'text' for debugging
ds = ds.map(to_text, batched=True, remove_columns=[], num_proc=1)

# token length (no special tokens) and descriptive stats
def _len_map(batch):
    return {"n_tokens": [len(tok.encode(t, add_special_tokens=False)) for t in batch["text"]]}
ds = ds.map(_len_map, batched=True, num_proc=1)

arr = np.array(ds["n_tokens"], dtype=np.int32)
def pct(a, p): return int(np.percentile(a, p)) if len(a) else 0
print(f"\n=== Dataset token-length stats (no truncation) ===")
print(f"examples: {len(ds)}   MAX_LEN: {MAX_LEN}")
print(f"min/mean/median: {arr.min()} / {int(arr.mean())} / {int(np.median(arr))}")
print(f"p90/p95/p99: {pct(arr,90)} / {pct(arr,95)} / {pct(arr,99)}")
print(f"max: {arr.max()}")
over = int((arr > MAX_LEN).sum())
print(f"num > MAX_LEN: {over}  ({over*100.0/len(arr):.1f}%)")

# drop any example that won't fit -> guarantees no truncation
total_before = len(ds)
ds = ds.filter(lambda ex: ex["n_tokens"] <= MAX_LEN)
print(f"\n=== Filtering by MAX_LEN={MAX_LEN} ===")
print(f"total: {total_before}  kept: {len(ds)}  dropped: {total_before-len(ds)}  ({(total_before-len(ds))*100.0/total_before:.1f}%)")

# --- preview longest few + median + a couple randoms (SAVE full text to files) ---
os.makedirs("outputs/preview", exist_ok=True)
n_tokens = np.array(ds["n_tokens"], dtype=np.int32)
sorted_idx = np.argsort(n_tokens)

def _save_example(idx, tag):
    i = int(idx)  # <<< FIX: cast numpy.int64 -> int for datasets indexer
    tokens = int(ds[i]["n_tokens"])
    txt = ds[i]["text"]
    path = f"outputs/preview/full_example_{tag}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"[saved] {path} (tokens={tokens})")

# longest three
for j, idx in enumerate(sorted_idx[-3:][::-1]):
    _save_example(idx, f"long_{j}")

# median
mid = sorted_idx[len(sorted_idx)//2]
_save_example(mid, "median")

# a couple randoms
rng = np.random.default_rng(SEED)
for ridx in rng.choice(sorted_idx, size=min(2, len(sorted_idx)), replace=False):
    _save_example(ridx, f"rand_{int(ridx)}")

# ---------------- formatting for TRL ----------------
def formatting_func(example):
    return example["text"]  # TRL expects str or list[str]

# ---------------- W&B ----------------
import wandb
wandb.init(project=PROJECT, entity=ENTITY, config={
    "model_name": MODEL_NAME,
    "epochs": NUM_EPOCHS,
    "per_device_batch_size": PER_DEVICE_BATCH,
    "grad_accum_steps": GRAD_ACCUM,
    "lr": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "warmup_ratio": WARMUP_RATIO,
    "max_len": MAX_LEN,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": LORA_DROPOUT,
})
print(f"W&B run URL: {wandb.run.url}")

# ---------------- TRL config (epochs, save every 100 steps) ----------------
train_cfg = SFTConfig(
    output_dir="outputs",
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="linear",
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    logging_steps=LOGGING_STEPS,
    logging_strategy="steps",
    bf16=True, fp16=False,
    optim="adamw_torch",
    gradient_checkpointing=True,
    group_by_length=True,
    seed=SEED,
    report_to=["wandb"],
    packing=False,                       # keep False to avoid Flash-Attn requirements
    remove_unused_columns=False,
    dataloader_num_workers=0,            # safer on SageMaker + huge models
    dataloader_pin_memory=False,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    save_safetensors=True,
    save_only_model=True,                # save adapter-only at checkpoints
)

# ---------- Print run config summary ----------
world_size = 1
updates_per_epoch = math.ceil(len(ds) / (PER_DEVICE_BATCH * GRAD_ACCUM * world_size))
total_update_steps = updates_per_epoch * NUM_EPOCHS
print("\n=== Run config summary ===")
print(json.dumps({
    "dataset_examples_after_filter": len(ds),
    "estimated_updates_per_epoch": updates_per_epoch,
    "estimated_total_update_steps": total_update_steps,
}, indent=2))

print("\n=== TrainingArguments (subset) ===")
print(train_cfg)

# ---------------- Trainer ----------------
trainer = SFTTrainer(
    model=model,
    args=train_cfg,
    train_dataset=ds,
    formatting_func=formatting_func,
    processing_class=tok,
)

if __name__ == "__main__":
    trainer.train()
    # final save: LoRA adapter only
    trainer.model.save_pretrained("outputs/lora_adapter")
    tok.save_pretrained("outputs/lora_adapter")
    print("Training complete.")
    print(f"W&B run URL: {wandb.run.url}")

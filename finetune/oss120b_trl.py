#!/usr/bin/env python
# finetune/oss120b_trl_singleload_lenfilter.py
# - Single-load 120B (MXFP4 -> bf16) with manual device_map via init_empty_weights.
# - TRL SFT over JSONL {"messages":[...]} using tokenizer chat_template.
# - Attention-only LoRA by default (safe); EXPERTS=False avoids MoE ParamWrapper.
# - Filters out over-length examples so nothing is truncated.
# - Trains for 2 epochs; saves every 100 steps; prints config & dataset stats.

import os
# ---- env early ----
os.environ["TMPDIR"] = "/tmp"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.8,expandable_segments:True"
)
os.environ.setdefault("HF_HOME", "/mnt/custom-file-systems/efs/_hf_cache")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import math
import json
import numpy as np
import torch, multiprocessing as mp
from datasets import load_dataset
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
)
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

# ---- spawn start method ----
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# ---------------- user/config knobs ----------------
DATA_PATH = "train_cot_long.jsonl"  # each line: {"messages":[...]}
MODEL_NAME = "openai/gpt-oss-120b"
SEED = 3407
MAX_LEN = 2048                      # fit all but the longest ~0.5%; we will DROP > MAX_LEN
EXPERTS = False                     # keep False unless you add tiny expert targets
LORA_R = 32                         # bump if you have headroom
LORA_ALPHA = 64                     # ~2x r is a good default
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 4
NUM_EPOCHS = 2
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 3
LEARNING_RATE = 2e-4

torch.manual_seed(SEED)

# ---------------- tokenizer ----------------
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tok.pad_token = tok.pad_token or tok.eos_token
tok.model_max_length = MAX_LEN  # TRL will use this for truncation (but we filter instead)

# ---------------- device_map (single-load path) ----------------
assert torch.cuda.device_count() >= 8, "Need >= 8 GPUs for this script"
# tighten per-GPU cap so mapper spreads across all 8; no CPU offload by default
max_memory = {i: "64GiB" for i in range(torch.cuda.device_count())}
max_memory["cpu"] = "0GiB"  # set to "200GiB" if you want some CPU spill

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
    attn_implementation="eager",
)
model.gradient_checkpointing_enable()
model.config.use_cache = False
print("Device map:", getattr(model, "hf_device_map", device_map))

# ---------------- LoRA ----------------
attn_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj"]

target_parameters = None
if EXPERTS:
    # If you decide to touch experts later, keep it tiny, e.g. one layer down_proj:
    # target_parameters = ["23.mlp.experts.down_proj"]  # example
    # and set LORA_R small (e.g., 2–4) with lora_dropout=0.0
    raise NotImplementedError("Set EXPERTS=False for this script configuration.")

peft_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.0,             # required if you later use ParamWrapper targets
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=attn_targets,
    target_parameters=target_parameters,
)
model = get_peft_model(model, peft_cfg)

# ===== Inspect model & LoRA targets =====
def print_model_summary(m):
    print("\n=== Model module summary (name -> type) ===")
    for name, mod in m.named_modules():
        if name == "":
            continue
        print(f"{name}: {mod.__class__.__name__}")

def collect_lora_targets(m):
    lora_modules = set()
    lora_params = set()
    for name, mod in m.named_modules():
        mod_name = mod.__class__.__name__.lower()
        if hasattr(mod, "lora_A") or hasattr(mod, "lora_B") or "lora" in mod_name:
            lora_modules.add(name)
        # ParamWrapper targets show up as parametrizations on parameters
        if hasattr(mod, "parametrizations"):
            for p_name in list(getattr(mod, "_parameters", {}).keys()):
                if p_name in mod.parametrizations:
                    plist = mod.parametrizations[p_name]
                    if any("lora" in p.__class__.__name__.lower() for p in plist):
                        lora_params.add(f"{name}.{p_name}")
    return sorted(lora_modules), sorted(lora_params)

# Optional: print full module tree (huge). Comment out if too verbose.
# print_model_summary(model)

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

# --- optional: preview longest few and median example (save full text) ---
os.makedirs("outputs/preview", exist_ok=True)
sorted_idx = np.argsort(np.array(ds["n_tokens"]))
if len(sorted_idx):
    # longest 3
    for ridx in sorted_idx[-3:][::-1]:
        tokens = ds[ridx]["n_tokens"]
        txt = ds[ridx]["text"]
        path = f"outputs/preview/kept_example_{ridx}_tokens{tokens}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt)
        print(f"[saved] {path}")
    # median
    mid = sorted_idx[len(sorted_idx)//2]
    pathm = f"outputs/preview/kept_example_{mid}_tokens{ds[mid]['n_tokens']}_median.txt"
    with open(pathm, "w", encoding="utf-8") as f:
        f.write(ds[mid]["text"])
    print(f"[saved] {pathm}")

# ---------------- formatting for TRL ----------------
def formatting_func(example):
    return example["text"]  # TRL expects str or list[str]; no truncation done here

# ---------------- TRL config (2 epochs, save every 100 steps) ----------------
train_cfg = SFTConfig(
    output_dir="outputs",
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,       # <-- epochs, not fixed steps
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="linear",
    warmup_ratio=0.03,                 # small ratio when using epochs
    weight_decay=0.01,
    logging_steps=10,
    bf16=True, fp16=False,
    optim="adamw_torch",
    gradient_checkpointing=True,
    group_by_length=True,
    seed=SEED,
    report_to="none",
    packing=False,
    remove_unused_columns=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    save_safetensors=True,
)

# ---------- Print run config summary before training ----------
world_size = 1  # single process; model sharded via device_map (no DDP)
eff_batch_tokens = f"<= {MAX_LEN}"
updates_per_epoch = math.ceil(len(ds) / (PER_DEVICE_BATCH * GRAD_ACCUM * world_size))
total_update_steps = updates_per_epoch * NUM_EPOCHS

print("\n=== Run config summary ===")
print(json.dumps({
    "model_name": MODEL_NAME,
    "bf16": True,
    "epochs": NUM_EPOCHS,
    "save_steps": SAVE_STEPS,
    "save_total_limit": SAVE_TOTAL_LIMIT,
    "per_device_batch_size": PER_DEVICE_BATCH,
    "grad_accum_steps": GRAD_ACCUM,
    "world_size_processes": world_size,
    "effective_batch_size_per_update": PER_DEVICE_BATCH * GRAD_ACCUM * world_size,
    "max_seq_len_tokens": MAX_LEN,
    "estimated_updates_per_epoch": updates_per_epoch,
    "estimated_total_update_steps": total_update_steps,
    "dataset_examples_after_filter": len(ds),
    "lora": {
        "targets": attn_targets,
        "r": LORA_R,
        "alpha": LORA_ALPHA,
        "dropout": 0.0,
        "experts": EXPERTS
    }
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
    trainer.model.save_pretrained("outputs/lora_adapter")
    tok.save_pretrained("outputs/lora_adapter")

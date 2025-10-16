#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HF Transformers training script that adapts to your TRL version:
- Prefer TRL SFTTrainer (new or old API). If TRL is missing, fall back to HF Trainer.
- Apply an Accelerate unwrap_model compatibility shim unconditionally to avoid
  `keep_torch_compile` crashes on older accelerate versions.

Other behavior unchanged (LoRA, prompts, chat template, hyperparams).
"""

import os
import re
import json
import warnings
import inspect

import torch
import pandas as pd
import numpy as np

from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import Mxfp4Config  # available in recent Transformers

# Prefer new SDPA context; fall back if needed
os.environ.setdefault("HF_PYTORCH_ATTENTION_BACKEND", "eager")
try:
    from torch.nn.attention import sdpa_kernel
    sdpa_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
except Exception:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)

# Memory fragmentation guards (safe defaults)
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.8,expandable_segments:True",
)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --- Config you used ---
MODEL_ID = "openai/gpt-oss-120b"
MAX_SEQ_LEN = 4096

print(f"Loading model: {MODEL_ID}")
print(f"Max sequence length: {MAX_SEQ_LEN}")

# --------------------------
# Build a balanced device_map across 8x A100 80GB
# --------------------------
from accelerate import init_empty_weights
try:
    from accelerate.utils import get_balanced_device_map
except Exception:
    get_balanced_device_map = None
try:
    from accelerate.utils import infer_auto_device_map
except Exception:
    infer_auto_device_map = None

assert torch.cuda.device_count() >= 8, "This script expects >= 8 GPUs visible."

# 64GiB per GPU is a conservative cap that fits 120B (bf16 after dequant) + LoRA
max_memory = {i: "64GiB" for i in range(torch.cuda.device_count())}
max_memory["cpu"] = "0GiB"

cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
with init_empty_weights():
    empty_model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

no_split = ["GptOssDecoderLayer", "DecoderLayer", "GPTBlock", "TransformerLayer"]

if get_balanced_device_map is not None:
    device_map = get_balanced_device_map(
        empty_model, max_memory=max_memory, no_split_module_classes=no_split
    )
elif infer_auto_device_map is not None:
    device_map = infer_auto_device_map(
        empty_model, max_memory=max_memory, no_split_module_classes=no_split
    )
else:
    device_map = "balanced_low_0"

# --- Load tokenizer/model (no BitsAndBytes; dequantize MXFP4 -> bf16) ---
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
tokenizer.padding_side = "left"

quant_cfg = Mxfp4Config(dequantize=True)  # <- upcast MXFP4 blocks to bf16 for training

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    quantization_config=quant_cfg,       # <- key line (dequantize on load)
    torch_dtype=torch.bfloat16,          # bf16 target dtype
    device_map=device_map,               # spread across 8x A100
    low_cpu_mem_usage=True,
    attn_implementation="eager",
)
model.gradient_checkpointing_enable()
model.config.use_cache = False
print("Device map:", getattr(model, "hf_device_map", device_map))

# --- LoRA with PEFT (unchanged) ---
from peft import LoraConfig, get_peft_model

print("\nConfiguring LoRA (matches your Unsloth setup)...")
lora_target_modules = [
    # All standard linear layers
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "embed_tokens", "lm_head",

    # Early layers (0-7)
    "1.mlp.experts.gate_up_proj", "1.mlp.experts.down_proj",
    "3.mlp.experts.gate_up_proj", "3.mlp.experts.down_proj",
    "5.mlp.experts.gate_up_proj", "5.mlp.experts.down_proj",
    "7.mlp.experts.gate_up_proj", "7.mlp.experts.down_proj",

    # Middle layers (8-15)
    "9.mlp.experts.gate_up_proj", "9.mlp.experts.down_proj",
    "11.mlp.experts.gate_up_proj", "11.mlp.experts.down_proj",
    "13.mlp.experts.gate_up_proj", "13.mlp.experts.down_proj",
    "15.mlp.experts.gate_up_proj", "15.mlp.experts.down_proj",

    # Upper layers (16-23)
    "17.mlp.experts.gate_up_proj", "17.mlp.experts.down_proj",
    "19.mlp.experts.gate_up_proj", "19.mlp.experts.down_proj",
    "21.mlp.experts.gate_up_proj", "21.mlp.experts.down_proj",
    "23.mlp.experts.gate_up_proj", "23.mlp.experts.down_proj",

    # Deep layers (24-31)
    "25.mlp.experts.gate_up_proj", "25.mlp.experts.down_proj",
    "27.mlp.experts.gate_up_proj", "27.mlp.experts.down_proj",
    "29.mlp.experts.gate_up_proj", "29.mlp.experts.down_proj",
    "31.mlp.experts.gate_up_proj", "31.mlp.experts.down_proj",
]

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    target_modules=lora_target_modules,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

def print_trainable_parameters(m):
    trainable, total = 0, 0
    for _, p in m.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = 100 * trainable / total if total else 0
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")

print_trainable_parameters(model)

# -----------------------------
# Load and process CSV with reasoning (unchanged)
# -----------------------------
print("\nLoading enhanced CSV with GPToss120B reasoning...")
enhanced_df = pd.read_csv("Enhanced_Training_Data_with_GPToss120B_Reasoning.csv")
print(f"Loaded {len(enhanced_df)} cases")
if "reasoning_matched" in enhanced_df.columns:
    print(f"Cases with reasoning: {enhanced_df['reasoning_matched'].sum()}")
    matched_df = enhanced_df[enhanced_df["reasoning_matched"] == True].copy()
else:
    warnings.warn("Column 'reasoning_matched' not found; using all rows.")
    matched_df = enhanced_df.copy()
print(f"Processing {len(matched_df)} cases with GPToss120B reasoning")

def enhanced_formatting_prompts_func(examples):
    """Format with GPToss120B reasoning as thinking content."""
    convos = []
    n = len(examples["PostDescription"])
    for i in range(n):
        combined_description = examples["PostDescription"][i]
        differential_diagnosis = examples["DifferentialDiagnosisList"][i]
        diagnosis = examples["FinalDiagnosis"][i]
        reasoning = examples["gptoss120b_reasoning"][i]

        dd_list = [dd.strip() for dd in str(differential_diagnosis).split(",")]
        dd_formatted = "\n".join(dd_list)

        user_prompt = f"""You are an expert radiologist demonstrating step-by-step diagnostic reasoning.

Case presentation:

{combined_description}

Differential diagnoses to consider:
{dd_formatted}

Generate systematic Chain-of-Thought reasoning that shows how clinicians think through cases:

1. **Connect symptoms to findings**: Link clinical presentation with imaging observations
2. **Map to differentials**: Show how findings support or contradict each differential diagnosis
3. **Systematic elimination**: Explicitly rule out less likely options with reasoning
4. **Converge to answer**: Demonstrate the logical path to the correct diagnosis"""

        conversation = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": diagnosis, "thinking": reasoning},
        ]
        convos.append(conversation)

    return {"messages": convos}

print("\nConverting to Hugging Face Dataset...")
dataset = Dataset.from_pandas(matched_df, preserve_index=False)

print("Applying enhanced formatting with GPToss120B reasoning as thinking...")
dataset = dataset.map(enhanced_formatting_prompts_func, batched=True, remove_columns=list(dataset.features))

# Quick context length stats BEFORE chat template
print("\n" + "=" * 50)
print("PRELIMINARY Context Length Analysis (before chat template):")
sample_size = min(100, len(dataset))
raw_lengths = []
for i in range(sample_size):
    messages = dataset[i]["messages"]
    total = ""
    for msg in messages:
        if msg.get("content"):
            total += str(msg["content"])
        if msg.get("thinking"):
            total += str(msg["thinking"])
    raw_lengths.append(len(total))

if raw_lengths:
    print("Raw text length statistics (chars):")
    print(f"- Min: {min(raw_lengths):,}")
    print(f"- Max: {max(raw_lengths):,}")
    print(f"- Average: {sum(raw_lengths)/len(raw_lengths):,.0f}")
    print(f"- 95th percentile: {sorted(raw_lengths)[int(0.95*len(raw_lengths))]:,}")
    est = [l // 4 for l in raw_lengths]
    print("\nEstimated token lengths:")
    print(f"- Min: {min(est):,}")
    print(f"- Max: {max(est):,}")
    print(f"- Average: {sum(est)/len(est):,.0f}")
    print(f"- 95th percentile: {sorted(est)[int(0.95*len(est))]:,}")
else:
    print("Dataset empty; skipping stats.")

out_dir = "processed_dataset_with_gptoss120b_thinking"
dataset.save_to_disk(out_dir)
print(f"Processed dataset saved to '{out_dir}'")
print("=" * 50)
print("Ready for final chat template processing!")

# -----------------------------
# Apply chat template (unchanged)
# -----------------------------
from datasets import load_from_disk

print("\nLoading processed dataset...")
dataset = load_from_disk(out_dir)
print(f"Loaded {len(dataset)} examples")

def final_formatting_func(examples):
    """Apply chat template with medium reasoning."""
    convos = examples["messages"]
    texts = []
    for convo in convos:
        text = tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False,
            reasoning_effort="medium",
        )
        texts.append(text)
    return {"text": texts}

print("Applying chat template...")
dataset = dataset.map(final_formatting_func, batched=True, remove_columns=list(dataset.features))

if len(dataset) > 0:
    print("\nSample formatted example:")
    print("=" * 50)
    sample = dataset[0]["text"]
    print(sample[:1500] + ("\n...[truncated]..." if len(sample) > 1500 else ""))

# -----------------------------
# Training (keep your knobs)
# -----------------------------
import wandb
try:
    from trl import SFTTrainer, SFTConfig
    _trl_available = True
except Exception as e:
    warnings.warn(f"TRL not available; will fall back to Transformers Trainer. Error: {e}")
    _trl_available = False

# (Optional) W&B login
try:
    wandb.login(key=os.environ.get("WANDB_API_KEY", ""))
except Exception as e:
    warnings.warn(f"W&B login skipped or failed: {e}")

# ---- Accelerate version check + universal unwrap_model shim ----
try:
    import accelerate
    from inspect import signature
    # If unwrap_model doesn't accept keep_torch_compile, patch it to drop the kwarg.
    if "keep_torch_compile" not in signature(accelerate.Accelerator.unwrap_model).parameters:
        _orig_unwrap = accelerate.Accelerator.unwrap_model
        def _shim_unwrap(self, model, *args, **kwargs):
            kwargs.pop("keep_torch_compile", None)
            return _orig_unwrap(self, model, *args, **kwargs)
        accelerate.Accelerator.unwrap_model = _shim_unwrap
        print("Applied Accelerate unwrap_model compatibility shim.")
except Exception as e:
    print(f"Could not apply unwrap_model shim (continuing): {e}")

# Conditional data_seed based on accelerate version (informational only)
try:
    from packaging import version
    _acc_ok = version.parse(accelerate.__version__) >= version.parse("1.1.0")
except Exception:
    _acc_ok = False

print("\nStarting Thinking Style Training with GPToss120B Reasoning")
print("=" * 60)

base_kwargs = dict(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    learning_rate=1e-4,
    max_seq_length=MAX_SEQ_LEN,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=25,
    report_to="wandb",
    run_name="eurorad_thinking_gptoss120b_reasoning_linearMOEOct7",
    output_dir="eurorad_thinking_gptoss120b_reasoning_linearMOEOct7",
    save_strategy="steps",
    save_steps=100,
    save_total_limit=10,
    gradient_checkpointing=True,
    dataloader_num_workers=0,
    remove_unused_columns=False,
    seed=3407,
    bf16=True,
    fp16=False,
)
if _acc_ok and _trl_available:
    base_kwargs["data_seed"] = 3407
else:
    if not _acc_ok:
        print("Note: accelerate<1.1.0 detected — proceeding without `data_seed` to avoid error.")

trainer = None

if _trl_available:
    # Prefer TRL path always (avoids direct HF Trainer usage differences)
    training_args = SFTConfig(**base_kwargs)

    # Our dataset already has 'text'; use formatting_func to preserve behavior.
    def _fmt_fn(examples):
        return examples["text"]

    # SFTTrainer API differences: some versions want `tokenizer`, others `processing_class`
    try:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            formatting_func=_fmt_fn,
            tokenizer=tokenizer,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            formatting_func=_fmt_fn,
            processing_class=tokenizer,
        )
else:
    # Fallback: vanilla Transformers Trainer with tokenized dataset
    print("TRL unavailable; using Transformers Trainer pipeline.")
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

    # Tokenize the `text` column and create labels
    def _tok_fn(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=False,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = dataset.map(_tok_fn, batched=True, remove_columns=list(dataset.features))
    dc = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    ta_kwargs = dict(
        per_device_train_batch_size=base_kwargs["per_device_train_batch_size"],
        gradient_accumulation_steps=base_kwargs["gradient_accumulation_steps"],
        num_train_epochs=base_kwargs["num_train_epochs"],
        learning_rate=base_kwargs["learning_rate"],
        weight_decay=base_kwargs["weight_decay"],
        lr_scheduler_type=base_kwargs["lr_scheduler_type"],
        warmup_ratio=base_kwargs["warmup_ratio"],
        logging_steps=base_kwargs["logging_steps"],
        report_to=base_kwargs["report_to"],
        run_name=base_kwargs["run_name"],
        output_dir=base_kwargs["output_dir"],
        save_strategy=base_kwargs["save_strategy"],
        save_steps=base_kwargs["save_steps"],
        save_total_limit=base_kwargs["save_total_limit"],
        gradient_checkpointing=base_kwargs["gradient_checkpointing"],
        dataloader_num_workers=base_kwargs["dataloader_num_workers"],
        seed=base_kwargs["seed"],
        bf16=base_kwargs["bf16"],
        fp16=base_kwargs["fp16"],
    )

    # bitsandbytes optimizer if present; else AdamW Torch
    try:
        import bitsandbytes as bnb  # noqa: F401
        ta_kwargs["optim"] = "adamw_bnb_8bit"
    except Exception:
        ta_kwargs["optim"] = "adamw_torch"
        print("bitsandbytes not found; using AdamW Torch optimizer.")

    training_args = TrainingArguments(**ta_kwargs)

    # Prefer processing_class to silence tokenizer deprecation if available
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=dc,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=dc,
            tokenizer=tokenizer,
        )

print("\nStarting thinking style training...")
trainer.train()

print("\nDone.")
print(f"Saved artifacts under: {trainer.args.output_dir if hasattr(trainer, 'args') else 'outputs'}")

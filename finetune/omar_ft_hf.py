#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HF Transformers training script that adapts to your TRL version:
- If TRL supports `SFTTrainer(..., dataset_text_field='text')`, we use it.
- Otherwise we fallback to Transformers `Trainer` with a tokenized dataset.

Other behavior unchanged (LoRA, prompts, chat template, hyperparams).
"""

import os
import warnings
import inspect

# ---- CRITICAL: keep tmp path short to avoid AF_UNIX 'path too long' with multiprocessing ----
os.environ.setdefault("TMPDIR", "/tmp")

# Perf / stability toggles (safe)
os.environ.setdefault("HF_PYTORCH_ATTENTION_BACKEND", "eager")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.8,expandable_segments:True",
)

import torch
import pandas as pd
import numpy as np

from datasets import Dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Mxfp4Config,
)
from accelerate import init_empty_weights

def main():
    # gpt-oss uses eager kernels (no SDPA)
    try:
        from torch.backends.cuda import sdp_kernel
        sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    except Exception:
        pass

    # Multiprocessing robustness: spawn + file_system sharing avoids fd passing issues
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    try:
        import torch.multiprocessing as tmp_mp
        tmp_mp.set_sharing_strategy("file_system")
    except Exception:
        pass

    # Small speed win on A100 (keeps bf16 numerics fine)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        from accelerate.utils import get_balanced_device_map
    except Exception:
        get_balanced_device_map = None
    try:
        from accelerate.utils import infer_auto_device_map
    except Exception:
        infer_auto_device_map = None

    # ========= Config you used (unchanged defaults) =========
    MODEL_ID = "openai/gpt-oss-120b"
    MAX_SEQ_LEN = 4096

    print(f"Loading model: {MODEL_ID}")
    print(f"Max sequence length: {MAX_SEQ_LEN}")

    # ========= Tokenizer =========
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"

    # ========= Build a balanced device_map over 8×A100-80GB =========
    assert torch.cuda.device_count() >= 8, "Need >= 8 GPUs for this script"
    max_memory = {i: "64GiB" for i in range(torch.cuda.device_count())}
    max_memory["cpu"] = "0GiB"

    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    with init_empty_weights():
        empty = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

    no_split = ["GptOssDecoderLayer", "DecoderLayer", "GPTBlock", "TransformerLayer"]
    if get_balanced_device_map is not None:
        device_map = get_balanced_device_map(empty, max_memory=max_memory, no_split_module_classes=no_split)
    elif infer_auto_device_map is not None:
        device_map = infer_auto_device_map(empty, max_memory=max_memory, no_split_module_classes=no_split)
    else:
        device_map = "auto"

    # ========= Real model load (MXFP4 -> bf16) =========
    quant_cfg = Mxfp4Config(dequantize=True)  # upcast MXFP4 blocks for training

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # gpt-oss does not support SDPA; keep eager
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    print("Device map:", getattr(model, "hf_device_map", device_map))

    # ========= LoRA (UNCHANGED per your request) =========
    from peft import LoraConfig, get_peft_model

    print("\nConfiguring LoRA (unchanged)…")
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
    # Apply chat template
    # -----------------------------
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
    # Training
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

    # Conditional data_seed based on accelerate version
    try:
        import accelerate
        from packaging import version
        _acc_ok = version.parse(accelerate.__version__) >= version.parse("1.1.0")
    except Exception:
        _acc_ok = False

    # ---- accelerate unwrap_model compat shim (old accelerate) ----
    try:
        import accelerate
        from inspect import signature as _sig
        if "keep_torch_compile" not in _sig(accelerate.Accelerator.unwrap_model).parameters:
            _orig_unwrap = accelerate.Accelerator.unwrap_model
            def _unwrap_model(self, model, *args, **kwargs):
                kwargs.pop("keep_torch_compile", None)
                return _orig_unwrap(self, model, *args, **kwargs)
            accelerate.Accelerator.unwrap_model = _unwrap_model
            print("Patched accelerate.Accelerator.unwrap_model for keep_torch_compile compatibility.")
    except Exception as _e:
        print("Accelerate compat shim not applied:", _e)

    print("\nStarting Thinking Style Training with GPToss120B Reasoning")
    print("=" * 60)

    # NOTE: Only the loader knobs changed to kill AF_UNIX errors (num_workers=0, no pin_memory)
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
        dataloader_num_workers=0,    # <<< fix AF_UNIX
        dataloader_pin_memory=False, # <<< avoid extra shared-mem fds
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

    def _use_trl_sft_trainer() -> bool:
        if not _trl_available:
            return False
        try:
            sig = inspect.signature(SFTTrainer.__init__).parameters
            return "dataset_text_field" in sig
        except Exception:
            return False

    if _use_trl_sft_trainer():
        training_args = SFTConfig(**base_kwargs)

        supports_tokenizer_arg = "tokenizer" in inspect.signature(SFTTrainer.__init__).parameters
        if supports_tokenizer_arg:
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                args=training_args,
            )
        else:
            from transformers import DataCollatorForLanguageModeling
            dc = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                dataset_text_field="text",
                args=training_args,
                data_collator=dc,
            )

    else:
        # Fallback: vanilla Transformers Trainer with tokenized dataset
        print("TRL too old for `dataset_text_field`; falling back to Transformers Trainer pipeline.")
        from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

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
            dataloader_num_workers=0,     # <<< fix AF_UNIX
            dataloader_pin_memory=False,  # <<< fix AF_UNIX
            remove_unused_columns=False,
            seed=base_kwargs["seed"],
            bf16=base_kwargs["bf16"],
            fp16=base_kwargs["fp16"],
        )

        try:
            import bitsandbytes as bnb  # noqa: F401
            ta_kwargs["optim"] = "adamw_bnb_8bit"
        except Exception:
            ta_kwargs["optim"] = "adamw_torch"
            print("bitsandbytes not found; using AdamW Torch optimizer.")

        training_args = TrainingArguments(**ta_kwargs)

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
    print(f"Saved artifacts under: {trainer.args.output_dir}")

if __name__ == "__main__":
    main()

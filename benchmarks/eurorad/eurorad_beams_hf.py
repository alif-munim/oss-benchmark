#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Eurorad evaluation using Hugging Face Transformers with model defaults,
except for the explicit changes present in the provided script:
- Force eager attention (disable flash & mem-efficient SDPA; enable math).
- Use left padding and set pad_token = eos_token for generation.
- Run deterministic diverse beam search:
    num_beams=13, num_beam_groups=13, diversity_penalty=0.5,
    num_return_sequences=13, early_stopping=True, do_sample=False,
    max_new_tokens=3000.
- Max input length = 4096.

Notes
- No Unsloth.
- No LoRA by default (add --lora_path to attach one).
- Iteratively appends one row per case to the output CSV (safe with --resume).

This version updates ONLY:
- Prompting: uses Harmony-friendly system+user messages via the tokenizer’s chat template
  and integrates your original prompt content as a structured “Clinical Reasoning Summary”
  with numbered sections to encourage fuller traces.
- Extraction: Harmony-channel-aware final choice extraction with verbatim mapping.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "2")
os.environ["HF_PYTORCH_ATTENTION_BACKEND"] = "eager"  # force eager attention

from torch.backends.cuda import sdp_kernel
sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)

import re
import json
import argparse
from datetime import datetime
from collections import Counter
import unicodedata

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ------------------------- Optional PEFT (OFF by default) -------------------------
def maybe_attach_lora(model, lora_path: str | None):
    if not lora_path:
        return model, False
    try:
        from peft import PeftModel
    except Exception as e:
        raise RuntimeError("peft is required to attach a LoRA adapter.") from e
    model = PeftModel.from_pretrained(model, lora_path)
    return model.eval(), True


# ------------------------- Text utils -------------------------
def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = text.lower().strip()
    t = re.sub(r"[^\w\s-]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.replace("–", "-",)


# ------------------------- Harmony-aware extraction -------------------------
H_FINAL_RE = re.compile(
    r"<\|channel\|\>\s*final\s*<\|message\|\>(.*?)<\|return\|>",
    re.IGNORECASE | re.DOTALL,
)
H_ANALYSIS_RE = re.compile(
    r"<\|channel\|\>\s*analysis\s*<\|message\|\>(.*?)(?:<\|return\||$)",
    re.IGNORECASE | re.DOTALL,
)

def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "").strip()

def _clean_candidate(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = s.strip('"\''"“”‘’")
    if s.startswith("<") and s.endswith(">") and "\n" not in s:
        s = s[1:-1].strip()
    s = re.sub(r"^[\-\•\*\#]\s*", "", s)
    s = re.sub(r"(?:<\|endoftext\|>)+$", "", s).strip()
    return s

def extract_harmony_blocks(text: str) -> tuple[str, str]:
    """Return (analysis_text, final_candidate) if present; else empty strings."""
    a = H_ANALYSIS_RE.findall(text)
    f = H_FINAL_RE.findall(text)
    analysis = _clean_candidate(a[-1]) if a else ""
    final = _clean_candidate(f[-1]) if f else ""
    return analysis, final

def pick_verbatim(candidate: str, options: list[str]) -> str:
    """Return the exact option string if candidate matches after NFC normalization; else ''."""
    if not candidate:
        return ""
    norm2orig = {_nfc(o): o for o in options}
    return norm2orig.get(_nfc(candidate), "")

def best_line_match(text: str, options: list[str]) -> str:
    """Scan lines for a direct verbatim match (exact line equals an option)."""
    opts = set(options)
    for ln in (l.strip() for l in text.splitlines() if l.strip()):
        if ln in opts:
            return ln
    return ""

def extract_after_assistantfinal(gen_text: str, options: list[str]) -> str:
    """
    Strategy:
      1) Grab Harmony <final> block; map to verbatim option via NFC.
      2) If invalid/missing, look for any exact option line anywhere.
      3) If still missing, return ''.
    Always return the option string EXACTLY as in options (preserving dashes/diacritics).
    """
    analysis, final_candidate = extract_harmony_blocks(gen_text)
    verbatim = pick_verbatim(final_candidate, options)
    if verbatim:
        return verbatim
    verbatim = best_line_match(gen_text, options)
    if verbatim:
        return verbatim
    verbatim = best_line_match(analysis, options)
    if verbatim:
        return verbatim
    if final_candidate:
        verbatim = pick_verbatim(final_candidate, options)
        if verbatim:
            return verbatim
    return ""


# ------------------------- Prompt (Harmony-friendly, fuller analysis) -------------------------
def build_messages_for_case(combined_description: str, dd_list: list[str]) -> list[dict]:
    # System: strict channeling + richer analysis request using your original sections
    system_msg = (
        "You are an expert radiologist. Respond using Harmony channels:\n"
        "- Put all reasoning in the analysis channel.\n"
        "- In the final channel output exactly ONE diagnosis, verbatim from the provided list,\n"
        "  with no quotes, no brackets, no punctuation, no extra text.\n"
        "- Do not repeat the list. Do not add any other channels after final.\n\n"
        "In the analysis channel, produce a structured Clinical Reasoning Summary (~150–300 words),\n"
        "organized in four numbered parts (do not reveal inner thoughts beyond these summaries):\n"
        "1) Connect symptoms to findings — link clinical presentation with imaging observations.\n"
        "2) Map to differentials — succinctly state how key findings support or contradict EACH option.\n"
        "3) Systematic elimination — briefly rule out less likely options based on conflicting evidence.\n"
        "4) Converge to answer — justify the single best diagnosis from the list.\n"
    )

    # User: your case + verbatim list to copy from
    user_msg = (
        "Case presentation:\n\n"
        f"{combined_description}\n\n"
        "Differential diagnoses (choose ONE verbatim; copy the line exactly):\n"
        + "\n".join(dd_list)
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

def create_detailed_prompt(case) -> str:
    """
    Kept the original function name for compatibility.
    Now returns the (system,user) message objects used by the chat template.
    """
    combined_description = case["combined_description"]
    dd_list = case["differential_diagnosis"]
    messages = build_messages_for_case(combined_description, dd_list)
    return messages  # type: ignore[return-value]

def apply_chat_template_or_fallback(tokenizer, case) -> str:
    """
    Apply tokenizer's chat template; otherwise use a strict fallback text format
    that mirrors the Harmony channels and structured analysis request.
    """
    messages = create_detailed_prompt(case)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    combined_description = case["combined_description"]
    dd_list = case["differential_diagnosis"]
    fallback = (
        "SYSTEM:\n"
        "You are an expert radiologist. Produce two blocks:\n"
        "1) ANALYSIS — a structured Clinical Reasoning Summary (~150–300 words) with these sections:\n"
        "   (1) Connect symptoms to findings; (2) Map to differentials; (3) Systematic elimination; (4) Converge to answer.\n"
        "   Keep it concise and evidence-based.\n"
        "2) FINAL — exactly ONE diagnosis, verbatim from the list (no quotes/brackets/punctuation/extra text).\n\n"
        "USER:\n"
        "Case presentation:\n\n"
        f"{combined_description}\n\n"
        "Differential diagnoses (choose ONE verbatim; copy the line exactly):\n"
        + "\n".join(dd_list)
        + "\n\nASSISTANT:\n"
    )
    return fallback


# ------------------------- Data prep -------------------------
def prepare_test_data(df: pd.DataFrame):
    processed = []
    for _, row in df.iterrows():
        full_desc = str(row["PostDescription"])
        dd = [d.strip() for d in str(row["DifferentialDiagnosisList"]).split(",")]
        processed.append(
            {
                "case_id": row["case_id"],
                "combined_description": full_desc,
                "differential_diagnosis": dd,
                "diagnosis": str(row["FinalDiagnosis"]),
            }
        )
    return processed


# ------------------------- CSV I/O -------------------------
def append_row_to_csv(out_csv: str, row: dict, header_order: list[str] | None):
    df = pd.DataFrame([row])
    write_header = not os.path.exists(out_csv)
    if header_order:
        cols = [c for c in header_order if c in df.columns] + [c for c in df.columns if c not in header_order]
        df = df[cols]
    df.to_csv(out_csv, index=False, mode="a", header=write_header, encoding="utf-8")


def load_processed_ids_if_resuming(out_csv: str) -> set:
    if not os.path.exists(out_csv):
        return set()
    try:
        done = pd.read_csv(out_csv, usecols=["case_id"])["case_id"].astype(str).tolist()
    except Exception:
        done = pd.read_csv(out_csv)["case_id"].astype(str).tolist()
    return set(done)


# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser(description="Eurorad evaluation (HF Transformers, diverse beams, iterative CSV).")
    ap.add_argument("--input_csv", required=True, help="Path to input CSV with cases.")
    ap.add_argument("--output_csv", required=True, help="Path to output CSV (rows appended per case).")
    ap.add_argument("--model", default="openai/gpt-oss-20b", help="HF model id or local path.")
    ap.add_argument("--lora_path", default="", help="Optional PEFT LoRA path (OFF by default).")
    ap.add_argument("--max_seq_len", type=int, default=4096, help="Max input tokens (matches original).")
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="Skip cases already present in output CSV.")
    # Decoding params (exactly as original)
    ap.add_argument("--num_beams", type=int, default=13)
    ap.add_argument("--num_beam_groups", type=int, default=13)
    ap.add_argument("--diversity_penalty", type=float, default=0.5)
    ap.add_argument("--max_new_tokens", type=int, default=3000)
    args = ap.parse_args()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"  # explicit change from original

    # Model (no explicit external quant config; load as published)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=None,   # model default dtype
    )
    model, lora_used = maybe_attach_lora(model, args.lora_path if args.lora_path else None)
    model.eval()
    print(f"Model loaded. LoRA attached: {lora_used}. Devices: {sorted({str(p.device) for p in model.parameters()})}")

    # Data
    df = pd.read_csv(args.input_csv)
    required = {"case_id", "PostDescription", "DifferentialDiagnosisList", "FinalDiagnosis"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")
    test_data = prepare_test_data(df)
    print(f"Prepared {len(test_data)} cases")

    # Resume
    already_done = load_processed_ids_if_resuming(args.output_csv) if args.resume else set()
    if already_done:
        print(f"Resume: found {len(already_done)} completed case_ids in {args.output_csv}")

    # Stable header
    header_order = [
        "case_id", "actual_sample_number", "prompt_type", "decoding_method",
        "num_beams", "num_beam_groups", "diversity_penalty", "ground_truth",
        "available_options", "user_prompt", "input_token_count",
        "final_chosen_answer", "final_chosen_votes", "final_chosen_from_beam",
        "correct", "all_beam_extracted_answers",
    ] + sum(([f"beam_{i}_text", f"beam_{i}_final"] for i in range(1, args.num_beams + 1)), [])

    acc = []

    for idx, case in enumerate(test_data[args.start_index:], start=args.start_index):
        case_id = str(case["case_id"])
        if args.resume and case_id in already_done:
            print(f"[{idx+1}/{len(test_data)}] Skip (already processed): case_id={case_id}")
            continue

        gt = str(case["diagnosis"]).strip()

        # Harmony-friendly prompt via chat template (or strict fallback)
        formatted = apply_chat_template_or_fallback(tokenizer, case)

        # Tokenize
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_seq_len,
        )
        inputs = {k: v.to(model.device, non_blocking=True) for k, v in inputs.items()}
        input_token_count = int(inputs["input_ids"].shape[1])

        # Deterministic diverse beams (unchanged)
        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            num_beam_groups=args.num_beam_groups,
            diversity_penalty=args.diversity_penalty,
            num_return_sequences=args.num_beams,
            early_stopping=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        print(f"[{idx+1}/{len(test_data)}] case_id={case_id} | beams={args.num_beams} groups={args.num_beam_groups} div={args.diversity_penalty}")
        with torch.no_grad():
            sequences = model.generate(**inputs, **gen_kwargs)

        # Decode each beam (exclude prompt tokens)
        prompt_len = input_token_count
        beam_texts, beam_finals = [], []
        options = case["differential_diagnosis"]
        for i in range(args.num_beams):
            gen_tokens = sequences[i][prompt_len:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
            beam_texts.append(gen_text.strip())
            beam_finals.append(extract_after_assistantfinal(gen_text, options))

        # Majority vote (normalize); tie -> earliest (unchanged)
        norm = [normalize_text(x) for x in beam_finals]
        counts = Counter(norm)
        if counts:
            max_count = max(counts.values())
            tie_norms = {a for a, c in counts.items() if c == max_count}
            chosen_idx = next((i for i, a in enumerate(norm) if a in tie_norms), 0)
            chosen_answer, chosen_votes = beam_finals[chosen_idx], int(max_count)
        else:
            chosen_idx, chosen_answer, chosen_votes = 0, (beam_finals[0] if beam_finals else ""), 0

        chosen_from_beam = chosen_idx + 1
        is_correct = int(normalize_text(chosen_answer) == normalize_text(gt))
        acc.append(is_correct)

        # Row
        user_prompt_for_logging = (
            "Case presentation:\n\n"
            f"{case['combined_description']}\n\n"
            "Differential diagnoses (choose ONE verbatim; copy the line exactly):\n"
            + "\n".join(options)
        )
        row = {
            "case_id": case_id,
            "actual_sample_number": idx + 1,
            "prompt_type": "DETAILED_XML_TWO_BLOCKS",
            "decoding_method": "DIVERSE_BEAM_MAJORITY",
            "num_beams": args.num_beams,
            "num_beam_groups": args.num_beam_groups,
            "diversity_penalty": args.diversity_penalty,
            "ground_truth": gt,
            "available_options": " | ".join(options),
            "user_prompt": user_prompt_for_logging,
            "input_token_count": input_token_count,
            "final_chosen_answer": chosen_answer,
            "final_chosen_votes": chosen_votes,
            "final_chosen_from_beam": chosen_from_beam,
            "correct": is_correct,
            "all_beam_extracted_answers": json.dumps(beam_finals, ensure_ascii=False),
        }
        for i in range(args.num_beams):
            row[f"beam_{i+1}_text"] = beam_texts[i]
            row[f"beam_{i+1}_final"] = beam_finals[i]

        append_row_to_csv(args.output_csv, row, header_order)

        mean_acc = np.mean(acc)
        std_acc = np.std(acc) if len(acc) > 1 else 0.0
        print(f"   → chosen (beam {chosen_from_beam}, votes={chosen_votes}): {chosen_answer or '[EMPTY]'} | correct={bool(is_correct)} | running acc={mean_acc:.3f} ± {std_acc:.3f}")

    print("\nDone.")
    if acc:
        print(f"Final Majority Accuracy: {np.mean(acc):.3f} ± {np.std(acc) if len(acc)>1 else 0.0:.3f}")
    print(f"Output CSV: {args.output_csv}")


if __name__ == "__main__":
    main()

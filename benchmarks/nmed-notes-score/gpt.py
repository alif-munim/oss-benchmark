#!/usr/bin/env python3
# benchmarks/gpt_responses.py
#
# Likert scoring (1–5, .5 allowed) via Responses API with reasoning=medium (default).
# - Diagnosis or treatment rubric (--eval-mode)
# - Resume-safe incremental writes
# - Outputs CSV (+ JSONL mirror) with raw text, score, model, effort, usage note
#
# Requires: pip install openai pandas tqdm

import csv
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from tqdm import tqdm
import re
import sys

# ---- Defaults ----
DEFAULT_DATASET = Path("/home/bowang/Documents/alif/oss-benchmark/data/datasets/diagnosis_task.csv")
DEFAULT_RESULTS = Path("/home/bowang/Documents/alif/oss-benchmark/results")
DEFAULT_MODEL = "gpt-5-2025-08-07"  # reasoning-capable; accepts reasoning={"effort": ...}
DEFAULT_TEXT_COLUMN = "Disease_description"

# ---- Prompts ----
DIAG_PROMPT = """You are asked to evaluate the quality of a model’s diagnostic output using the following rubric:

**Scoring Rubric (Likert scale 1–5):**
1. Most relevant options not mentioned.
2. Some or many relevant options not mentioned.
3. Most relevant options mentioned.
4. Most relevant options mentioned.
5. All relevant options mentioned.

**Instruction:**  
Given the following task description, the true disease, and the model output, assign a single integer score from 1 to 5 according to the rubric. Half-point scores (e.g., 1.5, 2.5, 3.5, 4.5) are allowed if the quality falls between two rubric levels.
Output **only the score**, with no explanation or justification.

**Inputs**:
"""

TREAT_PROMPT = """You are asked to evaluate the quality of a model’s treatment suggestion output using the following rubric:

**Scoring Rubric (Likert scale 1–5):**
1. All or most suggested options redundant or unjustified.  
2. Some suggested options redundant or unjustified.  
3. Some suggested options redundant or unjustified.  
4. Few suggested options redundant or unjustified.  
5. No suggested options redundant or unjustified.  

**Instruction:**  
Given the following task description, the true disease, and the model output, assign a score from 1 to 5 according to the rubric. Half-point scores (e.g., 1.5, 2.5, 3.5, 4.5) are allowed if the quality falls between two rubric levels.
Output **only the score**, with no explanation or justification.

**Inputs**:
"""

def get_eval_prompt(mode: str) -> str:
    return DIAG_PROMPT if mode == "diagnosis" else TREAT_PROMPT

# ---- Parsing helpers ----
SCORE_RE = re.compile(r"\b([1-5](?:\.5|\.0)?)\b")
FINAL_CH_RE = re.compile(r"<\|channel\|>final<\|message\|>(.*)$", re.DOTALL)

def extract_final(text: str) -> str:
    if not text:
        return ""
    m = FINAL_CH_RE.search(text)
    return (m.group(1).strip() if m else text.strip())

def parse_score(raw: str) -> Optional[str]:
    if not raw:
        return None
    t = extract_final(raw)
    m = SCORE_RE.search(t)
    return m.group(1) if m else None

def is_done_score(s: str) -> bool:
    return isinstance(s, str) and bool(SCORE_RE.fullmatch((s.strip() or "")))

def build_user_message(row: dict, text_column: str, eval_prompt: str) -> str:
    body = row.get(text_column, "")
    if not isinstance(body, str):
        body = str(body)
    return eval_prompt + body

def out_path(dataset_csv: Path, model_tag: str, results_dir: Path, eval_mode: str, effort: Optional[str] = None) -> Path:
    parts = [dataset_csv.stem, model_tag, "responses", eval_mode, "eval"]
    if effort:
        parts.append(f"re-{effort}")
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / ("_".join(parts) + ".csv")

# ---- IO / resume helpers ----
def load_csv(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_rows(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f_out:
        w = csv.DictWriter(f_out, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def merge_fieldnames(dataset_rows: List[dict], extra_cols: List[str]) -> List[str]:
    base = list(dataset_rows[0].keys()) if dataset_rows else []
    return base + [c for c in extra_cols if c not in base]

def load_resume_state(out_csv: Path, n_rows: int) -> List[dict]:
    if not out_csv.exists():
        return []
    try:
        prev_rows = load_csv(out_csv)
        if len(prev_rows) < n_rows:
            pad = [{} for _ in range(n_rows - len(prev_rows))]
            prev_rows = prev_rows + pad
        elif len(prev_rows) > n_rows:
            prev_rows = prev_rows[:n_rows]
        return prev_rows
    except Exception:
        return []

def row_done(prev_row: dict) -> bool:
    if not prev_row:
        return False
    sc = prev_row.get("LLM_eval_score", "")
    return is_done_score(sc)

# -------------------------
# Responses API (reasoning models)
# -------------------------
def ask_model_responses(client: OpenAI, model: str, user_text: str, effort: str, debug: bool) -> Tuple[str, str]:
    resp = client.responses.create(
        model=model,
        reasoning={"effort": effort},  # default = medium via CLI default
        input=[{"role": "user", "content": user_text}],
    )
    if debug:
        print("RAW RESPONSE JSON:")
        print(resp.model_dump_json(indent=2))
    answer = (resp.output_text or "").strip()

    note = ""
    usage = getattr(resp, "usage", None)
    if usage is not None:
        utd = getattr(usage, "output_tokens_details", None)
        if utd is not None and hasattr(utd, "reasoning_tokens"):
            note = f"reasoning_tokens={utd.reasoning_tokens}, output_tokens={usage.output_tokens}"
    return answer, note

def run_responses(dataset_csv: Path, results_dir: Path, text_column: str, model: str, effort: str,
                  resume: bool, output_csv: Optional[Path], debug: bool, eval_mode: str):
    client = OpenAI()
    rows = load_csv(dataset_csv)
    if not rows:
        print("Empty dataset.")
        sys.exit(1)

    eval_prompt = get_eval_prompt(eval_mode)
    out_csv = output_csv or out_path(dataset_csv, model, results_dir, eval_mode, effort)
    extra = ["LLM_eval_raw", "LLM_eval_score", "Model", "Reasoning Effort", "Note"]
    fieldnames = merge_fieldnames(rows, extra)

    prev_rows = load_resume_state(out_csv, len(rows)) if resume else []
    out_rows: List[dict] = []

    for i, row in enumerate(tqdm(rows, desc=f"Processing (Responses {model}, {eval_mode}, re={effort})", unit="row")):
        if prev_rows and row_done(prev_rows[i]):
            merged = {**row, **prev_rows[i]}
            out_rows.append(merged)
            continue

        user_text = build_user_message(row, text_column, eval_prompt)
        raw, note = ask_model_responses(client, model, user_text, effort, debug)
        score = parse_score(raw)

        merged = dict(row)
        merged["LLM_eval_raw"] = raw
        merged["LLM_eval_score"] = "" if score is None else score
        merged["Model"] = model
        merged["Reasoning Effort"] = effort
        merged["Note"] = note
        out_rows.append(merged)

        write_rows(out_csv, fieldnames, out_rows)  # incremental

    write_rows(out_csv, fieldnames, out_rows)

    jsonl_path = out_csv.with_suffix(".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
        for r in out_rows:
            f_jsonl.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved → {out_csv}")
    print(f"Saved → {jsonl_path}")

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Reasoning-capable model (e.g., gpt-5-2025-08-07).")
    ap.add_argument("--effort", choices=["low", "medium", "high"], default="medium",
                    help="Reasoning effort (default: medium).")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to dataset CSV.")
    ap.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN, help="Column with the task text.")
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS, help="Directory to write outputs.")
    ap.add_argument("--resume", action="store_true", help="Resume from existing output (skip scored rows).")
    ap.add_argument("--output-csv", type=Path, default=None, help="Explicit output path (used with --resume).")
    ap.add_argument("--eval-mode", choices=["diagnosis", "treatment"], default="diagnosis",
                    help="Selects rubric/prompt.")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    run_responses(args.dataset, args.results_dir, args.text_column, args.model, args.effort,
                  args.resume, args.output_csv, args.debug, args.eval_mode)

if __name__ == "__main__":
    main()

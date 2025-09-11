#!/usr/bin/env python3
# benchmarks/ophthalmology_responses.py
#
# Ophthalmology MCQ evaluation via Responses API (reasoning models only).
# - Uses reasoning={"effort":"medium"} by default.
# - Supports --resume to skip already-scored rows (by RowID).
# - Writes CSV and a JSONL mirror.
#
# Dataset columns expected:
#   - Question  (string with the formatted MCQ)
#   - (optional) ID  (used to derive stable RowID; else row index is used)
#
# Output columns added:
#   RowID, Model Answer, Model, Reasoning Effort, Note
#
# Usage example:
#   python benchmarks/ophthalmology_responses.py \
#     --dataset-csv /path/to/ophthalmology.csv \
#     --results-dir /path/to/results \
#     --output-csv ophthal_gpt5_medium.csv \
#     --resume

import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
from openai import OpenAI
from tqdm import tqdm

# ---- Defaults ----
DEFAULT_DATASET = Path("/home/bowang/Documents/alif/oss-benchmark/data/datasets/ophthalmology.csv")
DEFAULT_RESULTS_DIR = Path("/home/bowang/Documents/alif/oss-benchmark/results")
DEFAULT_MODEL = "gpt-5-2025-08-07"   # reasoning-capable
DEFAULT_EFFORT = "medium"            # low | medium | high

ROW_ID_COL = "RowID"
ADD_COLS = ["Model Answer", "Model", "Reasoning Effort", "Note"]

SYS_PROMPT = (
    "You are a careful ophthalmology assistant.\n"
    "You will be given a multiple-choice case with options labeled A–Z.\n"
    "Some questions may have multiple correct answers, others only one.\n"
    "Select ALL correct answers. Respond with ONLY the capital letters (A–Z), "
    "concatenated without spaces (e.g., 'A', 'BD', 'ACE').\n"
    "Do not include any explanation or words, only the letters."
)

# -------------------------
# Utilities
# -------------------------
def compute_row_id(idx_zero_based: int, row: Dict[str, str]) -> str:
    rid = str(row.get("ID", "")).strip()
    return rid if rid else f"row-{idx_zero_based:06d}"

def ensure_fieldnames(dataset_fieldnames: List[str]) -> List[str]:
    out = list(dataset_fieldnames)
    if ROW_ID_COL not in out:
        out.append(ROW_ID_COL)
    for c in ADD_COLS:
        if c not in out:
            out.append(c)
    return out

def load_dataset_rows(dataset_csv: Path) -> List[Dict[str, str]]:
    with open(dataset_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def read_existing_results(out_csv: Path) -> Dict[str, Dict[str, str]]:
    if not out_csv.exists():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    with open(out_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_rowid = ROW_ID_COL in (reader.fieldnames or [])
        for r in reader:
            rid = (r.get(ROW_ID_COL) or r.get("ID") or "").strip() if has_rowid else (r.get("ID") or r.get(ROW_ID_COL) or "").strip()
            if rid:
                out[rid] = dict(r)
    return out

def should_skip_existing(row_dict: Dict[str, str], rerun_errors: bool) -> bool:
    ans = (row_dict.get("Model Answer") or "").strip()
    if not ans:
        return False
    if rerun_errors and ans.startswith("ERROR:"):
        return False
    return True

def build_final_rows_in_dataset_order(
    dataset_rows: List[Dict[str, str]],
    existing_by_id: Dict[str, Dict[str, str]],
    new_by_id: Dict[str, Dict[str, str]],
) -> Iterable[Dict[str, str]]:
    for i, ds_row in enumerate(dataset_rows):
        rid = compute_row_id(i, ds_row)
        base = dict(ds_row)
        base[ROW_ID_COL] = rid
        if rid in existing_by_id:
            merged = dict(base)
            merged.update(existing_by_id[rid])
            yield merged
        elif rid in new_by_id:
            merged = dict(base)
            merged.update(new_by_id[rid])
            yield merged
        else:
            for c in ADD_COLS:
                base.setdefault(c, "" if c != "Model" else "")
            yield base

def write_csv(out_csv: Path, fieldnames: List[str], rows: Iterable[Dict[str, str]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

def write_jsonl_from_csv(csv_path: Path) -> None:
    jsonl_path = csv_path.with_suffix(".jsonl")
    with open(csv_path, newline="", encoding="utf-8") as f_in, \
         open(jsonl_path, "w", encoding="utf-8") as f_out:
        for r in csv.DictReader(f_in):
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

# -------------------------
# Responses API (reasoning models)
# -------------------------
def ask_model_responses(client: OpenAI, model: str, question: str, effort: str, debug: bool):
    resp = client.responses.create(
        model=model,
        reasoning={"effort": effort},  # "low" | "medium" | "high"
        input=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": question},
        ],
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

def run_responses_mode(
    dataset_csv: Path,
    out_csv: Path,
    model: str,
    effort: str,
    resume: bool,
    rerun_errors: bool,
    debug: bool
) -> None:
    client = OpenAI()
    ds_rows = load_dataset_rows(dataset_csv)
    if not ds_rows:
        raise RuntimeError("Empty dataset CSV.")

    fieldnames = ensure_fieldnames(list(ds_rows[0].keys()))
    existing = read_existing_results(out_csv) if resume else {}
    new_results: Dict[str, Dict[str, str]] = {}

    with tqdm(total=len(ds_rows), desc=f"Processing (Responses {model}, re={effort})", unit="q") as bar:
        for i, row in enumerate(ds_rows):
            rid = compute_row_id(i, row)
            if resume and rid in existing and should_skip_existing(existing[rid], rerun_errors):
                bar.update(1)
                continue

            q = row.get("Question", "")
            if q:
                ans, note = ask_model_responses(client, model, q, effort, debug)
            else:
                ans, note = "", "no_question_text"

            new_results[rid] = {
                ROW_ID_COL: rid,
                "Model Answer": ans,
                "Model": model,
                "Reasoning Effort": effort,
                "Note": note,
            }
            bar.update(1)

    final_rows = build_final_rows_in_dataset_order(ds_rows, existing, new_results)
    write_csv(out_csv, fieldnames, final_rows)
    write_jsonl_from_csv(out_csv)

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Reasoning-capable model (e.g., gpt-5-2025-08-07 or gpt-5).")
    ap.add_argument("--effort", choices=["low", "medium", "high"], default=DEFAULT_EFFORT,
                    help="Reasoning effort (default: medium).")
    ap.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET, help="Path to the input dataset CSV.")
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                    help="Directory for outputs when --output-csv is not specified.")
    ap.add_argument("--output-csv", "--out-csv", dest="output_csv", type=Path, default=None,
                    help="Exact output CSV path/name. If relative, placed under --results-dir. .csv suffix auto-added.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from existing output: skip rows with a non-empty 'Model Answer'.")
    ap.add_argument("--rerun-errors", action="store_true",
                    help="When resuming, re-run rows whose 'Model Answer' starts with 'ERROR:'.")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # Resolve output path
    if args.output_csv is None:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = args.dataset_csv.stem
        out_csv = args.results_dir / f"{dataset_name}_{args.model}_responses_re-{args.effort}.csv"
    else:
        out_csv = args.output_csv
        if out_csv.suffix.lower() != ".csv":
            out_csv = out_csv.with_suffix(".csv")
        if not out_csv.is_absolute():
            out_csv = args.results_dir / out_csv
        out_csv.parent.mkdir(parents=True, exist_ok=True)

    run_responses_mode(
        dataset_csv=args.dataset_csv,
        out_csv=out_csv,
        model=args.model,
        effort=args.effort,
        resume=args.resume,
        rerun_errors=args.rerun_errors,
        debug=args.debug,
    )

if __name__ == "__main__":
    main()

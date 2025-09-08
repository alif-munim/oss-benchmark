#!/usr/bin/env python3
# benchmarks/gpt.py
# Modes:
#   responses -> gpt-5 via Responses API (reasoning effort)
#   batch     -> gpt-4o via Batch API (all questions)
#   chat      -> Chat Completions API (e.g., gpt-5 or gpt-4o) over all questions
#
# New:
#   - --out-csv lets you set the exact output CSV path/name.
#   - --resume skips rows already present in --out-csv (non-empty "Model Answer").
#   - Robust RowID handling for safe merging/resume across modes.
#   - Optional --dataset-csv override and --results-dir base folder (used when --out-csv not given).
#   - Batch mode only submits remaining rows when --resume is set.
#   - Optional --batch-id to only poll/merge an existing batch result.

import csv
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
from openai import OpenAI
from tqdm import tqdm

# ---- Defaults (override via CLI) ----
DEFAULT_DATASET = Path("/home/bowang/Documents/alif/oss-benchmark/data/datasets/ophthalmology.csv")
DEFAULT_RESULTS_DIR = Path("/home/bowang/Documents/alif/oss-benchmark/results")
MODEL_GPT5 = "gpt-5"
MODEL_GPT4O = "gpt-4o"
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
    if rid:
        return rid
    # Stable synthetic ID based on dataset order
    return f"row-{idx_zero_based:06d}"

def ensure_fieldnames(dataset_fieldnames: List[str]) -> List[str]:
    # Add RowID + extra columns if missing
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
        # allow fallback to "ID" if RowID wasn't present in an older file
        has_rowid = ROW_ID_COL in (reader.fieldnames or [])
        for r in reader:
            rid = (r.get(ROW_ID_COL) or r.get("ID") or "").strip()
            if not rid:
                continue
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
            # Use existing result row (already contains add cols)
            merged = dict(base)
            merged.update(existing_by_id[rid])
            yield merged
        elif rid in new_by_id:
            merged = dict(base)
            merged.update(new_by_id[rid])
            yield merged
        else:
            # Not processed; emit base row with empty add cols
            for c in ADD_COLS:
                merged_val = "" if c != "Model" else ""
                base.setdefault(c, merged_val)
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
# Non-batch path (Responses API, gpt-5)
# -------------------------
def ask_model_responses(client: OpenAI, question: str, effort: str, debug: bool) -> Tuple[str, str]:
    resp = client.responses.create(
        model=MODEL_GPT5,
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
        if debug:
            print("USAGE:", {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "reasoning_tokens": getattr(utd, "reasoning_tokens", None) if utd else None,
            })
    return answer, note

def run_responses_mode(
    dataset_csv: Path,
    out_csv: Path,
    effort: str,
    resume: bool,
    rerun_errors: bool,
    debug: bool
) -> None:
    client = OpenAI()
    ds_rows = load_dataset_rows(dataset_csv)
    dataset_name = dataset_csv.stem

    # Determine fieldnames
    fieldnames = ensure_fieldnames(list(ds_rows[0].keys()) if ds_rows else [])

    existing = read_existing_results(out_csv) if resume else {}
    new_results: Dict[str, Dict[str, str]] = {}

    # Process only pending rows
    with tqdm(total=len(ds_rows), desc="Processing (Responses gpt-5)", unit="q") as bar:
        for i, row in enumerate(ds_rows):
            rid = compute_row_id(i, row)
            if resume and rid in existing and should_skip_existing(existing[rid], rerun_errors):
                bar.update(1)
                continue

            q = row.get("Question", "")
            if q:
                ans, note = ask_model_responses(client, q, effort, debug)
            else:
                ans, note = "", "no_question_text"

            row_out = {
                ROW_ID_COL: rid,
                "Model Answer": ans,
                "Model": MODEL_GPT5,
                "Reasoning Effort": effort,
                "Note": note,
            }
            new_results[rid] = row_out
            bar.update(1)

    # Merge and write
    final_rows = build_final_rows_in_dataset_order(ds_rows, existing, new_results)
    write_csv(out_csv, fieldnames, final_rows)
    write_jsonl_from_csv(out_csv)

# -------------------------
# Batch path (Chat Completions, gpt-4o)
# -------------------------
def build_batch_input_jsonl(pending: List[Tuple[str, Dict[str, str]]], jsonl_path: Path) -> None:
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rid, row in pending:
            body = {
                "model": MODEL_GPT4O,
                "messages": [
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": row.get("Question", "")},
                ],
            }
            obj = {
                "custom_id": f"id_{rid}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def submit_batch(client: OpenAI, jsonl_path: Path):
    batch_file = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return batch.id

def poll_batch_until_done(client: OpenAI, batch_id: str, poll_secs: float = 5.0):
    with tqdm(desc="Batch status", unit="poll") as bar:
        while True:
            b = client.batches.retrieve(batch_id)
            status = b.status
            bar.set_postfix_str(status)
            bar.update(1)
            if status in ("completed", "failed", "canceled", "expired"):
                return b
            time.sleep(poll_secs)

def download_batch_output(client: OpenAI, output_file_id: str) -> List[dict]:
    content = client.files.content(output_file_id)
    text = content.text
    out: List[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out

def extract_answer_from_choice(choice_obj: dict) -> str:
    try:
        msg = choice_obj["message"]
        return (msg.get("content") or "").strip()
    except Exception:
        return ""

def run_batch_mode(
    dataset_csv: Path,
    out_csv: Path,
    resume: bool,
    rerun_errors: bool,
    debug: bool,
    batch_id_cli: str = ""
) -> None:
    client = OpenAI()
    ds_rows = load_dataset_rows(dataset_csv)
    fieldnames = ensure_fieldnames(list(ds_rows[0].keys()) if ds_rows else [])

    existing = read_existing_results(out_csv) if resume else {}

    # Determine pending rows
    pending: List[Tuple[str, Dict[str, str]]] = []
    for i, row in enumerate(ds_rows):
        rid = compute_row_id(i, row)
        if resume and rid in existing and should_skip_existing(existing[rid], rerun_errors):
            continue
        pending.append((rid, row))

    outputs: List[dict] = []

    if batch_id_cli:
        # Only poll an existing batch id
        final_batch = poll_batch_until_done(client, batch_id_cli)
        if final_batch.status != "completed":
            raise RuntimeError(f"Batch not completed: {final_batch.status}")
        outputs = download_batch_output(client, final_batch.output_file_id)
    else:
        if pending:
            dataset_name = dataset_csv.stem
            batch_jsonl = out_csv.with_suffix("")  # base
            batch_jsonl = batch_jsonl.parent / f"{dataset_name}_{MODEL_GPT4O}_batch_input.jsonl"
            build_batch_input_jsonl(pending, batch_jsonl)
            new_batch_id = submit_batch(client, batch_jsonl)
            final_batch = poll_batch_until_done(client, new_batch_id)
            if final_batch.status != "completed":
                raise RuntimeError(f"Batch not completed: {final_batch.status}")
            outputs = download_batch_output(client, final_batch.output_file_id)
        else:
            # Nothing pending; just re-write csv/jsonl to normalize columns
            final_rows = build_final_rows_in_dataset_order(ds_rows, existing, {})
            write_csv(out_csv, fieldnames, final_rows)
            write_jsonl_from_csv(out_csv)
            return

    # Map answers
    answers: Dict[str, str] = {}
    for item in outputs:
        cid = item.get("custom_id")
        if item.get("error"):
            answers[cid] = f"ERROR: {item['error']}"
            continue
        resp = item.get("response", {})
        try:
            choices = resp["body"]["choices"]
            if choices:
                answers[cid] = extract_answer_from_choice(choices[0])
            else:
                answers[cid] = ""
        except Exception:
            answers[cid] = ""

    # Build new results for the pending set
    new_results: Dict[str, Dict[str, str]] = {}
    for i, row in enumerate(ds_rows):
        rid = compute_row_id(i, row)
        if resume and rid in existing and should_skip_existing(existing[rid], rerun_errors):
            continue
        cid = f"id_{rid}"
        if cid not in answers:
            continue
        ans = answers[cid]
        new_results[rid] = {
            ROW_ID_COL: rid,
            "Model Answer": ans,
            "Model": MODEL_GPT4O,
            "Reasoning Effort": "",
            "Note": "",
        }

    # Merge and write
    final_rows = build_final_rows_in_dataset_order(ds_rows, existing, new_results)
    write_csv(out_csv, fieldnames, final_rows)
    write_jsonl_from_csv(out_csv)

# -------------------------
# Chat Completions path (generic chat API; default gpt-5)
# -------------------------
def ask_model_chat(client: OpenAI, model: str, question: str, debug: bool) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    if debug:
        print("RAW CHAT RESPONSE OBJECT:")
        print(resp)
    content = resp.choices[0].message.content or ""
    return content.strip()

def run_chat_mode(
    dataset_csv: Path,
    out_csv: Path,
    chat_model: str,
    resume: bool,
    rerun_errors: bool,
    debug: bool
) -> None:
    client = OpenAI()
    ds_rows = load_dataset_rows(dataset_csv)
    fieldnames = ensure_fieldnames(list(ds_rows[0].keys()) if ds_rows else [])

    existing = read_existing_results(out_csv) if resume else {}
    new_results: Dict[str, Dict[str, str]] = {}

    with tqdm(total=len(ds_rows), desc=f"Processing (Chat {chat_model})", unit="q") as bar:
        for i, row in enumerate(ds_rows):
            rid = compute_row_id(i, row)
            if resume and rid in existing and should_skip_existing(existing[rid], rerun_errors):
                bar.update(1)
                continue
            q = row.get("Question", "")
            ans = ask_model_chat(client, chat_model, q, debug) if q else ""
            new_results[rid] = {
                ROW_ID_COL: rid,
                "Model Answer": ans,
                "Model": chat_model,
                "Reasoning Effort": "",
                "Note": "",
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
    ap.add_argument("--mode", choices=["responses", "batch", "chat"], default="batch",
                    help="responses=gpt-5 (Responses API); batch=gpt-4o Batch API; chat=Chat Completions API")
    ap.add_argument("--effort", choices=["low", "medium", "high"], default="low",
                    help="Reasoning effort for responses mode (gpt-5). Ignored in batch/chat modes.")
    ap.add_argument("--chat-model", default="gpt-5",
                    help="Model for chat mode (e.g., gpt-5, gpt-4o).")
    ap.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET,
                    help="Path to the input dataset CSV.")
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                    help="Directory for outputs when --out-csv is not specified.")
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="Exact output CSV path/name. If not set, an automatic name is used in --results-dir.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from existing --out-csv: skip rows with a non-empty 'Model Answer'.")
    ap.add_argument("--rerun-errors", action="store_true",
                    help="When resuming, re-run rows whose 'Model Answer' starts with 'ERROR:'.")
    ap.add_argument("--batch-id", default="",
                    help="Batch mode: poll/merge an existing batch by id (skips submit).")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    dataset_csv: Path = args.dataset_csv
    dataset_name = dataset_csv.stem

    # Determine default output path if not provided
    if args.out_csv is None:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        if args.mode == "responses":
            out_csv = args.results_dir / f"{dataset_name}_{MODEL_GPT5}_{args.effort}.csv"
        elif args.mode == "chat":
            out_csv = args.results_dir / f"{dataset_name}_{args.chat_model}_chat.csv"
        else:
            out_csv = args.results_dir / f"{dataset_name}_{MODEL_GPT4O}_batch.csv"
    else:
        out_csv = args.out_csv
        out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "responses":
        run_responses_mode(dataset_csv, out_csv, args.effort, args.resume, args.rerun_errors, args.debug)
    elif args.mode == "chat":
        run_chat_mode(dataset_csv, out_csv, args.chat_model, args.resume, args.rerun_errors, args.debug)
    else:
        run_batch_mode(dataset_csv, out_csv, args.resume, args.rerun_errors, args.debug, args.batch_id)

if __name__ == "__main__":
    main()

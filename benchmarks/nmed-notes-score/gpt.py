#!/usr/bin/env python3
# benchmarks/gpt.py
# Modes:
#   responses -> gpt-5 via Responses API (reasoning effort) over all rows
#   batch     -> gpt-4o via Batch API (all rows; resumable merge and output reuse)
#   chat      -> Chat Completions API (e.g., gpt-5 or gpt-4o) over all rows
#
# Task: Likert scoring (1–5) using your evaluation prompt.
# CSV must contain a text column (default: Disease_description).
# Output columns: LLM_eval_raw, LLM_eval_score, Model, Reasoning Effort, Note
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
MODEL_GPT5 = "gpt-5"
MODEL_GPT4O = "gpt-4o"
DEFAULT_TEXT_COLUMN = "Disease_description"

# ---- Evaluation prompt (verbatim) ----
EVAL_PROMPT = """You are asked to evaluate the quality of a model’s diagnostic output using the following rubric:

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

def build_user_message(row: dict, text_column: str) -> str:
    body = row.get(text_column, "")
    if not isinstance(body, str):
        body = str(body)
    return EVAL_PROMPT + body

def out_path(dataset_csv: Path, tag: str, results_dir: Path, mode_tag: str, effort: Optional[str] = None) -> Path:
    parts = [dataset_csv.stem, tag, mode_tag, "eval"]
    if effort:
        parts.append(f"re-{effort}")
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / ("_".join(parts) + ".csv")

def batch_io_paths(dataset_csv: Path, results_dir: Path) -> Tuple[Path, Path]:
    base = results_dir / dataset_csv.stem
    return Path(f"{base}_{MODEL_GPT4O}_batch_input.jsonl"), Path(f"{base}_{MODEL_GPT4O}_batch_output.jsonl")

# ---- Resume helpers (shared) ----
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
        # If lengths mismatch, we still try by index; pad/crop to n_rows
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
# Responses API (gpt-5)
# -------------------------
def ask_model_responses(client: OpenAI, user_text: str, effort: str, debug: bool) -> Tuple[str, str]:
    resp = client.responses.create(
        model=MODEL_GPT5,
        reasoning={"effort": effort},
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

def run_responses_mode(dataset_csv: Path, results_dir: Path, text_column: str, effort: str, resume: bool, output_csv: Optional[Path], debug: bool):
    client = OpenAI()
    rows = load_csv(dataset_csv)
    if not rows:
        print("Empty dataset.")
        sys.exit(1)

    out_csv = output_csv or out_path(dataset_csv, MODEL_GPT5, results_dir, "responses", effort)
    extra = ["LLM_eval_raw", "LLM_eval_score", "Model", "Reasoning Effort", "Note"]
    fieldnames = merge_fieldnames(rows, extra)

    prev_rows = load_resume_state(out_csv, len(rows)) if resume else []
    out_rows: List[dict] = []

    for i, row in enumerate(tqdm(rows, desc="Processing (Responses gpt-5)", unit="row")):
        if prev_rows and row_done(prev_rows[i]):
            # Keep previous output row intact
            merged = {**row, **prev_rows[i]}
            out_rows.append(merged)
            continue

        user_text = build_user_message(row, text_column)
        raw, note = ask_model_responses(client, user_text, effort, debug)
        score = parse_score(raw)

        merged = dict(row)
        merged["LLM_eval_raw"] = raw
        merged["LLM_eval_score"] = "" if score is None else score
        merged["Model"] = MODEL_GPT5
        merged["Reasoning Effort"] = effort
        merged["Note"] = note
        out_rows.append(merged)

        # incremental save
        write_rows(out_csv, fieldnames, out_rows)

    # final save
    write_rows(out_csv, fieldnames, out_rows)

    # optional JSONL mirror
    jsonl_path = out_csv.with_suffix(".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
        for r in out_rows:
            f_jsonl.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved → {out_csv}")
    print(f"Saved → {jsonl_path}")

# -------------------------
# Batch API (chat, gpt-4o)
# -------------------------
def build_batch_input_jsonl(rows: List[dict], text_column: str, jsonl_path: Path):
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            rid = str(row.get("ID", "") or row.get("case_id", "") or i)
            user_text = build_user_message(row, text_column)
            body = {
                "model": MODEL_GPT4O,
                "messages": [{"role": "user", "content": user_text}],
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
    return [json.loads(line) for line in text.splitlines() if line.strip()]

def run_batch_mode(dataset_csv: Path, results_dir: Path, text_column: str, resume: bool, output_csv: Optional[Path], debug: bool):
    client = OpenAI()
    rows = load_csv(dataset_csv)
    if not rows:
        print("Empty dataset.")
        sys.exit(1)

    out_csv = output_csv or out_path(dataset_csv, MODEL_GPT4O, results_dir, "batch")
    extra = ["LLM_eval_raw", "LLM_eval_score", "Model"]
    fieldnames = merge_fieldnames(rows, extra)

    batch_in_path, batch_out_path = batch_io_paths(dataset_csv, results_dir)

    # Reuse prior batch OUTPUT jsonl if present (resume merge only)
    if resume and batch_out_path.exists():
        if debug:
            print(f"Reusing existing batch output: {batch_out_path}")
        outputs = [json.loads(l) for l in batch_out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        # Build input and submit a new batch
        build_batch_input_jsonl(rows, text_column, batch_in_path)
        batch_id = submit_batch(client, batch_in_path)
        final_batch = poll_batch_until_done(client, batch_id)
        if final_batch.status != "completed":
            raise RuntimeError(f"Batch not completed: {final_batch.status}")
        outputs = download_batch_output(client, final_batch.output_file_id)
        # Persist outputs to allow resume of the merge step later
        with open(batch_out_path, "w", encoding="utf-8") as fo:
            for item in outputs:
                fo.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Map: custom_id -> raw text
    answers: Dict[str, str] = {}
    for item in outputs:
        cid = item.get("custom_id")
        if item.get("error"):
            answers[cid] = f"ERROR: {item['error']}"
            continue
        resp = item.get("response", {})
        try:
            choices = resp["body"]["choices"]
            raw = (choices[0]["message"].get("content") or "").strip() if choices else ""
        except Exception:
            raw = ""
        answers[cid] = raw

    # Resume-aware merge/write
    prev_rows = load_resume_state(out_csv, len(rows)) if resume else []
    out_rows: List[dict] = []

    for i, row in enumerate(tqdm(rows, desc="Merging batch outputs", unit="row")):
        rid = str(row.get("ID", "") or row.get("case_id", "") or i)
        cid = f"id_{rid}"

        if prev_rows and row_done(prev_rows[i]):
            merged = {**row, **prev_rows[i]}
            out_rows.append(merged)
            continue

        raw = answers.get(cid, "")
        score = parse_score(raw)

        merged = dict(row)
        merged["LLM_eval_raw"] = raw
        merged["LLM_eval_score"] = "" if score is None else score
        merged["Model"] = MODEL_GPT4O
        out_rows.append(merged)

        write_rows(out_csv, fieldnames, out_rows)  # incremental

    write_rows(out_csv, fieldnames, out_rows)

    jsonl_path = out_csv.with_suffix(".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
        for r in out_rows:
            f_jsonl.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved → {out_csv}")
    print(f"Saved → {jsonl_path}")
    if resume:
        print(f"(Resume artifacts) batch input: {batch_in_path}  |  batch output: {batch_out_path}")

# -------------------------
# Chat Completions (generic)
# -------------------------
def ask_model_chat(client: OpenAI, model: str, user_text: str, debug: bool) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_text}],
    )
    if debug:
        print("RAW CHAT RESPONSE OBJECT:")
        print(resp)
    return (resp.choices[0].message.content or "").strip()

def run_chat_mode(dataset_csv: Path, results_dir: Path, text_column: str, chat_model: str, resume: bool, output_csv: Optional[Path], debug: bool):
    client = OpenAI()
    rows = load_csv(dataset_csv)
    if not rows:
        print("Empty dataset.")
        sys.exit(1)

    out_csv = output_csv or out_path(dataset_csv, chat_model, results_dir, "chat")
    extra = ["LLM_eval_raw", "LLM_eval_score", "Model"]
    fieldnames = merge_fieldnames(rows, extra)

    prev_rows = load_resume_state(out_csv, len(rows)) if resume else []
    out_rows: List[dict] = []

    for i, row in enumerate(tqdm(rows, desc=f"Processing (Chat {chat_model})", unit="row")):
        if prev_rows and row_done(prev_rows[i]):
            merged = {**row, **prev_rows[i]}
            out_rows.append(merged)
            continue

        user_text = build_user_message(row, text_column)
        raw = ask_model_chat(client, chat_model, user_text, debug)
        score = parse_score(raw)

        merged = dict(row)
        merged["LLM_eval_raw"] = raw
        merged["LLM_eval_score"] = "" if score is None else score
        merged["Model"] = chat_model
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
    ap.add_argument("--mode", choices=["responses", "batch", "chat"], default="responses",
                    help="responses=gpt-5; batch=gpt-4o; chat=Chat Completions")
    ap.add_argument("--effort", choices=["low", "medium", "high"], default="low",
                    help="Reasoning effort for responses mode.")
    ap.add_argument("--chat-model", default="gpt-5", help="Model for chat mode (e.g., gpt-5, gpt-4o).")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to dataset CSV.")
    ap.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN, help="Column with the task text.")
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS, help="Directory to write outputs.")
    ap.add_argument("--resume", action="store_true", help="Resume from existing output (skip scored rows).")
    ap.add_argument("--output-csv", type=Path, default=None, help="Explicit output path (used with --resume).")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.mode == "responses":
        run_responses_mode(args.dataset, args.results_dir, args.text_column, args.effort, args.resume, args.output_csv, args.debug)
    elif args.mode == "chat":
        run_chat_mode(args.dataset, args.results_dir, args.text_column, args.chat_model, args.resume, args.output_csv, args.debug)
    else:
        run_batch_mode(args.dataset, args.results_dir, args.text_column, args.resume, args.output_csv, args.debug)

if __name__ == "__main__":
    main()

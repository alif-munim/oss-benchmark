#!/usr/bin/env python3
# benchmarks/gpt5.py
# Modes:
#   responses -> gpt-5 via Responses API (reasoning effort)
#   batch     -> gpt-4o via Batch API (all questions)
#   chat      -> Chat Completions API (e.g., gpt-5 or gpt-4o) over all questions

import csv
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List
from openai import OpenAI
from tqdm import tqdm

# ---- Paths / constants ----
DATASET_CSV = Path("/home/bowang/Documents/alif/oss-benchmark/data/datasets/ophthalmology.csv")
RESULTS_DIR = Path("/home/bowang/Documents/alif/oss-benchmark/results")
MODEL_GPT5 = "gpt-5"
MODEL_GPT4O = "gpt-4o"

SYS_PROMPT = (
    "You are a careful ophthalmology assistant.\n"
    "You will be given a multiple-choice case with options labeled A–Z.\n"
    "Some questions may have multiple correct answers, others only one.\n"
    "Select ALL correct answers. Respond with ONLY the capital letters (A–Z), "
    "concatenated without spaces (e.g., 'A', 'BD', 'ACE').\n"
    "Do not include any explanation or words, only the letters."
)

# -------------------------
# Non-batch path (Responses API, gpt-5)
# -------------------------
def ask_model_responses(client: OpenAI, question: str, effort: str, debug: bool):
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

    # Usage note (object attrs, not dict)
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

def run_responses_mode(effort: str, debug: bool):
    client = OpenAI()
    with open(DATASET_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    dataset_name = DATASET_CSV.stem
    out_csv = RESULTS_DIR / f"{dataset_name}_{MODEL_GPT5}_{effort}.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    add_cols = ["Model Answer", "Model", "Reasoning Effort", "Note"]
    fieldnames = list(rows[0].keys()) + [c for c in add_cols if c not in rows[0].keys()]

    with open(out_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in tqdm(rows, desc="Processing (Responses gpt-5)", unit="q"):
            q = row.get("Question", "")
            if q:
                ans, note = ask_model_responses(client, q, effort, debug)
            else:
                ans, note = "", "no_question_text"
            row_out = dict(row)
            row_out["Model Answer"] = ans
            row_out["Model"] = MODEL_GPT5
            row_out["Reasoning Effort"] = effort
            row_out["Note"] = note
            writer.writerow(row_out)

    jsonl_path = out_csv.with_suffix(".jsonl")
    with open(out_csv, newline="", encoding="utf-8") as f_copy, open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
        for r in csv.DictReader(f_copy):
            f_jsonl.write(json.dumps(r, ensure_ascii=False) + "\n")

    if debug:
        print(f"Wrote CSV → {out_csv}")
        print(f"Wrote JSONL → {jsonl_path}")

# -------------------------
# Batch path (Chat Completions, gpt-4o)
# -------------------------
def build_batch_input_jsonl(rows: List[Dict], jsonl_path: Path):
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            rid = str(row.get("ID", "")).strip() or f"row-{len(row)}"
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
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out

def extract_answer_from_choice(choice_obj: dict) -> str:
    try:
        msg = choice_obj["message"]
        return (msg.get("content") or "").strip()
    except Exception:
        return ""

def run_batch_mode(debug: bool):
    client = OpenAI()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATASET_CSV, newline="", encoding="utf-8") as f_in:
        rows = list(csv.DictReader(f_in))
    if not rows:
        raise RuntimeError("Empty dataset CSV.")

    dataset_name = DATASET_CSV.stem
    batch_jsonl = RESULTS_DIR / f"{dataset_name}_{MODEL_GPT4O}_batch_input.jsonl"
    build_batch_input_jsonl(rows, batch_jsonl)

    batch_id = submit_batch(client, batch_jsonl)
    final_batch = poll_batch_until_done(client, batch_id)
    if final_batch.status != "completed":
        raise RuntimeError(f"Batch not completed: {final_batch.status}")

    outputs = download_batch_output(client, final_batch.output_file_id)

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

    out_csv = RESULTS_DIR / f"{dataset_name}_{MODEL_GPT4O}_batch.csv"
    add_cols = ["Model Answer", "Model", "Reasoning Effort", "Note"]
    fieldnames = list(rows[0].keys()) + [c for c in add_cols if c not in rows[0].keys()]

    with open(DATASET_CSV, newline="", encoding="utf-8") as f_in, \
         open(out_csv, "w", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in tqdm(reader, desc="Merging batch outputs", unit="q"):
            rid = str(row.get("ID", "")).strip() or ""
            cid = f"id_{rid}"
            ans = answers.get(cid, "")
            row_out = dict(row)
            row_out["Model Answer"] = ans
            row_out["Model"] = MODEL_GPT4O
            row_out["Reasoning Effort"] = ""
            row_out["Note"] = ""
            writer.writerow(row_out)

    jsonl_path = out_csv.with_suffix(".jsonl")
    with open(out_csv, newline="", encoding="utf-8") as f_copy, open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
        for r in csv.DictReader(f_copy):
            f_jsonl.write(json.dumps(r, ensure_ascii=False) + "\n")

    if debug:
        print(f"Wrote CSV → {out_csv}")
        print(f"Wrote JSONL → {jsonl_path}")

# -------------------------
# Chat Completions path (generic chat API; default gpt-5)
# -------------------------
def ask_model_chat(client: OpenAI, model: str, question: str, debug: bool):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": question},
        ],
        # no temperature / max token args per user preference
    )
    if debug:
        print("RAW CHAT RESPONSE OBJECT:")
        print(resp)
    content = resp.choices[0].message.content or ""
    return content.strip()

def run_chat_mode(chat_model: str, debug: bool):
    client = OpenAI()
    with open(DATASET_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    dataset_name = DATASET_CSV.stem
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / f"{dataset_name}_{chat_model}_chat.csv"

    add_cols = ["Model Answer", "Model", "Reasoning Effort", "Note"]
    fieldnames = list(rows[0].keys()) + [c for c in add_cols if c not in rows[0].keys()]

    with open(out_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in tqdm(rows, desc=f"Processing (Chat {chat_model})", unit="q"):
            q = row.get("Question", "")
            ans = ask_model_chat(client, chat_model, q, debug) if q else ""
            row_out = dict(row)
            row_out["Model Answer"] = ans
            row_out["Model"] = chat_model
            row_out["Reasoning Effort"] = ""  # not applicable in chat mode
            row_out["Note"] = ""
            writer.writerow(row_out)

    jsonl_path = out_csv.with_suffix(".jsonl")
    with open(out_csv, newline="", encoding="utf-8") as f_copy, open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
        for r in csv.DictReader(f_copy):
            f_jsonl.write(json.dumps(r, ensure_ascii=False) + "\n")

    if debug:
        print(f"Wrote CSV → {out_csv}")
        print(f"Wrote JSONL → {jsonl_path}")

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
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.mode == "responses":
        run_responses_mode(args.effort, args.debug)
    elif args.mode == "chat":
        run_chat_mode(args.chat_model, args.debug)
    else:
        run_batch_mode(args.debug)

if __name__ == "__main__":
    main()

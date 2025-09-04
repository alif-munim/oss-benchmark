#!/usr/bin/env python3
# benchmarks/gpt.py
# Modes:
#   responses -> gpt-5 via Responses API (reasoning effort)
#   batch     -> gpt-4o via Batch API (all questions)
#   chat      -> Chat Completions API (e.g., gpt-5 or gpt-4o) over all rows
#
# Dataset schema expected (CSV headers):
#   case_id, OriginalDescription, PostDescription, DifferentialDiagnosisList, FinalDiagnosis
#
# Output adds:
#   Model Answer (verbatim option), Model Answer (raw), Match Type, Correct, Options

import csv
import json
import time
import argparse
import unicodedata
import difflib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from tqdm import tqdm

# ---- Defaults (override via CLI) ----
DEFAULT_DATASET = Path("/home/bowang/Documents/alif/oss-benchmark/data/datasets/eurorad_test.csv")
DEFAULT_RESULTS = Path("/home/bowang/Documents/alif/oss-benchmark/results")
MODEL_GPT5 = "gpt-5"
MODEL_GPT4O = "gpt-4o"

# -------------------------
# Prompt builder (per row)
# -------------------------
SYS_PROMPT = (
    "You are a careful radiology diagnosis selector.\n"
    "Given a clinical case description and a finite list of candidate diagnoses,\n"
    "choose the single most likely final diagnosis FROM THE LIST.\n"
    "Response rules:\n"
    "1) Output EXACTLY one option, copied VERBATIM from the list.\n"
    "2) Output ONLY the diagnosis text. No explanation. No punctuation. No quotes.\n"
)

def build_options_list(s: str) -> List[str]:
    # Split on commas and strip whitespace; keep internal punctuation
    opts = [o.strip() for o in (s or "").split(",") if o.strip()]
    # Deduplicate while preserving order
    seen = set()
    out = []
    for o in opts:
        if o not in seen:
            seen.add(o)
            out.append(o)
    return out

def norm_text(s: str) -> str:
    # Normalize for robust matching (case, hyphens, spaces, unicode)
    t = unicodedata.normalize("NFKC", s or "")
    t = t.replace("–", "-").replace("—", "-").replace("-", "-")  # various hyphens
    t = " ".join(t.strip().split())  # collapse whitespace
    return t.lower()

def map_to_option(raw_answer: str, options: List[str]) -> Tuple[str, str]:
    """
    Map model's raw answer to one of the options.
    Returns (mapped_option, match_type) where match_type ∈ {"exact", "normalized", "fuzzy", "no_match"}.
    """
    raw = (raw_answer or "").strip()
    if not raw:
        return "", "no_match"

    # Exact verbatim match
    if raw in options:
        return raw, "exact"

    # Normalized match
    norm2opt = {norm_text(o): o for o in options}
    if norm_text(raw) in norm2opt:
        return norm2opt[norm_text(raw)], "normalized"

    # Fuzzy (fallback): pick closest by ratio, but ONLY accept if reasonably close
    candidates = difflib.get_close_matches(raw, options, n=1, cutoff=0.8)
    if candidates:
        return candidates[0], "fuzzy"

    # Last resort: fuzzy on normalized strings
    norm_options = list(norm2opt.keys())
    norm_candidates = difflib.get_close_matches(norm_text(raw), norm_options, n=1, cutoff=0.9)
    if norm_candidates:
        return norm2opt[norm_candidates[0]], "fuzzy"

    return "", "no_match"

def build_user_prompt(case_id: str, post_desc: str, options: List[str]) -> str:
    opts_block = "\n".join(f"- {o}" for o in options)
    return (
        f"Case ID: {case_id}\n\n"
        f"Case description:\n{post_desc.strip()}\n\n"
        "Candidate diagnoses (choose ONE):\n"
        f"{opts_block}\n\n"
        "Return exactly one option from the list above, copied verbatim."
    )

# -------------------------
# Output path resolver
# -------------------------
def resolve_out_csv(results_dir: Path, default_filename: str, output_csv: Optional[Path]) -> Path:
    """
    If output_csv is provided:
      - Ensure .csv suffix.
      - If relative, place under results_dir.
      - Create parent directories.
    Otherwise, use results_dir/default_filename.
    """
    if output_csv is not None:
        p = Path(output_csv)
        if p.suffix.lower() != ".csv":
            p = p.with_suffix(".csv")
        if not p.is_absolute():
            p = results_dir / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / default_filename

# -------------------------
# Responses API (gpt-5)
# -------------------------
def ask_model_responses(client: OpenAI, user_prompt: str, effort: str, debug: bool) -> Tuple[str, str]:
    resp = client.responses.create(
        model=MODEL_GPT5,
        reasoning={"effort": effort},  # "low" | "medium" | "high"
        input=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user_prompt},
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

def run_responses_mode(dataset_csv: Path, results_dir: Path, effort: str, debug: bool, output_csv: Optional[Path]):
    client = OpenAI()
    with open(dataset_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    dataset_name = dataset_csv.stem
    default_name = f"{dataset_name}_{MODEL_GPT5}_{effort}.csv"
    out_csv = resolve_out_csv(results_dir, default_name, output_csv)

    add_cols = ["Model Answer", "Model Answer (raw)", "Match Type", "Correct", "Options", "Model", "Reasoning Effort", "Note"]
    fieldnames = list(rows[0].keys()) + [c for c in add_cols if c not in rows[0].keys()]

    correct = 0
    total = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in tqdm(rows, desc="Processing (Responses gpt-5)", unit="case"):
            case_id = str(row.get("case_id", "")).strip()
            desc = row.get("PostDescription") or row.get("OriginalDescription") or ""
            options = build_options_list(row.get("DifferentialDiagnosisList", ""))
            gold = (row.get("FinalDiagnosis") or "").strip()

            user_prompt = build_user_prompt(case_id, desc, options)
            raw_ans, note = ask_model_responses(client, user_prompt, effort, debug)
            mapped, mtype = map_to_option(raw_ans, options)

            is_correct = int(norm_text(mapped) == norm_text(gold)) if mapped else 0
            correct += is_correct
            total += 1

            row_out = dict(row)
            row_out["Model Answer"] = mapped
            row_out["Model Answer (raw)"] = raw_ans
            row_out["Match Type"] = mtype
            row_out["Correct"] = is_correct
            row_out["Options"] = " | ".join(options)
            row_out["Model"] = MODEL_GPT5
            row_out["Reasoning Effort"] = effort
            row_out["Note"] = note
            writer.writerow(row_out)

    jsonl_path = out_csv.with_suffix(".jsonl")
    with open(out_csv, newline="", encoding="utf-8") as f_copy, open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
        for r in csv.DictReader(f_copy):
            f_jsonl.write(json.dumps(r, ensure_ascii=False) + "\n")

    acc = correct / max(total, 1)
    print(f"Saved → {out_csv}")
    print(f"Saved → {jsonl_path}")
    print(f"Accuracy: {correct}/{total} = {acc:.3f}")

# -------------------------
# Batch API (chat, gpt-4o)
# -------------------------
def build_batch_input_jsonl(rows: List[Dict], jsonl_path: Path):
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            rid = str(row.get("ID", row.get("case_id", ""))).strip() or f"row-{len(row)}"
            desc = row.get("PostDescription") or row.get("OriginalDescription") or ""
            options = build_options_list(row.get("DifferentialDiagnosisList", ""))
            user_prompt = build_user_prompt(rid, desc, options)

            body = {
                "model": MODEL_GPT4O,
                "messages": [
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": user_prompt},
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

def run_batch_mode(dataset_csv: Path, results_dir: Path, debug: bool, output_csv: Optional[Path]):
    client = OpenAI()
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_csv, newline="", encoding="utf-8") as f_in:
        rows = list(csv.DictReader(f_in))
    if not rows:
        raise RuntimeError("Empty dataset CSV.")

    dataset_name = dataset_csv.stem
    batch_jsonl = results_dir / f"{dataset_name}_{MODEL_GPT4O}_batch_input.jsonl"
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

    default_name = f"{dataset_name}_{MODEL_GPT4O}_batch.csv"
    out_csv = resolve_out_csv(results_dir, default_name, output_csv)

    add_cols = ["Model Answer", "Model Answer (raw)", "Match Type", "Correct", "Options", "Model", "Reasoning Effort", "Note"]
    fieldnames = list(rows[0].keys()) + [c for c in add_cols if c not in rows[0].keys()]

    correct = 0
    total = 0

    with open(dataset_csv, newline="", encoding="utf-8") as f_in, \
         open(out_csv, "w", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in tqdm(reader, desc="Merging batch outputs", unit="case"):
            rid = str(row.get("ID", row.get("case_id", ""))).strip()
            cid = f"id_{rid}"
            raw_ans = answers.get(cid, "")
            options = build_options_list(row.get("DifferentialDiagnosisList", ""))
            mapped, mtype = map_to_option(raw_ans, options)
            gold = (row.get("FinalDiagnosis") or "").strip()
            is_correct = int(norm_text(mapped) == norm_text(gold)) if mapped else 0
            correct += is_correct
            total += 1

            row_out = dict(row)
            row_out["Model Answer"] = mapped
            row_out["Model Answer (raw)"] = raw_ans
            row_out["Match Type"] = mtype
            row_out["Correct"] = is_correct
            row_out["Options"] = " | ".join(options)
            row_out["Model"] = MODEL_GPT4O
            row_out["Reasoning Effort"] = ""
            row_out["Note"] = ""
            writer.writerow(row_out)

    jsonl_path = out_csv.with_suffix(".jsonl")
    with open(out_csv, newline="", encoding="utf-8") as f_copy, open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
        for r in csv.DictReader(f_copy):
            f_jsonl.write(json.dumps(r, ensure_ascii=False) + "\n")

    acc = correct / max(total, 1)
    print(f"Saved → {out_csv}")
    print(f"Saved → {jsonl_path}")
    print(f"Accuracy: {correct}/{total} = {acc:.3f}")

# -------------------------
# Chat Completions (generic)
# -------------------------
def ask_model_chat(client: OpenAI, model: str, user_prompt: str, debug: bool) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    if debug:
        print("RAW CHAT RESPONSE OBJECT:")
        print(resp)
    return (resp.choices[0].message.content or "").strip()

def run_chat_mode(dataset_csv: Path, results_dir: Path, chat_model: str, debug: bool, output_csv: Optional[Path]):
    client = OpenAI()
    with open(dataset_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    dataset_name = dataset_csv.stem
    default_name = f"{dataset_name}_{chat_model}_chat.csv"
    out_csv = resolve_out_csv(results_dir, default_name, output_csv)

    add_cols = ["Model Answer", "Model Answer (raw)", "Match Type", "Correct", "Options", "Model", "Reasoning Effort", "Note"]
    fieldnames = list(rows[0].keys()) + [c for c in add_cols if c not in rows[0].keys()]

    correct = 0
    total = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in tqdm(rows, desc=f"Processing (Chat {chat_model})", unit="case"):
            case_id = str(row.get("case_id", "")).strip()
            desc = row.get("PostDescription") or row.get("OriginalDescription") or ""
            options = build_options_list(row.get("DifferentialDiagnosisList", ""))
            gold = (row.get("FinalDiagnosis") or "").strip()

            user_prompt = build_user_prompt(case_id, desc, options)
            raw_ans = ask_model_chat(client, chat_model, user_prompt, debug)
            mapped, mtype = map_to_option(raw_ans, options)
            is_correct = int(norm_text(mapped) == norm_text(gold)) if mapped else 0
            correct += is_correct
            total += 1

            row_out = dict(row)
            row_out["Model Answer"] = mapped
            row_out["Model Answer (raw)"] = raw_ans
            row_out["Match Type"] = mtype
            row_out["Correct"] = is_correct
            row_out["Options"] = " | ".join(options)
            row_out["Model"] = chat_model
            row_out["Reasoning Effort"] = ""  # n/a
            row_out["Note"] = ""
            writer.writerow(row_out)

    jsonl_path = out_csv.with_suffix(".jsonl")
    with open(out_csv, newline="", encoding="utf-8") as f_copy, open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
        for r in csv.DictReader(f_copy):
            f_jsonl.write(json.dumps(r, ensure_ascii=False) + "\n")

    acc = correct / max(total, 1)
    print(f"Saved → {out_csv}")
    print(f"Saved → {jsonl_path}")
    print(f"Accuracy: {correct}/{total} = {acc:.3f}")

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["responses", "batch", "chat"], default="responses",
                    help="responses=gpt-5 (Responses API); batch=gpt-4o Batch API; chat=Chat Completions API")
    ap.add_argument("--effort", choices=["low", "medium", "high"], default="low",
                    help="Reasoning effort for responses mode (gpt-5). Ignored in batch/chat modes.")
    ap.add_argument("--chat-model", default="gpt-5", help="Model for chat mode (e.g., gpt-5, gpt-4o).")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to dataset CSV.")
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS, help="Directory to write outputs.")
    ap.add_argument("--output-csv", type=Path, default=None,
                    help="Explicit output CSV filename. If relative, it is placed under --results-dir. "
                         "The .csv suffix is auto-added if missing.")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.mode == "responses":
        run_responses_mode(args.dataset, args.results_dir, args.effort, args.debug, args.output_csv)
    elif args.mode == "chat":
        run_chat_mode(args.dataset, args.results_dir, args.chat_model, args.debug, args.output_csv)
    else:
        run_batch_mode(args.dataset, args.results_dir, args.debug, args.output_csv)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# benchmarks/eurorad_responses.py
#
# Radiology single-label selection via Responses API (reasoning=medium by default).
# Dataset schema (CSV): case_id, OriginalDescription, PostDescription,
#                       DifferentialDiagnosisList, FinalDiagnosis
#
# Output adds:
#   Model Answer, Model Answer (raw), Match Type, Correct, Options, Model, Reasoning Effort, Note
# Supports --resume to skip already-scored rows (by case_id) and incrementally rewrite the CSV.
#
# Requires: pip install openai tqdm

import csv
import json
import argparse
import unicodedata
import difflib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from tqdm import tqdm

# ---- Defaults ----
DEFAULT_DATASET = Path("/home/bowang/Documents/alif/oss-benchmark/data/datasets/eurorad_test.csv")
DEFAULT_RESULTS = Path("/home/bowang/Documents/alif/oss-benchmark/results")
DEFAULT_MODEL = "gpt-5-2025-08-07"   # reasoning-capable
DEFAULT_EFFORT = "medium"            # low | medium | high

# ---- Prompt ----
SYS_PROMPT = (
    "You are a careful radiology diagnosis selector.\n"
    "Given a clinical case description and a finite list of candidate diagnoses,\n"
    "choose the single most likely final diagnosis FROM THE LIST.\n"
    "Response rules:\n"
    "1) Output EXACTLY one option, copied VERBATIM from the list.\n"
    "2) Output ONLY the diagnosis text. No explanation. No punctuation. No quotes.\n"
)

def build_options_list(s: str) -> List[str]:
    opts = [o.strip() for o in (s or "").split(",") if o.strip()]
    seen, out = set(), []
    for o in opts:
        if o not in seen:
            seen.add(o)
            out.append(o)
    return out

def norm_text(s: str) -> str:
    t = unicodedata.normalize("NFKC", s or "")
    t = t.replace("–", "-").replace("—", "-")
    t = " ".join(t.strip().split())
    return t.lower()

def map_to_option(raw_answer: str, options: List[str]) -> Tuple[str, str]:
    raw = (raw_answer or "").strip()
    if not raw:
        return "", "no_match"
    if raw in options:
        return raw, "exact"

    norm2opt = {norm_text(o): o for o in options}
    nr = norm_text(raw)
    if nr in norm2opt:
        return norm2opt[nr], "normalized"

    cand = difflib.get_close_matches(raw, options, n=1, cutoff=0.8)
    if cand:
        return cand[0], "fuzzy"

    norm_opts = list(norm2opt.keys())
    cand2 = difflib.get_close_matches(nr, norm_opts, n=1, cutoff=0.9)
    if cand2:
        return norm2opt[cand2[0]], "fuzzy"

    return "", "no_match"

def build_user_prompt(case_id: str, post_desc: str, options: List[str]) -> str:
    opts_block = "\n".join(f"- {o}" for o in options)
    return (
        f"Case ID: {case_id}\n\n"
        f"Case description:\n{(post_desc or '').strip()}\n\n"
        "Candidate diagnoses (choose ONE):\n"
        f"{opts_block}\n\n"
        "Return exactly one option from the list above, copied verbatim."
    )

def resolve_out_csv(results_dir: Path, default_filename: str, output_csv: Optional[Path]) -> Path:
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
# Resume helpers
# -------------------------
ADD_COLS = ["Model Answer", "Model Answer (raw)", "Match Type", "Correct",
            "Options", "Model", "Reasoning Effort", "Note"]

def load_prev_results(path: Path) -> Dict[str, dict]:
    """
    Load previous output CSV into a dict: case_id -> row-dict.
    Returns {} if file doesn't exist.
    """
    if not path.exists():
        return {}
    prev: Dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = str(row.get("case_id", "")).strip()
            if cid:
                prev[cid] = row
    return prev

def row_done(prev_row: Optional[dict]) -> bool:
    """
    Consider a row 'done' if it has a non-empty Model Answer (raw) or Model Answer.
    """
    if not prev_row:
        return False
    return bool((prev_row.get("Model Answer (raw)") or "").strip() or
                (prev_row.get("Model Answer") or "").strip())

# -------------------------
# Responses API (reasoning models)
# -------------------------
def ask_model_responses(client: OpenAI, model: str, effort: str, user_prompt: str, debug: bool) -> Tuple[str, str]:
    resp = client.responses.create(
        model=model,
        reasoning={"effort": effort},  # low | medium | high
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
    return answer, note

def run_responses_mode(dataset_csv: Path, results_dir: Path, model: str, effort: str,
                       resume: bool, debug: bool, output_csv: Optional[Path]):
    client = OpenAI()
    with open(dataset_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("Empty dataset CSV.")

    dataset_name = dataset_csv.stem
    default_name = f"{dataset_name}_{model}_responses_re-{effort}.csv"
    out_csv = resolve_out_csv(results_dir, default_name, output_csv)

    # Load previous results if resuming
    prev_map = load_prev_results(out_csv) if resume else {}

    # Compose fieldnames from dataset + required add cols
    fieldnames = list(rows[0].keys()) + [c for c in ADD_COLS if c not in rows[0].keys()]

    correct = 0
    total = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for row in tqdm(rows, desc=f"Processing (Responses {model}, re={effort})", unit="case"):
            case_id = str(row.get("case_id", "")).strip() or ""
            desc = row.get("PostDescription") or row.get("OriginalDescription") or ""
            options = build_options_list(row.get("DifferentialDiagnosisList", ""))
            gold = (row.get("FinalDiagnosis") or "").strip()

            prev_row = prev_map.get(case_id) if case_id else None
            if resume and row_done(prev_row):
                # Reuse previous outputs; recompute accuracy if needed
                mapped = (prev_row.get("Model Answer") or "").strip()
                if mapped:
                    is_correct = int(norm_text(mapped) == norm_text(gold))
                else:
                    # Fallback recompute from raw if possible
                    mapped2, _ = map_to_option(prev_row.get("Model Answer (raw)", ""), options)
                    is_correct = int(norm_text(mapped2) == norm_text(gold)) if mapped2 else int(prev_row.get("Correct", "0") == "1")
                correct += is_correct
                total += 1

                row_out = dict(row)
                for k in ADD_COLS:
                    row_out[k] = prev_row.get(k, "") if prev_row else ""
                # Ensure Options reflects current options parsing
                row_out["Options"] = " | ".join(options)
                row_out["Correct"] = is_correct
                # Preserve model/effort if present; otherwise set current
                row_out["Model"] = row_out.get("Model") or model
                row_out["Reasoning Effort"] = row_out.get("Reasoning Effort") or effort
                writer.writerow(row_out)
                continue

            # Fresh inference
            user_prompt = build_user_prompt(case_id, desc, options)
            raw_ans, note = ask_model_responses(client, model, effort, user_prompt, debug)
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
            row_out["Model"] = model
            row_out["Reasoning Effort"] = effort
            row_out["Note"] = note
            writer.writerow(row_out)

    # Mirror to JSONL
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
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Reasoning-capable model (e.g., gpt-5-2025-08-07).")
    ap.add_argument("--effort", choices=["low", "medium", "high"], default=DEFAULT_EFFORT,
                    help="Reasoning effort (default: medium).")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to dataset CSV.")
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS, help="Directory to write outputs.")
    ap.add_argument("--output-csv", type=Path, default=None,
                    help="Explicit output CSV filename (placed under --results-dir if relative).")
    ap.add_argument("--resume", action="store_true", help="Skip rows already present in the output CSV.")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    run_responses_mode(args.dataset, args.results_dir, args.model, args.effort,
                       args.resume, args.debug, args.output_csv)

if __name__ == "__main__":
    main()

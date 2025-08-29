#!/usr/bin/env python3
# benchmarks/openrouter.py
#
# Radiology dataset adapter for OpenRouter.
# - Expects CSV columns: case_id, OriginalDescription, PostDescription,
#   DifferentialDiagnosisList, FinalDiagnosis
# - Returns EXACT diagnosis copied from the provided list.
# - Resumable; retries with backoff.
#
# Requires: pip install requests pandas tqdm

import os
import re
import json
import time
import argparse
import unicodedata
import difflib
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRY_STATUSES = {429, 500, 502, 503, 504, 520, 522, 524}

DEFAULT_SYSTEM = (
    "You are a careful radiology diagnosis selector.\n"
    "Given a case description and a finite list of candidate diagnoses, "
    "choose the single most likely final diagnosis FROM THE LIST.\n"
    "Response rules:\n"
    "1) Output EXACTLY one option, copied VERBATIM from the list.\n"
    "2) Output ONLY the diagnosis text. No explanation. No punctuation. No quotes."
)

DEFAULT_USER_TEMPLATE = """\
Case ID: {case_id}

Case description:
{case_text}

Candidate diagnoses (choose ONE):
{options_block}

Return exactly one option from the list above, copied verbatim.
"""

# ------------------------- helpers -------------------------
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
    if not raw or not options:
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
    candn = difflib.get_close_matches(nr, norm_opts, n=1, cutoff=0.9)
    if candn:
        return norm2opt[candn[0]], "fuzzy"

    return "", "no_match"

def build_user_prompt(row: pd.Series, user_template: str) -> str:
    case_id = str(row.get("case_id", "")).strip()
    desc = str(row.get("PostDescription") or row.get("OriginalDescription") or "").strip()
    options = build_options_list(row.get("DifferentialDiagnosisList", ""))
    opts_block = "\n".join(f"- {o}" for o in options) if options else "- (no options provided)"
    return user_template.format(case_id=case_id, case_text=desc, options_block=opts_block)

def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s)

def default_output_path(
    input_csv: Path,
    endpoint: str,
    results_dir: Path,
    reasoning_effort: Optional[str] = None,
    include_reasoning: bool = False,
    max_output_tokens: Optional[int] = None,
) -> Path:
    dataset_stem = input_csv.stem
    model_tag = sanitize(endpoint.replace("/", "-"))
    parts = [dataset_stem, model_tag]
    if reasoning_effort:
        parts.append(f"re-{sanitize(reasoning_effort)}")
    if include_reasoning:
        parts.append("cot")
    if max_output_tokens is not None:
        parts.append(f"max{int(max_output_tokens)}")
    filename = "_".join(parts) + ".csv"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / filename

def needs_rerun(raw: str, mapped: str) -> bool:
    if isinstance(mapped, str) and mapped.strip():
        return False
    if not raw or not str(raw).strip():
        return True
    if str(raw).strip().upper().startswith("ERROR:"):
        return True
    return True

# ------------------------- API call -------------------------
def call_openrouter(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_retries: int = 6,
    timeout: int = 60,
    include_reasoning: bool = False,
    max_output_tokens: Optional[int] = None,
    temperature: float = 0.0,
    verbose: bool = False,
) -> Tuple[str, Optional[str]]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY environment variable.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://chat.openai.com/",
        "X-Title": "Radiology Benchmark Runner",
    }

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
    }
    if max_output_tokens is not None:
        payload["max_tokens"] = int(max_output_tokens)
    if include_reasoning:
        payload["reasoning"] = {"exclude": False}

    backoff = 2.0
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(payload), timeout=timeout)

            if r.status_code in RETRY_STATUSES:
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                if verbose:
                    print(f"[retry] {last_err} (attempt {attempt}/{max_retries}); sleeping {backoff:.1f}s")
                time.sleep(backoff)
                backoff = min(backoff * 1.8, 30)
                continue

            if r.status_code != 200:
                raise RuntimeError(f"OpenRouter error {r.status_code}: {r.text[:500]}")

            try:
                data = r.json()
            except requests.exceptions.JSONDecodeError as je:
                last_err = RuntimeError(
                    f"JSON decode failed (attempt {attempt}/{max_retries}): {je}; body[:200]={r.text[:200]!r}"
                )
                if verbose:
                    print(f"[retry] {last_err}; sleeping {backoff:.1f}s")
                time.sleep(backoff)
                backoff = min(backoff * 1.8, 30)
                continue

            msg = data["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            reasoning = msg.get("reasoning")
            reasoning_text = json.dumps(reasoning, ensure_ascii=False) if isinstance(reasoning, (dict, list)) else (reasoning or None)
            return content, reasoning_text

        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            if attempt == max_retries:
                break
            if verbose:
                print(f"[retry] {type(e).__name__} (attempt {attempt}/{max_retries}); sleeping {backoff:.1f}s")
            time.sleep(backoff)
            backoff = min(backoff * 1.8, 30)

    raise RuntimeError(str(last_err) if last_err else "Failed after retries")

# ------------------------- workers -------------------------
def process_one(
    idx: int,
    row: pd.Series,
    endpoint: str,
    system_prompt: str,
    user_template: str,
    sleep_between: float,
    reasoning_effort: Optional[str],
    include_reasoning: bool,
    max_output_tokens: Optional[int],
    temperature: float,
    verbose: bool,
) -> Tuple[int, str, str, str, Optional[str]]:
    # Fill {effort} if present in system prompt
    try:
        system_filled = system_prompt.format(effort=(reasoning_effort or "medium"))
    except KeyError:
        system_filled = system_prompt

    user_prompt = build_user_prompt(row, user_template)
    raw, reasoning = call_openrouter(
        system_filled,
        user_prompt,
        endpoint,
        include_reasoning=include_reasoning,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        verbose=verbose,
    )
    options = build_options_list(row.get("DifferentialDiagnosisList", ""))
    mapped, mtype = map_to_option(raw, options)
    if sleep_between > 0:
        time.sleep(sleep_between)
    return idx, raw, mapped, mtype, reasoning

def task_wrapper(
    i: int,
    row: pd.Series,
    endpoint: str,
    system_prompt: str,
    user_template: str,
    sleep_between: float,
    reasoning_effort: Optional[str],
    include_reasoning: bool,
    max_output_tokens: Optional[int],
    temperature: float,
    verbose: bool,
) -> Tuple[int, str, str, str, Optional[str]]:
    try:
        return process_one(
            i, row, endpoint, system_prompt, user_template, sleep_between,
            reasoning_effort, include_reasoning, max_output_tokens, temperature, verbose
        )
    except Exception as e:
        return i, f"ERROR: {e}", "", "no_match", None

# ------------------------- main -------------------------
def main():
    p = argparse.ArgumentParser(description="OpenRouter radiology benchmark (verbatim option selection, resumable).")
    p.add_argument("input_csv", help="Path to input CSV (radiology cases).")
    p.add_argument("--endpoint", required=True, help="OpenRouter model id, e.g. openai/gpt-oss-20b or qwen/... ")
    p.add_argument("--results_dir", default="results", help="Directory to write results")
    p.add_argument("--output_csv", default=None, help="Explicit output path; otherwise auto-generated")
    p.add_argument("--workers", type=int, default=1, help="Concurrent workers")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep between requests per worker (seconds)")
    p.add_argument("--max", type=int, default=None, help="Process at most this many rows")
    p.add_argument("--resume", action="store_true", help="Resume if output exists (keep previous rows)")
    p.add_argument("--system", default=DEFAULT_SYSTEM, help="Override system prompt (may contain {effort})")
    p.add_argument("--user_template", default=DEFAULT_USER_TEMPLATE, help="Override user prompt template")
    p.add_argument("--reasoning_effort", choices=["low", "medium", "high"], default=None)
    p.add_argument("--include_reasoning", action="store_true")
    p.add_argument("--max_output_tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    input_csv = Path(args.input_csv)
    results_dir = Path(args.results_dir)
    output_csv = (
        Path(args.output_csv) if args.output_csv else
        default_output_path(
            input_csv,
            args.endpoint,
            results_dir,
            reasoning_effort=args.reasoning_effort,
            include_reasoning=args.include_reasoning,
            max_output_tokens=args.max_output_tokens,
        )
    )
    print(f"Using output file: {output_csv}")

    df = pd.read_csv(input_csv)

    # Ensure output columns exist
    cols = ["model_answer_raw", "model_answer", "match_type", "correct", "options", "model_reasoning_raw"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""

    # Resume merge
    if args.resume and output_csv.exists():
        prev = pd.read_csv(output_csv)
        for c in cols:
            if c in prev.columns:
                vals = list(prev[c])[: len(df)]
                if len(vals) < len(df):
                    vals += [""] * (len(df) - len(vals))
                df[c] = vals

    indices = df.index if args.max is None else df.index[: args.max]
    todo = [i for i in indices if needs_rerun(str(df.at[i, "model_answer_raw"]),
                                              str(df.at[i, "model_answer"]))]

    done = len(indices) - len(todo)
    print(f"resume: {done} done / {len(indices)} total, {len(todo)} to run")
    if not todo:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Saved (unchanged): {output_csv}")
        if "FinalDiagnosis" in df.columns:
            gold = df["FinalDiagnosis"].astype(str).apply(norm_text)
            pred = df["model_answer"].astype(str).apply(norm_text)
            acc = (gold == pred).mean() if len(df) else 0.0
            print(f"Accuracy: {acc:.3f}")
        return

    print(f"Running {len(todo)} requests against {args.endpoint} with {args.workers} workers…")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = []
        for i in todo:
            futures.append(
                ex.submit(
                    task_wrapper,
                    i,
                    df.loc[i],
                    args.endpoint,
                    args.system,
                    args.user_template,
                    args.sleep,
                    args.reasoning_effort,
                    args.include_reasoning,
                    args.max_output_tokens,
                    args.temperature,
                    args.verbose,
                )
            )

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Completing"):
            i, raw, mapped, mtype, reasoning = fut.result()
            df.at[i, "model_answer_raw"] = raw
            df.at[i, "model_answer"] = mapped
            df.at[i, "match_type"] = mtype
            df.at[i, "options"] = " | ".join(build_options_list(df.loc[i].get("DifferentialDiagnosisList", "")))
            if reasoning is not None:
                df.at[i, "model_reasoning_raw"] = reasoning
            if "FinalDiagnosis" in df.columns and mapped:
                gold = norm_text(str(df.loc[i, "FinalDiagnosis"]))
                df.at[i, "correct"] = int(norm_text(mapped) == gold)
            else:
                df.at[i, "correct"] = ""

            short = (raw[:160] + "…") if raw and len(raw) > 160 else raw
            print(f"[row {i}] mapped='{mapped or '?'}' | type={mtype} | raw: {short}")
            df.to_csv(output_csv, index=False)

    df.to_csv(output_csv, index=False)
    print(f"Saved results to: {output_csv}")

    if "FinalDiagnosis" in df.columns:
        mask = df["model_answer"].astype(str).str.len() > 0
        if mask.any():
            gold = df.loc[mask, "FinalDiagnosis"].astype(str).apply(norm_text)
            pred = df.loc[mask, "model_answer"].astype(str).apply(norm_text)
            acc = (gold == pred).mean() if len(gold) else 0.0
            print(f"Accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()

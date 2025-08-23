#!/usr/bin/env python3
# benchmarks/openrouter.py
import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple

import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Accept a contiguous run of A–Z (supports single- or multi-answer, no spaces)
LETTERS_RUN_RE = re.compile(r"\b([A-Z]{1,26})\b")
RETRY_STATUSES = {429, 500, 502, 503, 504, 520, 522, 524}

DEFAULT_SYSTEM = (
    "You are a careful ophthalmology question-answering assistant. "
    "You will be given a multiple-choice case with options labeled A–Z. "
    "Some questions have a single correct answer, while others have multiple correct answers. "
    "Select ALL correct answers. If only one answer is correct, return just that single letter. "
    "Respond with ONLY the capital letters (A–Z), concatenated together with no spaces or punctuation "
    "(e.g., 'ABE' for multiple answers, or 'D' if only one). Do not explain.\n"
    # "Reasoning: {effort}"
)

DEFAULT_USER_TEMPLATE = """\
Case:
{case_text}

Task: Choose the correct answer(s).

Return: One or more letters from A to Z, concatenated with no spaces (e.g., ABE or D).
"""

def parse_choice(raw: str) -> Optional[str]:
    if not raw:
        return None
    m = LETTERS_RUN_RE.search((raw or "").strip().upper())
    return m.group(1) if m else None

def call_openrouter(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_retries: int = 6,
    timeout: int = 60,
    include_reasoning: bool = False,
    max_output_tokens: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[str, Optional[str]]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY environment variable.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://chat.openai.com/",
        "X-Title": "General Benchmark Runner",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if max_output_tokens is not None:
        payload["max_tokens"] = int(max_output_tokens)

    if include_reasoning:
        payload["reasoning"] = {"exclude": False}

    backoff = 2.0
    last_err = None
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
            if isinstance(reasoning, (dict, list)):
                reasoning_text = json.dumps(reasoning, ensure_ascii=False)
            else:
                reasoning_text = (reasoning or None)
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

def build_user_prompt(row: pd.Series, user_template: str) -> str:
    q_col = next((c for c in row.index if c.lower() == "question"), None)
    if q_col:
        case_text = str(row[q_col]).strip()
    else:
        parts = [str(v) for v in row.values if isinstance(v, str)]
        case_text = "\n".join(parts).strip()
    return user_template.format(case_text=case_text)

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

def is_done_choice(s: str) -> bool:
    if not isinstance(s, str):
        return False
    return re.fullmatch(r"[A-Z]{1,26}", (s.strip() or "")) is not None

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
    verbose: bool,
) -> Tuple[int, str, Optional[str], Optional[str]]:
    prompt = build_user_prompt(row, user_template)
    try:
        system_filled = system_prompt.format(effort=(reasoning_effort or "medium"))
    except KeyError:
        system_filled = system_prompt

    raw, reasoning = call_openrouter(
        system_filled,
        prompt,
        endpoint,
        include_reasoning=include_reasoning,
        max_output_tokens=max_output_tokens,
        verbose=verbose,
    )
    choice = parse_choice(raw)
    if sleep_between > 0:
        time.sleep(sleep_between)
    return idx, raw, choice, reasoning

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
    verbose: bool,
) -> Tuple[int, str, Optional[str], Optional[str]]:
    try:
        return process_one(
            i, row, endpoint, system_prompt, user_template, sleep_between,
            reasoning_effort, include_reasoning, max_output_tokens, verbose
        )
    except Exception as e:
        return i, f"ERROR: {e}", None, None

def main():
    p = argparse.ArgumentParser(description="General-purpose OpenRouter CSV benchmarker (with reasoning controls)")
    p.add_argument("input_csv", help="Path to input CSV (e.g., data/datasets/ophthalmology.csv)")
    p.add_argument("--endpoint", required=True, help="OpenRouter model id, e.g. openai/gpt-oss-120b or qwen/...")
    p.add_argument("--results_dir", default="results", help="Directory to write results (default: results)")
    p.add_argument("--output_csv", default=None, help="Explicit output path; otherwise auto-generated")
    p.add_argument("--workers", type=int, default=1, help="Number of concurrent workers")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep between requests per worker (seconds)")
    p.add_argument("--max", type=int, default=None, help="Process at most this many rows")
    p.add_argument("--resume", action="store_true", help="Resume if output exists (keep previous rows)")
    p.add_argument("--system", default=DEFAULT_SYSTEM, help="Override system prompt (may contain {effort})")
    p.add_argument("--user_template", default=DEFAULT_USER_TEMPLATE, help="Override user prompt template")
    p.add_argument("--reasoning_effort", choices=["low", "medium", "high"], default=None)
    p.add_argument("--include_reasoning", action="store_true")
    p.add_argument("--max_output_tokens", type=int, default=None)
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
    model_raw_col = "model_answer_raw"
    model_choice_col = "model_answer_choice"
    model_reason_col = "model_reasoning_raw"
    if model_raw_col not in df.columns:
        df[model_raw_col] = ""
    if model_choice_col not in df.columns:
        df[model_choice_col] = ""
    if model_reason_col not in df.columns:
        df[model_reason_col] = ""

    if args.resume and output_csv.exists():
        prev = pd.read_csv(output_csv)
        for col in [model_raw_col, model_choice_col, model_reason_col]:
            if col in prev.columns:
                vals = list(prev[col])[: len(df)]
                if len(vals) < len(df):
                    vals += [""] * (len(df) - len(vals))
                df[col] = vals

    indices = df.index if args.max is None else df.index[: args.max]
    todo = [i for i in indices if not is_done_choice(str(df.at[i, model_choice_col]))]
    done_mask = df[model_choice_col].astype(str).apply(is_done_choice)
    print(f"resume: {done_mask.sum()} done / {len(df)} total, {len(todo)} to run")

    if not todo:
        print("Nothing to do (all rows populated).")
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Saved (unchanged): {output_csv}")
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
                    args.verbose,
                )
            )

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Completing"):
            i, raw, choice, reasoning = fut.result()
            df.at[i, model_raw_col] = raw
            df.at[i, model_choice_col] = "" if choice is None else choice
            if reasoning is not None:
                df.at[i, model_reason_col] = reasoning

            short = (raw[:160] + "…") if raw and len(raw) > 160 else raw
            if args.include_reasoning and reasoning:
                rshort = (reasoning[:120] + "…") if len(reasoning) > 120 else reasoning
                print(f"[row {i}] choice={choice or '?'} | {short} | reasoning: {rshort}")
            else:
                print(f"[row {i}] choice={choice or '?'} | {short}")

            df.to_csv(output_csv, index=False)

    df.to_csv(output_csv, index=False)
    print(f"Saved results to: {output_csv}")

if __name__ == "__main__":
    main()

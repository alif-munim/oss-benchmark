#!/usr/bin/env python3
# benchmarks/openrouter.py
#
# Batch evaluation of diagnostic tasks via OpenRouter (Likert scoring 1–5).
# - Expects a CSV with a text column containing each task description.
# - Sends each row to the model with the SAME evaluation prompt you provided.
# - Parses the model output into LLM_eval_score (accepts 1..5 and x.5).
# - Resumable; retries with backoff; incremental saving.
#
# Requires: pip install requests pandas tqdm
# Env: OPENROUTER_API_KEY=...

import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRY_STATUSES = {429, 500, 502, 503, 504, 520, 522, 524}

# ---------- Evaluation prompt (same as your reference) ----------
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

# Minimal system message (configurable via --system)
DEFAULT_SYSTEM = ""

# Column with the task text
DEFAULT_TEXT_COL = "Disease_description"

# ---------- Parsing ----------
# Accept 1..5 with optional .5 or .0 (e.g., 1, 2, 3.5, 4.0). Store the matched token as-is.
SCORE_RE = re.compile(r"\b([1-5](?:\.5|\.0)?)\b")

def extract_final(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"<\|channel\|>final<\|message\|>(.*)$", text, re.DOTALL)
    return (m.group(1).strip() if m else text.strip())

def parse_score(raw: str) -> Optional[str]:
    if not raw:
        return None
    t = extract_final(raw)
    m = SCORE_RE.search(t)
    return m.group(1) if m else None

def build_user_prompt(row: pd.Series, text_column: str) -> str:
    if text_column in row:
        task_text = str(row[text_column]).strip()
    else:
        task_text = "\n".join([str(v) for v in row.values if isinstance(v, str)]).strip()
    return EVAL_PROMPT + task_text

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
    parts = [dataset_stem, model_tag, "eval"]
    if reasoning_effort:
        parts.append(f"re-{sanitize(reasoning_effort)}")
    if include_reasoning:
        parts.append("cot")
    if max_output_tokens is not None:
        parts.append(f"max{int(max_output_tokens)}")
    filename = "_".join(parts) + ".csv"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / filename

def is_done_score(s: str) -> bool:
    return isinstance(s, str) and bool(SCORE_RE.fullmatch((s.strip() or "")))

def needs_rerun(raw: str, score: str) -> bool:
    if is_done_score(score or ""):
        return False
    if not raw or not str(raw).strip():
        return True
    if str(raw).strip().upper().startswith("ERROR:"):
        return True
    return True

# ---------- API call ----------
def call_openrouter(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_retries: int = 6,
    timeout: int = 60,
    include_reasoning: bool = False,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,  # None => use model default (do not send)
    verbose: bool = False,
) -> Tuple[str, Optional[str]]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY environment variable.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://chat.openai.com/",
        "X-Title": "Diagnostic Likert Benchmark",
    }

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    # Use model defaults unless explicitly provided
    if temperature is not None:
        payload["temperature"] = float(temperature)
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
            except ValueError as je:
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

# ---------- workers ----------
def process_one(
    idx: int,
    row: pd.Series,
    endpoint: str,
    system_prompt: str,
    text_column: str,
    sleep_between: float,
    reasoning_effort: Optional[str],
    include_reasoning: bool,
    max_output_tokens: Optional[int],
    temperature: Optional[float],
    verbose: bool,
) -> Tuple[int, str, Optional[str], Optional[str]]:
    # Fill {effort} if present in system prompt
    try:
        system_filled = system_prompt.format(effort=(reasoning_effort or "medium"))
    except KeyError:
        system_filled = system_prompt

    user_prompt = build_user_prompt(row, text_column)
    raw, reasoning = call_openrouter(
        system_filled,
        user_prompt,
        endpoint,
        include_reasoning=include_reasoning,
        max_output_tokens=max_output_tokens,
        temperature=temperature,  # None => model default
        verbose=verbose,
    )
    score = parse_score(raw)
    if sleep_between > 0:
        time.sleep(sleep_between)
    return idx, raw, score, reasoning

def task_wrapper(
    i: int,
    row: pd.Series,
    endpoint: str,
    system_prompt: str,
    text_column: str,
    sleep_between: float,
    reasoning_effort: Optional[str],
    include_reasoning: bool,
    max_output_tokens: Optional[int],
    temperature: Optional[float],
    verbose: bool,
) -> Tuple[int, str, Optional[str], Optional[str]]:
    try:
        return process_one(
            i, row, endpoint, system_prompt, text_column, sleep_between,
            reasoning_effort, include_reasoning, max_output_tokens, temperature, verbose
        )
    except Exception as e:
        return i, f"ERROR: {e}", None, None

# ---------- main ----------
def main():
    p = argparse.ArgumentParser(description="OpenRouter diagnostic Likert scoring (1–5), resumable.")
    p.add_argument("input_csv", help="Path to input CSV (expects a text column, default: Disease_description).")
    p.add_argument("--endpoint", required=True, help="OpenRouter model id, e.g. openai/gpt-oss-20b or qwen/…")
    p.add_argument("--results_dir", default="results", help="Directory to write results")
    p.add_argument("--output_csv", default=None, help="Explicit output path; otherwise auto-generated")
    p.add_argument("--workers", type=int, default=1, help="Concurrent workers")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep between requests per worker (seconds)")
    p.add_argument("--max", type=int, default=None, help="Process at most this many rows")
    p.add_argument("--resume", action="store_true", help="Resume if output exists (keep previous rows)")
    p.add_argument("--system", default=DEFAULT_SYSTEM, help="Override system prompt (may contain {effort})")
    p.add_argument("--text_column", default=DEFAULT_TEXT_COL, help="Column with task text (default: Disease_description)")
    p.add_argument("--reasoning_effort", choices=["low", "medium", "high"], default=None)
    p.add_argument("--include_reasoning", action="store_true", help="Include model reasoning field if supported (non-default).")
    p.add_argument("--max_output_tokens", type=int, default=None, help="Max tokens; omit to use model default.")
    p.add_argument("--temperature", type=float, default=None, help="Sampling temperature; omit to use model default.")
    p.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds.")
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
    RAW_COL = "LLM_eval_raw"
    SCORE_COL = "LLM_eval_score"
    REASON_COL = "LLM_eval_reasoning_raw"
    for c in (RAW_COL, SCORE_COL, REASON_COL):
        if c not in df.columns:
            df[c] = ""

    # Resume merge
    if args.resume and output_csv.exists():
        prev = pd.read_csv(output_csv)
        for c in (RAW_COL, SCORE_COL, REASON_COL):
            if c in prev.columns:
                vals = list(prev[c])[: len(df)]
                if len(vals) < len(df):
                    vals += [""] * (len(df) - len(vals))
                df[c] = vals

    indices = df.index if args.max is None else df.index[: args.max]
    todo = [i for i in indices if needs_rerun(str(df.at[i, RAW_COL]),
                                              str(df.at[i, SCORE_COL]))]

    done = len(indices) - len(todo)
    print(f"resume: {done} done / {len(indices)} total, {len(todo)} to run")
    if not todo:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Saved (unchanged): {output_csv}")
        return

    print(f"Running {len(todo)} requests against {args.endpoint} with {args.workers} workers…")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Thread pool
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
                    args.text_column,
                    args.sleep,
                    args.reasoning_effort,
                    args.include_reasoning,
                    args.max_output_tokens,
                    args.temperature,  # None => model default
                    args.verbose,
                )
            )

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Completing"):
            i, raw, score, reasoning = fut.result()
            df.at[i, RAW_COL] = raw
            df.at[i, SCORE_COL] = "" if score is None else score
            if reasoning is not None:
                df.at[i, REASON_COL] = reasoning

            short = (raw[:160] + "…") if raw and len(raw) > 160 else raw
            print(f"[row {i}] score={score or '?'} | {short}")
            df.to_csv(output_csv, index=False)

    df.to_csv(output_csv, index=False)
    print(f"Saved results to: {output_csv}")

if __name__ == "__main__":
    main()

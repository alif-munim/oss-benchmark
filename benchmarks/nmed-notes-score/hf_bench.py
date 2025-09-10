#!/usr/bin/env python3
# benchmarks/nmed-notes-score/hf_bench.py
#
# Likert scoring (1–5, .5 allowed) via HF Inference Providers on a CSV (resumable).
# - Supports two eval modes: diagnosis (default) and treatment (--eval_mode treatment).
# - Points to one text column (default: Disease_description); if missing, falls back to joining string columns.
# - Truncates overly long prompts (configurable) to avoid context errors.
# - Small default max_output_tokens recommended for numeric-only outputs.
# - Creates output directory even when --output_csv is used.
#
# Requires: pip install openai pandas tqdm

import os, re, argparse, time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ---------- Prompts ----------
DIAG_PROMPT = """You are asked to evaluate the quality of a model’s diagnostic output using the following rubric:

**Scoring Rubric (Likert scale 1–5):**
1. Most relevant options not mentioned.
2. Some or many relevant options not mentioned.
3. Most relevant options mentioned.
4. Most relevant options mentioned.
5. All relevant options mentioned.

**Instruction:**
Given the following task description, the true disease, and the model output, assign a single score from 1 to 5.
Half-points are allowed (e.g., 3.5). Output ONLY the score (no words, no units).
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

DEFAULT_SYSTEM = ""
DEFAULT_TEXT_COL = "Disease_description"

# ---------- Parsing ----------
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

def build_user_prompt(row: pd.Series, text_column: str, eval_prompt: str) -> str:
    if text_column in row:
        task_text = str(row[text_column]).strip()
    else:
        task_text = "\n".join([str(v) for v in row.values if isinstance(v, str)]).strip()
    return eval_prompt + task_text

def sanitize(s: str) -> str:
    import re as _re
    return _re.sub(r"[^A-Za-z0-9._-]+", "-", s)

def out_path(input_csv: Path, model: str, results_dir: Path,
             reasoning: Optional[str], max_out: Optional[int], api: str, mode: str) -> Path:
    tag = model.replace("/", "-").replace(":", "-")
    parts = [input_csv.stem, tag, api, mode, "eval"]
    if reasoning: parts.append(f"re-{sanitize(reasoning)}")
    if max_out:  parts.append(f"max{int(max_out)}")
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / ("_".join(parts) + ".csv")

def is_done_score(s: str) -> bool:
    return isinstance(s, str) and bool(SCORE_RE.fullmatch(s.strip() or ""))

def needs_rerun(raw: str, score: str) -> bool:
    if is_done_score(score or ""):
        return False
    if not raw or not raw.strip():
        return True
    if raw.strip().upper().startswith("ERROR:"):
        return True
    return True

# ---------- Clients ----------
class HFClientBase:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: Optional[float]):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.timeout = timeout

class HFChatClient(HFClientBase):
    def infer(self, system_text: str, user_text: str,
              reasoning_effort: Optional[str], max_output_tokens: Optional[int],
              temperature: Optional[float]) -> str:
        messages = []
        if system_text and system_text.strip():
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_text})

        kwargs: Dict[str, Any] = {"model": self.model, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if max_output_tokens is not None:
            kwargs["max_tokens"] = int(max_output_tokens)
        if self.timeout is not None:
            kwargs["timeout"] = float(self.timeout)

        cc = self.client.chat.completions.create(**kwargs)
        return (cc.choices[0].message.content or "").strip()

class HFResponsesClient(HFClientBase):
    def infer(self, system_text: str, user_text: str,
              reasoning_effort: Optional[str], max_output_tokens: Optional[int],
              temperature: Optional[float]) -> str:
        msgs: List[Dict[str, Any]] = []
        if system_text and system_text.strip():
            msgs.append({"role": "system", "content": system_text})
        msgs.append({"role": "user", "content": user_text})

        kwargs: Dict[str, Any] = {"model": self.model, "input": msgs}
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = int(max_output_tokens)
        if self.timeout is not None:
            kwargs["timeout"] = float(self.timeout)

        r = self.client.responses.create(**kwargs)
        text = getattr(r, "output_text", None) or ""
        if text and text.strip():
            return text.strip()

        blocks = []
        for blk in (getattr(r, "output", None) or []):
            for c in (getattr(blk, "content", None) or []):
                if getattr(c, "type", "") == "output_text" and getattr(c, "text", None):
                    blocks.append(c.text)
        return "\n".join(blocks).strip()

# ---------- Runner ----------
def main():
    ap = argparse.ArgumentParser(description="Likert scoring (1–5) via HF Inference Providers on a CSV (resumable).")
    ap.add_argument("input_csv")
    ap.add_argument("--model", required=True, help="e.g., openai/gpt-oss-20b:fireworks-ai or openai/gpt-oss-120b:cerebras")
    ap.add_argument("--api", choices=["chat", "responses"], default="chat",
                    help="Default 'chat' works with many providers; use 'responses' if preferred.")
    ap.add_argument("--router_url", default="https://router.huggingface.co/v1")
    ap.add_argument("--hf_token", default=os.getenv("HF_TOKEN"))
    ap.add_argument("--system", default=DEFAULT_SYSTEM)
    ap.add_argument("--text_column", default=DEFAULT_TEXT_COL, help="Column containing the task text (default: Disease_description)")
    ap.add_argument("--eval_mode", choices=["diagnosis", "treatment"], default="diagnosis",
                    help="Which rubric/prompt to use.")
    # Provider defaults unless overridden:
    ap.add_argument("--reasoning_effort", choices=["low","medium","high"], default=None)
    ap.add_argument("--max_output_tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    # QoL flags:
    ap.add_argument("--truncate_input_chars", type=int, default=12000, help="Hard cap for user prompt length; 0 disables.")
    ap.add_argument("--request_timeout", type=float, default=60.0, help="Per-request timeout (seconds).")
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N rows (debug).")
    ap.add_argument("--log_first_n_errors", type=int, default=3, help="Print the first N exceptions immediately.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--results", default="results")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--output_csv", default=None, help="Write to this path (use when resuming from an output CSV).")
    ap.add_argument("--max_retries", type=int, default=4)
    ap.add_argument("--base_backoff", type=float, default=2.0)
    args = ap.parse_args()

    if not args.hf_token:
        raise SystemExit("HF_TOKEN is required (env or --hf_token).")

    input_csv = Path(args.input_csv)

    # Select eval prompt by mode
    eval_prompt = DIAG_PROMPT if args.eval_mode == "diagnosis" else TREAT_PROMPT

    # Output path
    if args.output_csv:
        out_csv = Path(args.output_csv)
    else:
        out_csv = out_path(input_csv, args.model, Path(args.results),
                           args.reasoning_effort, args.max_output_tokens, args.api, args.eval_mode)
    print(f"Output: {out_csv}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load
    df = pd.read_csv(input_csv)
    if args.limit is not None:
        df = df.iloc[: max(0, int(args.limit))].copy()

    RAW_COL = "LLM_eval_raw"
    SCORE_COL = "LLM_eval_score"
    for col in (RAW_COL, SCORE_COL):
        if col not in df.columns:
            df[col] = ""

    # Resume logic
    if args.resume and out_csv.exists() and input_csv.resolve() != out_csv.resolve():
        prev = pd.read_csv(out_csv)
        for col in (RAW_COL, SCORE_COL):
            if col in prev.columns:
                vals = list(prev[col])[:len(df)]
                if len(vals) < len(df): vals += [""]*(len(df)-len(vals))
                df[col] = vals

    indices = list(df.index)
    todo: List[int] = [i for i in indices if needs_rerun(str(df.at[i, RAW_COL]), str(df.at[i, SCORE_COL]))]

    # Preflight: show column status and first prompt length
    print(f"Using text column: {args.text_column} ({'present' if args.text_column in df.columns else 'MISSING -> concatenating string columns'})")
    if todo:
        sample_prompt = build_user_prompt(df.loc[todo[0]], args.text_column, eval_prompt)
        plen = len(sample_prompt)
        if args.truncate_input_chars and plen > args.truncate_input_chars:
            print(f"First prompt length: {plen} chars (will truncate to {args.truncate_input_chars})")
        else:
            print(f"First prompt length: {plen} chars")

    print(f"resume: {len(indices)-len(todo)} done / {len(indices)} total, {len(todo)} to run")
    if not todo:
        df.to_csv(out_csv, index=False)
        print(f"Saved (unchanged): {out_csv}")
        return

    # Client
    if args.api == "chat":
        client = HFChatClient(args.router_url, args.hf_token, args.model, timeout=args.request_timeout)
        sys_text = args.system
    else:
        client = HFResponsesClient(args.router_url, args.hf_token, args.model, timeout=args.request_timeout)
        sys_text = args.system

    MAX_RETRIES = max(0, int(args.max_retries))
    BASE_BACKOFF = max(0.1, float(args.base_backoff))
    TRUNC = max(0, int(args.truncate_input_chars or 0))
    LOG_ERRS_LEFT = [max(0, int(args.log_first_n_errors))]

    def maybe_truncate(s: str) -> str:
        if TRUNC and len(s) > TRUNC:
            return s[:TRUNC] + "\n\n(…truncated…)"
        return s

    def call_one(i: int) -> Tuple[int, str, Optional[str]]:
        user_prompt = maybe_truncate(build_user_prompt(df.loc[i], args.text_column, eval_prompt))
        attempt = 0
        while True:
            try:
                raw = (client.infer(
                    sys_text,
                    user_prompt,
                    args.reasoning_effort,
                    args.max_output_tokens,
                    args.temperature,
                ) or "").strip()
                if not raw:
                    raise RuntimeError("empty_response")
                score = parse_score(raw)
                return i, raw, score
            except Exception as e:
                attempt += 1
                if LOG_ERRS_LEFT[0] > 0:
                    print(f"[row {i}] {type(e).__name__}: {e}")
                    LOG_ERRS_LEFT[0] -= 1
                if attempt > MAX_RETRIES:
                    return i, f"ERROR: {type(e).__name__}: {e}", None
                time.sleep(BASE_BACKOFF * (2 ** (attempt - 1)))

    # Parallel execution + incremental save
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex, tqdm(total=len(todo), desc="Requests") as pbar:
        futs = [ex.submit(call_one, i) for i in todo]
        for fut in as_completed(futs):
            i, raw, score = fut.result()
            df.at[i, RAW_COL] = raw or ""
            df.at[i, SCORE_COL] = "" if score is None else score
            df.to_csv(out_csv, index=False)
            pbar.update(1)

    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()

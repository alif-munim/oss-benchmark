#!/usr/bin/env python3
# benchmarks/hf_bench.py
#
# Run GPT-OSS (20B or 120B) via Hugging Face Inference Providers on a CSV.
# Resumable: re-runs only rows that are empty, unparsable, or errored.
# Adds retries with exponential backoff. Supports Chat or Responses API.
#
# This version runs the "diagnostic Likert scoring" task using the SAME prompt
# as in your reference script. It expects a column (default: Disease_description).
#
# Uses provider default generation settings unless you explicitly override via CLI.
#
# Requires: pip install openai pandas tqdm

import os, re, argparse, time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ---------- Prompt (same as your reference) ----------
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

# Minimal system message (kept configurable via --system)
DEFAULT_SYSTEM = ""

# Column with the task text (defaults to "Disease_description")
DEFAULT_TEXT_COL = "Disease_description"

# ---------- Parsing ----------
# Accept 1..5 with optional .5 (e.g., 1, 2, 3.5, 4.0). We coerce to "1".."5" or "*.*".
SCORE_RE = re.compile(r"\b([1-5](?:\.5|\.0)?)\b")

def extract_final(text: str) -> str:
    # Handles providers that wrap output in special channels. If not present, return raw.
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
        # Fallback: concatenate any string-ish fields
        task_text = "\n".join([str(v) for v in row.values if isinstance(v, str)]).strip()
    return EVAL_PROMPT + task_text

def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s)

def out_path(input_csv: Path, model: str, results_dir: Path,
             reasoning: Optional[str], max_out: Optional[int], api: str) -> Path:
    tag = model.replace("/", "-").replace(":", "-")
    parts = [input_csv.stem, tag, api, "eval"]
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
    def __init__(self, base_url: str, api_key: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

class HFChatClient(HFClientBase):
    def infer(self, system_text: str, user_text: str,
              reasoning_effort: Optional[str], max_output_tokens: Optional[int],
              temperature: Optional[float]) -> str:
        messages = []
        if system_text and system_text.strip():
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_text})

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        # Only override provider defaults if user supplied flags
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if max_output_tokens is not None:
            kwargs["max_tokens"] = int(max_output_tokens)

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
        # Only override defaults if provided
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = int(max_output_tokens)

        r = self.client.responses.create(**kwargs)
        text = getattr(r, "output_text", None) or ""
        if text and text.strip():
            return text.strip()

        # Fallback parse from blocks
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
                    help="Default 'chat' works with :cerebras and many providers. Use 'responses' if preferred.")
    ap.add_argument("--router_url", default="https://router.huggingface.co/v1")
    ap.add_argument("--hf_token", default=os.getenv("HF_TOKEN"))
    ap.add_argument("--system", default=DEFAULT_SYSTEM)
    ap.add_argument("--text_column", default=DEFAULT_TEXT_COL, help="Column containing the task text (default: Disease_description)")
    # Use provider defaults unless user overrides these:
    ap.add_argument("--reasoning_effort", choices=["low","medium","high"], default=None)
    ap.add_argument("--max_output_tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
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

    # Output path
    if args.output_csv:
        out_csv = Path(args.output_csv)
    else:
        out_csv = out_path(input_csv, args.model, Path(args.results),
                           args.reasoning_effort, args.max_output_tokens, args.api)
    print(f"Output: {out_csv}")

    # Load
    df = pd.read_csv(input_csv)
    # Columns for raw model text and parsed score
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

    print(f"resume: {len(indices)-len(todo)} done / {len(indices)} total, {len(todo)} to run")
    if not todo:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"Saved (unchanged): {out_csv}")
        return

    # Client
    if args.api == "chat":
        client = HFChatClient(args.router_url, args.hf_token, args.model)
        sys_text = args.system
    else:
        client = HFResponsesClient(args.router_url, args.hf_token, args.model)
        sys_text = args.system

    MAX_RETRIES = max(0, int(args.max_retries))
    BASE_BACKOFF = max(0.1, float(args.base_backoff))

    def call_one(i: int) -> Tuple[int, str, Optional[str]]:
        user_prompt = build_user_prompt(df.loc[i], args.text_column)
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

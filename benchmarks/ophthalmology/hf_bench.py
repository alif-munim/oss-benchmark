#!/usr/bin/env python3
# benchmarks/hf_bench_dataset.py
#
# Run GPT-OSS (20B or 120B) via Hugging Face Inference Providers on a CSV.
# Resumable: re-runs only rows that are empty, unparsable, or errored.
# Adds retries with exponential backoff. Supports Chat or Responses API.
#
# Usage (20B Fireworks):
#   export HF_TOKEN=hf_...
#   python benchmarks/hf_bench_dataset.py data/datasets/ophthalmology.csv \
#     --model openai/gpt-oss-20b:fireworks-ai \
#     --api chat \
#     --reasoning_effort high \
#     --max_output_tokens 2048 \
#     --workers 4 \
#     --results results
#
# Resume on the SAME output file (re-run only missing/errored rows):
#   python benchmarks/hf_bench_dataset.py results/ophthalmology_openai-gpt-oss-20b-fireworks-ai_chat_re-high_max2048.csv \
#     --model openai/gpt-oss-20b:fireworks-ai \
#     --api chat \
#     --reasoning_effort high \
#     --max_output_tokens 2048 \
#     --workers 2 \
#     --results results \
#     --resume \
#     --output_csv results/ophthalmology_openai-gpt-oss-20b-fireworks-ai_chat_re-high_max2048.csv
#
# Requires: pip install openai pandas tqdm

import os, re, argparse, time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ---------- Prompts / parsing ----------
DEFAULT_SYSTEM = (
    "You are a careful ophthalmology question-answering assistant. "
    "You will be given a multiple-choice case with options labeled A–Z. "
    "Some questions have a single correct answer, while others have multiple correct answers. "
    "Select ALL correct answers. If only one answer is correct, return just that single letter. "
    "Respond with ONLY the capital letters (A–Z), concatenated together with no spaces or punctuation "
    "(e.g., 'ABE' for multiple answers, or 'D' if only one). Do not explain."
)
DEFAULT_USER_TEMPLATE = (
    "Case:\n{case_text}\n\n"
    "Task: Choose the correct answer(s).\n\n"
    "Return: One or more letters from A to Z, concatenated with no spaces (e.g., ABE or D).\n"
)

# Allow any sequence of 1–26 capital letters A–Z (no spaces)
LETTERS_RUN_RE = re.compile(r"\b([A-Z]{1,26})\b")
FINAL_CH_RE = re.compile(r"<\|channel\|>final<\|message\|>(.*)$", re.DOTALL)

def extract_final(text: str) -> str:
    if not text:
        return ""
    m = FINAL_CH_RE.search(text)
    return (m.group(1).strip() if m else text.strip())

def parse_choice(raw: str) -> Optional[str]:
    if not raw:
        return None
    m = LETTERS_RUN_RE.search(raw.strip().upper())
    return m.group(1) if m else None

def build_user_prompt(row: pd.Series, user_template: str) -> str:
    q_col = next((c for c in row.index if c.lower() == "question"), None)
    if q_col:
        case_text = str(row[q_col]).strip()
    else:
        case_text = "\n".join([str(v) for v in row.values if isinstance(v, str)]).strip()
    return user_template.format(case_text=case_text)

def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s)

def out_path(input_csv: Path, model: str, results_dir: Path,
             reasoning: Optional[str], max_out: Optional[int], api: str) -> Path:
    tag = model.replace("/", "-").replace(":", "-")
    parts = [input_csv.stem, tag, api]
    if reasoning: parts.append(f"re-{sanitize(reasoning)}")
    if max_out: parts.append(f"max{int(max_out)}")
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / ("_".join(parts) + ".csv")

def is_done_choice(s: str) -> bool:
    return isinstance(s, str) and re.fullmatch(r"[A-Z]{1,26}", (s.strip() or "")) is not None

def needs_rerun(raw: str, choice: str) -> bool:
    # Re-run if:
    #  - no valid A–Z sequence
    #  - raw empty
    #  - raw starts with "ERROR:"
    if is_done_choice(choice or ""):
        return False
    if not raw or not raw.strip():
        return True
    if raw.strip().upper().startswith("ERROR:"):
        return True
    # prose without a parseable choice → re-run
    return True

# ---------- Clients ----------
class HFClientBase:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.0):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature

class HFChatClient(HFClientBase):
    def infer(self, system_text: str, user_text: str,
              reasoning_effort: Optional[str], max_output_tokens: Optional[int]) -> str:
        sys = system_text if not reasoning_effort else f"{system_text}\nReturn only the letters A–Z.\nReasoning: {reasoning_effort}."
        cc = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user_text},
            ],
            temperature=self.temperature,
            max_tokens=(int(max_output_tokens) if max_output_tokens is not None else 512),
        )
        return extract_final((cc.choices[0].message.content or "").strip())

class HFResponsesClient(HFClientBase):
    def infer(self, system_text: str, user_text: str,
              reasoning_effort: Optional[str], max_output_tokens: Optional[int]) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            "temperature": self.temperature,
        }
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = int(max_output_tokens)
        r = self.client.responses.create(**kwargs)

        text = getattr(r, "output_text", None) or ""
        if text and text.strip():
            return extract_final(text.strip())
        blocks = []
        for blk in (getattr(r, "output", None) or []):
            for c in (getattr(blk, "content", None) or []):
                if getattr(c, "type", "") == "output_text" and getattr(c, "text", None):
                    blocks.append(c.text)
        return extract_final("\n".join(blocks).strip())

# ---------- Runner ----------
def main():
    ap = argparse.ArgumentParser(description="Run GPT-OSS (20B/120B) via HF Inference Providers on a CSV (resumable).")
    ap.add_argument("input_csv")
    ap.add_argument("--model", required=True, help="e.g., openai/gpt-oss-20b:fireworks-ai or openai/gpt-oss-120b:cerebras")
    ap.add_argument("--api", choices=["chat", "responses"], default="chat",
                    help="Default 'chat' works with :cerebras and many providers. Use 'responses' if preferred.")
    ap.add_argument("--router_url", default="https://router.huggingface.co/v1")
    ap.add_argument("--hf_token", default=os.getenv("HF_TOKEN"))
    ap.add_argument("--system", default=DEFAULT_SYSTEM)
    ap.add_argument("--user_template", default=DEFAULT_USER_TEMPLATE)
    ap.add_argument("--reasoning_effort", choices=["low","medium","high"], default=None)
    ap.add_argument("--max_output_tokens", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--results", default="results")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
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
    for col in ("model_answer_raw", "model_answer_choice"):
        if col not in df.columns:
            df[col] = ""

    # Resume logic: if resuming from an output CSV, merge its cols back
    if args.resume and out_csv.exists() and input_csv.resolve() != out_csv.resolve():
        prev = pd.read_csv(out_csv)
        for col in ("model_answer_raw", "model_answer_choice"):
            if col in prev.columns:
                vals = list(prev[col])[:len(df)]
                if len(vals) < len(df): vals += [""]*(len(df)-len(vals))
                df[col] = vals

    indices = list(df.index)
    todo: List[int] = [i for i in indices if needs_rerun(str(df.at[i,"model_answer_raw"]),
                                                         str(df.at[i,"model_answer_choice"]))]

    print(f"resume: {len(indices)-len(todo)} done / {len(indices)} total, {len(todo)} to run")
    if not todo:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"Saved (unchanged): {out_csv}")
        return

    # Client
    if args.api == "chat":
        client = HFChatClient(args.router_url, args.hf_token, args.model, temperature=args.temperature)
        sys_text = args.system  # reasoning duplicated in client
    else:
        client = HFResponsesClient(args.router_url, args.hf_token, args.model, temperature=args.temperature)
        sys_text = args.system if not args.reasoning_effort else f"{args.system}\nReasoning: {args.reasoning_effort}."

    MAX_RETRIES = max(0, int(args.max_retries))
    BASE_BACKOFF = max(0.1, float(args.base_backoff))

    def call_one(i: int) -> Tuple[int, str, Optional[str]]:
        user_prompt = build_user_prompt(df.loc[i], args.user_template)
        attempt = 0
        while True:
            try:
                raw = (client.infer(sys_text, user_prompt, args.reasoning_effort, args.max_output_tokens) or "").strip()
                if not raw:
                    raise RuntimeError("empty_response")
                choice = parse_choice(raw)
                return i, raw, choice
            except Exception as e:
                attempt += 1
                if attempt > MAX_RETRIES:
                    return i, f"ERROR: {type(e).__name__}: {e}", None
                time.sleep(BASE_BACKOFF * (2 ** (attempt - 1)))

    # Parallel execution + incremental save
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex, tqdm(total=len(todo), desc="Requests") as pbar:
        futs = [ex.submit(call_one, i) for i in todo]
        for fut in as_completed(futs):
            i, raw, choice = fut.result()
            df.at[i, "model_answer_raw"] = raw or ""
            df.at[i, "model_answer_choice"] = "" if choice is None else choice
            df.to_csv(out_csv, index=False)
            pbar.update(1)

    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()

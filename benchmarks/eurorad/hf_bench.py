#!/usr/bin/env python3
# benchmarks/eurorad/hf_bench.py
#
# Run GPT-OSS (20B / 120B) via Hugging Face Inference Providers on a radiology CSV.
# Resumable: only re-runs rows with empty/errored raw outputs or unmapped answers.
# Adds retries with exponential backoff. Supports Chat or Responses API.
#
# Expected CSV headers:
#   case_id, OriginalDescription, PostDescription, DifferentialDiagnosisList, FinalDiagnosis
#
# Output columns added/used:
#   model_answer_raw, model_answer, match_type, correct, options
#
# Requires: pip install openai pandas tqdm

import os, re, argparse, time, unicodedata, difflib
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ---------- Prompts / parsing ----------
DEFAULT_SYSTEM = (
    "You are a careful radiology diagnosis selector.\n"
    "Given a case description and a finite list of candidate diagnoses, "
    "select the single most likely final diagnosis FROM THE LIST.\n"
    "Response rules:\n"
    "1) Output EXACTLY one option, copied VERBATIM from the list.\n"
    "2) Output ONLY the diagnosis text. No explanation. No punctuation. No quotes."
)

DEFAULT_USER_TEMPLATE = (
    "Case ID: {case_id}\n\n"
    "Case description:\n{case_text}\n\n"
    "Candidate diagnoses (choose ONE):\n{options_block}\n\n"
    "Return exactly one option from the list above, copied verbatim."
)

FINAL_CH_RE = re.compile(r"<\|channel\|>final<\|message\|>(.*)$", re.DOTALL)

def extract_final(text: str) -> str:
    if not text:
        return ""
    m = FINAL_CH_RE.search(text)
    return (m.group(1).strip() if m else text.strip())

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

    # exact
    if raw in options:
        return raw, "exact"

    # normalized exact
    norm2opt = {norm_text(o): o for o in options}
    nr = norm_text(raw)
    if nr in norm2opt:
        return norm2opt[nr], "normalized"

    # fuzzy (strict)
    cand = difflib.get_close_matches(raw, options, n=1, cutoff=0.8)
    if cand:
        return cand[0], "fuzzy"

    # fuzzy on normalized
    norm_opts = list(norm2opt.keys())
    candn = difflib.get_close_matches(nr, norm_opts, n=1, cutoff=0.9)
    if candn:
        return norm2opt[candn[0]], "fuzzy"

    return "", "no_match"

def build_user_prompt(row: pd.Series, user_template: str = DEFAULT_USER_TEMPLATE) -> str:
    case_id = str(row.get("case_id", "")).strip()
    desc = str(row.get("PostDescription") or row.get("OriginalDescription") or "").strip()
    options = build_options_list(row.get("DifferentialDiagnosisList", ""))
    opts_block = "\n".join(f"- {o}" for o in options) if options else "- (no options provided)"
    return user_template.format(case_id=case_id, case_text=desc, options_block=opts_block)

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

def needs_rerun(raw: str, mapped: str) -> bool:
    if isinstance(mapped, str) and mapped.strip():
        return False
    if not raw or not str(raw).strip():
        return True
    if str(raw).strip().upper().startswith("ERROR:"):
        return True
    return True

# ---------- Clients ----------
class HFClientBase:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: Optional[float] = None):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature

class HFChatClient(HFClientBase):
    def infer(self, system_text: str, user_text: str,
              reasoning_effort: Optional[str], max_output_tokens: Optional[int]) -> str:
        sys = system_text if not reasoning_effort else f"{system_text}\nReasoning: {reasoning_effort}."
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys},
                {"role": "user", "content": user_text},
            ],
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if max_output_tokens is not None:
            kwargs["max_tokens"] = int(max_output_tokens)
        cc = self.client.chat.completions.create(**kwargs)
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
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
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
    ap = argparse.ArgumentParser(description="Run GPT-OSS via HF Inference Providers on a radiology CSV (resumable).")
    ap.add_argument("input_csv")
    ap.add_argument("--model", required=True, help="e.g., openai/gpt-oss-20b:fireworks-ai or openai/gpt-oss-120b:cerebras")
    ap.add_argument("--api", choices=["chat", "responses"], default="chat")
    ap.add_argument("--router_url", default="https://router.huggingface.co/v1")
    ap.add_argument("--hf_token", default=os.getenv("HF_TOKEN"))
    ap.add_argument("--system", default=DEFAULT_SYSTEM)
    ap.add_argument("--user_template", default=DEFAULT_USER_TEMPLATE)
    ap.add_argument("--reasoning_effort", choices=["low","medium","high"], default=None)
    ap.add_argument("--max_output_tokens", type=int, default=None)   # use provider default if None
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--results", default="results")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--temperature", type=float, default=None)       # use provider default if None
    ap.add_argument("--output_csv", default=None, help="Write to this path (also use when resuming).")
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

    # Ensure output columns exist
    for col in ("model_answer_raw", "model_answer", "match_type", "correct", "options"):
        if col not in df.columns:
            df[col] = ""

    # Resume logic: if resuming from an output CSV, merge its cols back
    if args.resume and out_csv.exists() and input_csv.resolve() != out_csv.resolve():
        prev = pd.read_csv(out_csv)
        for col in ("model_answer_raw", "model_answer", "match_type", "correct", "options"):
            if col in prev.columns:
                vals = list(prev[col])[:len(df)]
                if len(vals) < len(df): vals += [""]*(len(df)-len(vals))
                df[col] = vals

    indices = list(df.index)
    todo: List[int] = [i for i in indices if needs_rerun(str(df.at[i,"model_answer_raw"]),
                                                         str(df.at[i,"model_answer"]))]

    print(f"resume: {len(indices)-len(todo)} done / {len(indices)} total, {len(todo)} to run")
    if not todo:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"Saved (unchanged): {out_csv}")
        if "FinalDiagnosis" in df.columns:
            gold = df["FinalDiagnosis"].astype(str).apply(norm_text)
            pred = df["model_answer"].astype(str).apply(norm_text)
            acc = (gold == pred).mean() if len(df) else 0.0
            print(f"Accuracy: {acc:.3f}")
        return

    # Client
    if args.api == "chat":
        client = HFChatClient(args.router_url, args.hf_token, args.model, temperature=args.temperature)
        sys_text = args.system
    else:
        client = HFResponsesClient(args.router_url, args.hf_token, args.model, temperature=args.temperature)
        sys_text = args.system if not args.reasoning_effort else f"{args.system}\nReasoning: {args.reasoning_effort}."

    MAX_RETRIES = max(0, int(args.max_retries))
    BASE_BACKOFF = max(0.1, float(args.base_backoff))

    def call_one(i: int) -> Tuple[int, str, str, str]:
        user_prompt = build_user_prompt(df.loc[i], args.user_template)
        options = build_options_list(df.loc[i].get("DifferentialDiagnosisList", ""))
        attempt = 0
        while True:
            try:
                raw = (client.infer(sys_text, user_prompt, args.reasoning_effort, args.max_output_tokens) or "").strip()
                if not raw:
                    raise RuntimeError("empty_response")
                mapped, mtype = map_to_option(raw, options)
                return i, raw, mapped, mtype
            except Exception as e:
                attempt += 1
                if attempt > MAX_RETRIES:
                    return i, f"ERROR: {type(e).__name__}: {e}", "", "no_match"
                time.sleep(BASE_BACKOFF * (2 ** (attempt - 1)))

    # Parallel execution + incremental save
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex, tqdm(total=len(todo), desc="Requests") as pbar:
        futs = [ex.submit(call_one, i) for i in todo]
        for fut in as_completed(futs):
            i, raw, mapped, mtype = fut.result()
            df.at[i, "model_answer_raw"] = raw or ""
            df.at[i, "model_answer"] = mapped or ""
            df.at[i, "match_type"] = mtype
            df.at[i, "options"] = " | ".join(build_options_list(df.loc[i].get("DifferentialDiagnosisList", "")))
            if "FinalDiagnosis" in df.columns and mapped:
                gold = norm_text(str(df.loc[i, "FinalDiagnosis"]))
                df.at[i, "correct"] = int(norm_text(mapped) == gold)
            else:
                df.at[i, "correct"] = ""
            df.to_csv(out_csv, index=False)
            pbar.update(1)

    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    if "FinalDiagnosis" in df.columns:
        mask = df["model_answer"].astype(str).str.len() > 0
        if mask.any():
            gold = df.loc[mask, "FinalDiagnosis"].astype(str).apply(norm_text)
            pred = df.loc[mask, "model_answer"].astype(str).apply(norm_text)
            acc = (gold == pred).mean() if len(gold) else 0.0
            print(f"Accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust GEPA-style evaluation for Eurorad MC tasks with optional few-shot retrieval and majority voting.

Key features:
- Label-only metric (default)
- Optional GEPA with required reflection LM
- Warmup ping and tolerant LM wrapper (no _settings override)
- Skippable base evaluation (--skip-base-eval)
- Few-shot retrieval (--fewshot-k with inclusion control)
- Majority vote / self-consistency (--vote-k)
- Toggle progress bars (--no-progress)
- Deterministic per-example JSONL dump (--out-jsonl)

Example:
python gepa_eurorad.py \
  --hf-dataset alif-munim/eurorad-omar-120b \
  --model openai/gpt-4.1-mini \
  --reflection-model openai/gpt-4.1-mini \
  --use-gepa --gepa-budget medium \
  --fewshot-k 3 --fewshot-include both \
  --vote-k 5 \
  --skip-base-eval \
  --out-jsonl outputs/run1.jsonl
"""

import os
import sys
import time
import json
import math
import random
import signal
import logging
import argparse
from typing import Any, Optional, List, Tuple

import dspy
from datasets import load_dataset

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------- Safe exceptions / ping ----------
try:
    from openai import APIError, RateLimitError, APITimeoutError, OpenAI
except Exception:
    class APIError(Exception):
        status_code: int = 500
    class RateLimitError(Exception): ...
    class APITimeoutError(Exception): ...
    OpenAI = None  # type: ignore

def openai_ping(model: str, timeout: float) -> tuple[bool, Optional[str]]:
    """Tiny Responses API call to verify access; uses safe max_output_tokens."""
    if OpenAI is None:
        return True, None
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=timeout)
        client.responses.create(
            model=model.replace("openai/", ""),
            input="ping",
            max_output_tokens=32,  # >=16 to avoid 400
        )
        return True, None
    except Exception as e:
        return False, repr(e)

# ---------- Robust LM wrapper (no _settings override) ----------
class RobustWrappedLM(dspy.BaseLM):
    """
    Tolerant wrapper that:
      - Accepts both prompts and messages styles
      - Retries 429 / timeouts / 5xx with exponential backoff
    """
    def __init__(self, inner_lm: dspy.BaseLM, max_retries: int = 8,
                 base_backoff: float = 0.5, backoff_cap: float = 20.0,
                 jitter: float = 0.25, retry_on_5xx: bool = True):
        if not isinstance(inner_lm, dspy.BaseLM):
            raise TypeError(f"inner_lm must be dspy.BaseLM, got {type(inner_lm)}")

        model_name = getattr(inner_lm, "model", None)
        if not model_name:
            try:
                s = getattr(inner_lm, "_settings", None)
                model_name = getattr(s, "model", None)
            except Exception:
                pass
        super().__init__(model=model_name or "wrapped-lm")

        self.inner = inner_lm
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.backoff_cap = backoff_cap
        self.jitter = jitter
        self.retry_on_5xx = retry_on_5xx
        self.retry_on = (RateLimitError, APITimeoutError)

    # Do NOT override _settings

    def __call__(self, *args, **kwargs):
        return self._run(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self._run(*args, **kwargs)

    def _run(self, *args, **kwargs):
        prompts = None
        messages = None

        if args:
            if len(args) == 1 and isinstance(args[0], (list, str)):
                prompts = args[0]
            else:
                prompts = args[0]

        if "prompts" in kwargs and kwargs["prompts"] is not None:
            prompts = kwargs["prompts"]
        if "messages" in kwargs and kwargs["messages"] is not None:
            messages = kwargs["messages"]

        def inner_call():
            if hasattr(self.inner, "generate"):
                if messages is not None:
                    return self.inner.generate(messages=messages, **{k:v for k,v in kwargs.items() if k!="messages"})
                if prompts is not None:
                    return self.inner.generate(prompts, **{k:v for k,v in kwargs.items() if k!="prompts"})
                return self.inner.generate([], **kwargs)
            else:
                if messages is not None:
                    return self.inner(messages=messages, **{k:v for k,v in kwargs.items() if k!="messages"})
                if prompts is not None:
                    return self.inner(prompts, **{k:v for k,v in kwargs.items() if k!="prompts"})
                return self.inner([], **kwargs)

        for attempt in range(self.max_retries + 1):
            try:
                return inner_call()
            except self.retry_on:
                if attempt >= self.max_retries:
                    raise
                time.sleep(self._sleep_time(attempt))
            except APIError as e:
                status = getattr(e, "status_code", 500)
                if self.retry_on_5xx and status >= 500 and attempt < self.max_retries:
                    time.sleep(self._sleep_time(attempt))
                else:
                    raise

    def _sleep_time(self, attempt: int) -> float:
        backoff = min(self.base_backoff * (2 ** attempt), self.backoff_cap)
        factor = 1.0 + random.uniform(-self.jitter, self.jitter)
        return max(0.05, backoff * factor)

# ---------- Signature & Program ----------
class DiagnoseSig(dspy.Signature):
    """Read the case and output exactly ONE diagnosis from the candidate list, copied verbatim."""
    text = dspy.InputField()
    final_answer = dspy.OutputField(desc="Return exactly one option from the list above, copied verbatim.")

class EuroradProgram(dspy.Module):
    """
    Optionally inject few-shot exemplars and support vote-k majority voting.
    """
    def __init__(self, vote_k: int = 1, instruction_preamble: Optional[str] = None):
        super().__init__()
        self.solve = dspy.ChainOfThought(DiagnoseSig)
        self.vote_k = max(1, int(vote_k))
        self.instruction_preamble = instruction_preamble or (
            "Task: Select and return exactly ONE diagnosis from the candidate list, copied verbatim. "
            "Do not include any extra words or punctuation."
        )

        # few-shot exemplars are set externally via set_fewshot_examples()
        self._fewshot_block = ""

    def set_fewshot_examples(self, examples_text_block: str):
        self._fewshot_block = examples_text_block or ""

    def _format_input(self, text: str) -> str:
        parts = [self.instruction_preamble]
        if self._fewshot_block:
            parts.append("\n### Examples\n" + self._fewshot_block.strip())
        parts.append("\n### Case\n" + text.strip())
        return "\n\n".join(parts).strip()

    def forward(self, text: str):
        prompt = self._format_input(text)
        if self.vote_k == 1:
            out = self.solve(text=prompt)
            return dspy.Prediction(final_answer=out.final_answer)

        # Majority vote with K samples
        votes = []
        for _ in range(self.vote_k):
            out = self.solve(text=prompt)
            votes.append(str(getattr(out, "final_answer", "")).strip())
        best = self._majority(votes)
        return dspy.Prediction(final_answer=best)

    @staticmethod
    def _majority(items: List[str]) -> str:
        from collections import Counter
        c = Counter([i for i in items if i])
        if not c:
            return ""
        most_common, _ = c.most_common(1)[0]
        return most_common

# ---------- Metrics ----------
def normalize(s: Any) -> str:
    return " ".join(str(s).strip().split())

def label_only_metric_with_feedback_factory(label_field: str, solution_field: str):
    def metric(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None):
        gold_ans = normalize(getattr(gold, label_field))
        pred_raw = getattr(pred, "final_answer", "")
        pred_ans = normalize(pred_raw)
        score = 1.0 if pred_ans == gold_ans else 0.0
        fb = (f"Correct. Gold='{gold_ans}'."
              if score == 1.0 else f"Incorrect. Gold='{gold_ans}'. Pred='{pred_raw}'.")
        sol = getattr(gold, solution_field, None)
        if sol:
            fb += f"\n\nSolution / rationale:\n{sol}"
        return dspy.Prediction(score=score, feedback=fb)
    return metric

def as_feedback_metric(metric_fn):
    def wrapped(gold, pred, trace=None, pred_name=None, pred_trace=None):
        m = metric_fn(gold, pred, trace=trace, pred_name=pred_name, pred_trace=pred_trace)
        fb = getattr(m, "feedback", "")
        if not isinstance(fb, str):
            fb = str(fb)
        return dspy.Prediction(score=m.score, feedback=fb)
    return wrapped

# ---------- Data ----------
def split_dataset(hf_dataset: str, text_field: str, label_field: str, solution_field: str,
                  seed: int, train_frac: float, val_frac: float, test_frac: float):
    ds = load_dataset(hf_dataset)
    rows = []
    if "train" in ds:
        rows.extend(list(ds["train"]))
    for k in ("validation", "val", "dev", "test"):
        if k in ds:
            rows.extend(list(ds[k]))
    if not rows:
        raise ValueError("No data rows found.")

    rng = random.Random(seed)
    rng.shuffle(rows)

    total = train_frac + val_frac + test_frac
    if total <= 0:
        raise ValueError("Fractions must sum > 0.")
    train_frac /= total
    val_frac /= total
    test_frac /= total

    n = len(rows)
    tr = int(round(n * train_frac))
    va = int(round(n * val_frac))
    te = n - tr - va

    def to_examples(rr):
        out = []
        for r in rr:
            text = r.get(text_field, "")
            label = r.get(label_field, "")
            sol = r.get(solution_field, "")
            ex = dspy.Example({"text": text, label_field: label, solution_field: sol}).with_inputs("text")
            out.append(ex)
        return out

    return to_examples(rows[:tr]), to_examples(rows[tr:tr+va]), to_examples(rows[tr+va:tr+va+te])

# ---------- Few-shot retrieval (lexical Jaccard) ----------
def _tokenize(s: str) -> set:
    return set([t for t in "".join([c.lower() if c.isalnum() else " " for c in s]).split() if t])

def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / float(len(a | b))

def build_fewshot_block(
    query_text: str,
    pool: List[dspy.Example],
    k: int,
    include_mode: str,
    label_field: str,
    solution_field: str,
) -> Tuple[str, List[int]]:
    """
    include_mode in {"input", "label", "reasoning", "both"}:
      - "input": only include the input text (no answers), for pattern priming
      - "label": include input + the gold final_answer
      - "reasoning": include input + rationale
      - "both": include input + rationale + final_answer
    """
    k = max(0, int(k))
    if k == 0 or not pool:
        return "", []

    qtok = _tokenize(query_text)
    scored = []
    for idx, ex in enumerate(pool):
        itok = _tokenize(getattr(ex, "text", ""))
        scored.append((jaccard(qtok, itok), idx))
    scored.sort(key=lambda x: x[0], reverse=True)
    take = [idx for _, idx in scored[:k]]

    lines = []
    for i, idx in enumerate(take, start=1):
        ex = pool[idx]
        text = getattr(ex, "text", "").strip()
        label = getattr(ex, label_field, "").strip()
        sol = getattr(ex, solution_field, "").strip()

        section = [f"Example {i}"]
        section.append(f"Case:\n{text}")
        if include_mode in ("label", "both"):
            section.append(f"Gold diagnosis:\n{label}")
        if include_mode in ("reasoning", "both"):
            # keep short: cap very long rationales
            if len(sol) > 2000:
                sol = sol[:2000] + " …"
            section.append(f"Why (summary):\n{sol if sol else '(not provided)'}")
        lines.append("\n".join(section).strip())

    block = "\n\n---\n\n".join(lines)
    return block, take

# ---------- Utility: structured eval & dump ----------
def run_and_dump_jsonl(
    prog: EuroradProgram,
    dataset: List[dspy.Example],
    label_field: str,
    solution_field: str,
    out_path: Optional[str] = None,
    fewshot_cfg: Optional[dict] = None,
) -> dict:
    """
    Runs the program over dataset, computes metric, and optionally writes JSONL.
    Returns summary dictionary.
    """
    metric_fn = label_only_metric_with_feedback_factory(label_field, solution_field)
    rows = []
    n_correct = 0

    for ex in dataset:
        text = getattr(ex, "text", "")
        gold = getattr(ex, label_field, "")
        sol = getattr(ex, solution_field, "")

        # (Re)build few-shot block per-item to avoid leaking info across queries
        if fewshot_cfg:
            block, _ = build_fewshot_block(
                query_text=text,
                pool=fewshot_cfg["pool"],
                k=fewshot_cfg["k"],
                include_mode=fewshot_cfg["include"],
                label_field=label_field,
                solution_field=solution_field,
            )
            prog.set_fewshot_examples(block)
        else:
            prog.set_fewshot_examples("")

        pred = prog(text=text)
        pred_raw = getattr(pred, "final_answer", "")
        m = metric_fn(ex, pred)

        n_correct += int(getattr(m, "score", 0.0) > 0.5)
        rows.append({
            "text": text,
            "example_final_answer": gold,
            "reasoning": sol,
            "pred_final_answer": pred_raw,
            "metric_score": getattr(m, "score", 0.0),
            "metric_feedback": getattr(m, "feedback", ""),
        })

    acc = n_correct / max(1, len(dataset))
    summary = {"count": len(dataset), "correct": n_correct, "accuracy": acc}

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info("Wrote detailed predictions to %s", out_path)

    return summary

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dataset", required=True, type=str)
    ap.add_argument("--model", required=True, type=str, help="e.g., openai/gpt-4.1-mini or openai/gpt-4o-mini")

    # Reflection LM for GEPA
    ap.add_argument("--reflection-model", type=str, default=None,
                    help="Model for GEPA reflection (defaults to --model)")
    ap.add_argument("--reflection-temperature", type=float, default=1.0)
    ap.add_argument("--reflection-max-tokens", type=int, default=4096)

    # Fields match your dataset
    ap.add_argument("--text-field", default="input")
    ap.add_argument("--label-field", default="final_answer")
    ap.add_argument("--solution-field", default="reasoning")

    # Splits
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--test-frac", type=float, default=0.25)

    # Runtime / LM settings
    ap.add_argument("--num-threads", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bars / tables during DSPy eval")

    # Few-shot & voting
    ap.add_argument("--fewshot-k", type=int, default=0, help="Number of retrieved few-shot exemplars per case")
    ap.add_argument("--fewshot-include", type=str, default="label",
                    choices=["input", "label", "reasoning", "both"],
                    help="What to include in the few-shot exemplars")
    ap.add_argument("--vote-k", type=int, default=1, help="Self-consistency: number of votes to sample per case")

    # GEPA options
    ap.add_argument("--use-gepa", action="store_true")
    ap.add_argument("--gepa-budget", type=str, default="medium",
                    choices=["light", "medium", "heavy", "auto"],
                    help="GEPA preset; 'auto' will be mapped to 'medium'.")
    ap.add_argument("--gepa-max-full-evals", type=int, default=0)

    # Control flow
    ap.add_argument("--skip-base-eval", action="store_true",
                    help="Skip the initial unoptimized evaluation")

    # Outputs
    ap.add_argument("--out-jsonl", type=str, default=None,
                    help="Write per-example predictions to this JSONL")

    args = ap.parse_args()

    # Warmup sanity ping
    ok, err = openai_ping(model=args.model, timeout=args.timeout)
    if not ok:
        log.warning("OpenAI warmup ping failed (non-fatal): %s", err)

    # Primary LM
    base_inner = dspy.LM(
        model=args.model,
        api_key=os.environ.get("OPENAI_API_KEY"),
        request_timeout=args.timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    base_lm = RobustWrappedLM(base_inner)
    dspy.configure(lm=base_lm)

    # Reflection LM (required by GEPA)
    refl_model = args.reflection_model or args.model
    reflection_inner = dspy.LM(
        model=refl_model,
        api_key=os.environ.get("OPENAI_API_KEY"),
        request_timeout=args.timeout,
        max_tokens=args.reflection_max_tokens,
        temperature=args.reflection_temperature,
    )
    reflection_lm = RobustWrappedLM(reflection_inner)

    # Load data
    trainset, devset, testset = split_dataset(
        hf_dataset=args.hf_dataset,
        text_field=args.text_field,
        label_field=args.label_field,
        solution_field=args.solution_field,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )
    log.info(f"Split sizes -> train: {len(trainset)} | val: {len(devset)} | test: {len(testset)}")
    if len(testset) == 0:
        log.error("Test set is empty.")
        sys.exit(1)

    # Warmup through pipeline
    if args.warmup > 0 and len(testset) > 0:
        log.info(f"Warmup: running {args.warmup} sample(s) through the program…")
        p = EuroradProgram(vote_k=max(1, args.vote_k))
        for i in range(min(args.warmup, len(testset))):
            _ = p(text=testset[i].text)

    # Metric & evaluator helper
    metric_fn = label_only_metric_with_feedback_factory(args.label_field, args.solution_field)

    def evaluate_prog(prog, tag=""):
        kwargs = dict(
            num_threads=args.num_threads,
            display_progress=not args.no_progress,
            display_table=0 if args.no_progress else 5,
            max_errors=100,
            provide_traceback=True,   # show useful errors instead of swallowing them
        )
        evaluate = dspy.Evaluate(metric=metric_fn, devset=testset, **kwargs)
        log.info(f"Evaluating {tag or 'program'}…")
        return evaluate(prog)

    # Prepare a base program with voting and (later) few-shot injection
    base_prog = EuroradProgram(vote_k=max(1, args.vote_k))

    # 1) Optional base eval
    if not args.skip_base_eval:
        _ = evaluate_prog(base_prog, tag="unoptimized")
    else:
        log.info("Skipping unoptimized evaluation (--skip-base-eval).")

    # 2) Optional GEPA optimization
    optimized_prog = None
    if args.use_gepa:
        from dspy import GEPA
        log.info("Starting GEPA optimization…")

        metric_for_gepa = as_feedback_metric(metric_fn)
        gepa_kwargs = dict(
            metric=metric_for_gepa,
            num_threads=args.num_threads,
            track_stats=True,
            track_best_outputs=True,
            reflection_lm=reflection_lm,
        )

        # Prefer explicit cap if provided
        if args.gepa_max_full_evals > 0:
            gepa_kwargs["max_full_evals"] = args.gepa_max_full_evals
        else:
            # Map 'auto' -> 'medium' to satisfy DSPy versions that require a fixed preset
            preset = args.gepa_budget
            if preset == "auto":
                preset = "medium"
                log.info("Mapping --gepa-budget auto -> 'medium'.")
            gepa_kwargs["auto"] = preset

        compiler = GEPA(**gepa_kwargs)
        optimized_prog = compiler.compile(
            student=EuroradProgram(vote_k=max(1, args.vote_k)),
            trainset=trainset if len(trainset) > 0 else devset,
            valset=devset if len(devset) > 0 else testset,
        )
        _ = evaluate_prog(optimized_prog, tag="GEPA-optimized")
    else:
        if args.skip_base_eval:
            log.info("Nothing else to run (skipped base eval and GEPA disabled).")

    # ---------- Final explicit evaluation & JSONL dump ----------
    final_prog = optimized_prog or base_prog

    # Few-shot config for the explicit dump loop
    fewshot_cfg = None
    if args.fewshot_k > 0 and len(trainset) > 0:
        fewshot_cfg = dict(
            pool=trainset,
            k=args.fewshot_k,
            include=args.fewshot_include,
        )
        log.info("Few-shot retrieval enabled: k=%d, include=%s", args.fewshot_k, args.fewshot_include)

    summary = run_and_dump_jsonl(
        prog=final_prog,
        dataset=testset,
        label_field=args.label_field,
        solution_field=args.solution_field,
        out_path=args.out_jsonl,
        fewshot_cfg=fewshot_cfg,
    )

    log.info("Final explicit test accuracy: %.2f%% (%d/%d)",
             100.0 * summary["accuracy"], summary["correct"], summary["count"])

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(130))
    main()

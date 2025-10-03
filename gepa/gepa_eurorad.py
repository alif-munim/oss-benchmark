#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust GEPA-style evaluation for Eurorad MC tasks.
- Label-only metric (default)
- Optional GEPA with required reflection LM
- Warmup ping and tolerant LM wrapper
- Skippable base evaluation (--skip-base-eval)
"""

import os
import sys
import time
import random
import signal
import logging
import argparse
from typing import Any, Optional, List

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

# ---------- Robust LM wrapper ----------
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
        settings = getattr(inner_lm, "_settings", {}) or {}
        super().__init__(model=settings.get("model", "wrapped-lm"))
        self.inner = inner_lm
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.backoff_cap = backoff_cap
        self.jitter = jitter
        self.retry_on_5xx = retry_on_5xx
        self.retry_on = (RateLimitError, APITimeoutError)

    @property
    def _settings(self):
        return getattr(self.inner, "_settings", {}) or {"model": getattr(self, "model", "wrapped-lm")}

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
            elif args:
                prompts = args[0]
        if "prompts" in kwargs and kwargs["prompts"] is not None:
            prompts = kwargs["prompts"]
        if "messages" in kwargs and kwargs["messages"] is not None:
            messages = kwargs["messages"]

        def inner_call():
            if hasattr(self.inner, "generate"):
                if messages is not None:
                    return self.inner.generate(messages=messages)
                if prompts is not None:
                    return self.inner.generate(prompts)
                return self.inner.generate([])
            else:
                if messages is not None:
                    return self.inner(messages=messages)
                if prompts is not None:
                    return self.inner(prompts)
                return self.inner([])

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
            except Exception:
                raise

    def _sleep_time(self, attempt: int) -> float:
        import random as _r
        backoff = min(self.base_backoff * (2 ** attempt), self.backoff_cap)
        factor = 1.0 + _r.uniform(-self.jitter, self.jitter)
        return max(0.05, backoff * factor)

# ---------- Signature & Program ----------
class DiagnoseSig(dspy.Signature):
    """Read the case and output exactly ONE diagnosis from the candidate list, copied verbatim."""
    text = dspy.InputField()
    final_answer = dspy.OutputField(desc="Return exactly one option from the list above, copied verbatim.")

class EuroradProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.solve = dspy.ChainOfThought(DiagnoseSig)

    def forward(self, text: str):
        out = self.solve(text=text)
        return dspy.Prediction(final_answer=out.final_answer)

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
    ap.add_argument("--warmup", type=int, default=1)

    # GEPA options
    ap.add_argument("--use-gepa", action="store_true")
    ap.add_argument("--gepa-budget", type=str, default="medium",
                    choices=["light", "medium", "heavy", "auto"],
                    help="GEPA preset; 'auto' will be mapped to 'medium'.")
    ap.add_argument("--gepa-max-full-evals", type=int, default=0)

    # Control flow
    ap.add_argument("--skip-base-eval", action="store_true",
                    help="Skip the initial unoptimized evaluation")

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
    if args.warmup > 0:
        log.info(f"Warmup: running {args.warmup} sample(s) through the program…")
        p = EuroradProgram()
        for i in range(min(args.warmup, len(testset))):
            _ = p(text=testset[i].text)

    # Metric & evaluator helper
    metric_fn = label_only_metric_with_feedback_factory(args.label_field, args.solution_field)

    def evaluate_prog(prog, tag=""):
        kwargs = dict(num_threads=args.num_threads, display_progress=True, display_table=5, max_errors=100)
        evaluate = dspy.Evaluate(metric=metric_fn, devset=testset, **kwargs)
        log.info(f"Evaluating {tag or 'program'}…")
        return evaluate(prog)

    # 1) Optional base eval
    if not args.skip_base_eval:
        base_prog = EuroradProgram()
        _ = evaluate_prog(base_prog, tag="unoptimized")
    else:
        log.info("Skipping unoptimized evaluation (--skip-base-eval).")

    # 2) Optional GEPA optimization
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
            # Map 'auto' -> a valid preset (medium) to satisfy DSPy
            preset = args.gepa_budget
            if preset == "auto":
                preset = "medium"
                log.info("Mapping --gepa-budget auto -> 'medium' preset for this DSPy version.")
            gepa_kwargs["auto"] = preset

        compiler = GEPA(**gepa_kwargs)
        optimized_prog = compiler.compile(
            student=EuroradProgram(),
            trainset=trainset if len(trainset) > 0 else devset,
            valset=devset if len(devset) > 0 else testset,
        )
        _ = evaluate_prog(optimized_prog, tag="GEPA-optimized")
    else:
        if args.skip_base_eval:
            log.info("Nothing else to run (skipped base eval and GEPA disabled).")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(130))
    main()

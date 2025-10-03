#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust GEPA-style evaluation for Eurorad MC tasks (label-only metric by default).
- Fixes "__call__ missing 'prompts'" by using a tolerant LM wrapper
- Handles both prompts- and messages-style calls
- Warmup ping with valid max_output_tokens
- Optional GEPA and LLM judge remain easy to add later
"""

import os
import sys
import time
import random
import signal
import logging
import argparse
from typing import List, Union, Optional, Any, Dict

import dspy
from datasets import load_dataset

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------- Safe exceptions ----------
try:
    from openai import APIError, RateLimitError, APITimeoutError, OpenAI
except Exception:
    class APIError(Exception):
        status_code: int = 500
    class RateLimitError(Exception): ...
    class APITimeoutError(Exception): ...
    OpenAI = None  # type: ignore


# ---------- Warmup ping (optional but helpful) ----------
def openai_ping(model: str, timeout: float) -> tuple[bool, Optional[str]]:
    """Tiny Responses API call to verify access; uses safe max_output_tokens."""
    if OpenAI is None:
        # Not fatal; skip if openai package not present.
        return True, None
    try:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=timeout,
        )
        # NOTE: 4o-mini / 4.1-mini require >=16
        client.responses.create(
            model=model.replace("openai/", ""),  # dspy uses "openai/xxx"; OpenAI client expects "xxx"
            input="ping",
            max_output_tokens=32,
        )
        return True, None
    except Exception as e:
        return False, repr(e)


# ---------- Robust LM wrapper (DSPy-compatible) ----------
class RobustWrappedLM(dspy.BaseLM):
    """
    Tolerant wrapper that:
      - Accepts *args, **kwargs; supports both 'prompts' and 'messages'
      - Retries 429 / timeouts / 5xx with exponential backoff + jitter
      - Delegates to an inner dspy.BaseLM (supports .generate(...) and __call__(...))
    """

    def __init__(
        self,
        inner_lm: dspy.BaseLM,
        max_retries: int = 8,
        base_backoff: float = 0.5,
        backoff_cap: float = 20.0,
        jitter: float = 0.25,
        retry_on_5xx: bool = True,
    ):
        if not isinstance(inner_lm, dspy.BaseLM):
            raise TypeError(f"inner_lm must be dspy.BaseLM, got {type(inner_lm)}")
        inner_settings = getattr(inner_lm, "_settings", {}) or {}
        model_name = inner_settings.get("model", "wrapped-lm")
        super().__init__(model=model_name)

        self.inner = inner_lm
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.backoff_cap = backoff_cap
        self.jitter = jitter
        self.retry_on_5xx = retry_on_5xx
        self.retry_on = (RateLimitError, APITimeoutError)

    @property
    def _settings(self):
        # Expose inner settings so DSPy shows the real model
        return getattr(self.inner, "_settings", {}) or {"model": getattr(self, "model", "wrapped-lm")}

    def __call__(self, *args, **kwargs):
        """Tolerant entrypoint: forward to ._run once, return inner's native shape."""
        return self._run(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Tolerant entrypoint used by DSPy in some paths."""
        return self._run(*args, **kwargs)

    # ---- Core runner with retries -------------------------------------------------
    def _run(self, *args, **kwargs):
        # Normalize inputs: accept either positional or kw, prompts or messages
        prompts = None
        messages = None

        if args:
            # If a single positional arg, treat as prompts/messages depending on type
            if len(args) == 1:
                if isinstance(args[0], (list, str)):
                    prompts = args[0]
            else:
                # Rarely used; try first arg as prompts
                prompts = args[0]

        # Keyword paths
        if "prompts" in kwargs and kwargs["prompts"] is not None:
            prompts = kwargs["prompts"]
        if "messages" in kwargs and kwargs["messages"] is not None:
            messages = kwargs["messages"]

        # Inner call function: prefer inner.generate if present
        def inner_call():
            if hasattr(self.inner, "generate"):
                if messages is not None:
                    return self.inner.generate(messages=messages)
                if prompts is not None:
                    return self.inner.generate(prompts)
                # Last resort: call empty; inner may raise
                return self.inner.generate([])
            else:
                if messages is not None:
                    return self.inner(messages=messages)
                if prompts is not None:
                    return self.inner(prompts)
                return self.inner([])

        for attempt in range(self.max_retries + 1):
            try:
                out = inner_call()
                return out
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
                # Unknown error: bubble up
                raise

    def _sleep_time(self, attempt: int) -> float:
        import random as _r
        backoff = min(self.base_backoff * (2 ** attempt), self.backoff_cap)
        factor = 1.0 + _r.uniform(-self.jitter, self.jitter)
        return max(0.05, backoff * factor)


# ---------- Signatures & Program ----------
class DiagnoseSig(dspy.Signature):
    """Read the case and output exactly ONE diagnosis from the candidate list, copied verbatim."""
    text = dspy.InputField()
    final_answer = dspy.OutputField(desc="Return exactly one option from the list above, copied verbatim.")

class EuroradProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        # Chain-of-thought used internally; only final_answer is returned
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
              if score == 1.0
              else f"Incorrect. Gold='{gold_ans}'. Pred='{pred_raw}'.")
        sol = getattr(gold, solution_field, None)
        if sol:
            fb += f"\n\nSolution / rationale:\n{sol}"
        return dspy.Prediction(score=score, feedback=fb)
    return metric


# ---------- Data ----------
def split_dataset(
    hf_dataset: str,
    text_field: str,
    label_field: str,
    solution_field: str,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
):
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
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=1)

    args = ap.parse_args()

    # Warmup sanity ping (best-effort)
    ok, err = openai_ping(model=args.model, timeout=args.timeout)
    if not ok:
        log.warning("OpenAI warmup ping failed (non-fatal): %s", err)

    # Build inner DSPy LM (no temperature set => default)
    inner = dspy.LM(
        model=args.model,
        api_key=os.environ.get("OPENAI_API_KEY"),
        request_timeout=args.timeout,
        max_tokens=args.max_tokens,
    )
    lm = RobustWrappedLM(inner_lm=inner)
    dspy.configure(lm=lm)

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

    # Optional quick warmup with the pipeline (not just the API)
    if args.warmup > 0:
        log.info(f"Warmup: running {args.warmup} sample(s) through the program…")
        p = EuroradProgram()
        for i in range(min(args.warmup, len(testset))):
            _ = p(text=testset[i].text)

    # Program + metric
    program = EuroradProgram()
    metric_fn = label_only_metric_with_feedback_factory(args.label_field, args.solution_field)

    # Evaluate
    kwargs = dict(num_threads=args.num_threads, display_progress=True, display_table=5, max_errors=100)
    evaluate = dspy.Evaluate(metric=metric_fn, devset=testset, **kwargs)
    log.info("Evaluating…")
    _ = evaluate(program)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(130))
    main()

# gepa_eurorad_safe.py
# GEPA on alif-munim/eurorad-omar-120b with robust metric, safe LM, and clean logging.

import os
import re
import io
import random
import contextlib

# --- API key from environment (no prompts) ---
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY not found. Export it first, e.g.:\n  export OPENAI_API_KEY=sk-..."
    )

# --- Imports ---
import dspy
from datasets import load_dataset
from gepa.adapters.dspy_full_program_adapter.full_program_adapter import DspyAdapter
from gepa import optimize


# ========= Dataset loading =========
def load_eurorad_examples():
    ds = load_dataset("alif-munim/eurorad-omar-120b")

    def to_examples(split):
        exs = []
        for row in ds[split]:
            ex = (
                dspy.Example(
                    input=row["input"],
                    final_answer=row["final_answer"],
                    reasoning=row.get("reasoning", ""),
                )
                .with_inputs("input")
            )
            exs.append(ex)
        return exs

    train = to_examples("train")
    val = to_examples("validation")
    test = to_examples("test")
    return train, val, test


train_set, val_set, test_set = load_eurorad_examples()
print(f"Loaded splits: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

# Optional deterministic shuffle for train/val (not test)
random.Random(0).shuffle(train_set)
random.Random(0).shuffle(val_set)


# ========= Utility: candidate extraction & normalization =========
def extract_options(text: str):
    """
    Robustly pull the candidate list after the 'Candidate diagnoses (choose ONE):' line
    until a blank line or EOF.
    """
    if not isinstance(text, str):
        return []
    m = re.search(
        r"Candidate diagnoses\s*\(choose\s*ONE\)\s*:\s*(.+?)(?:\n\s*\n|$)",
        text,
        flags=re.I | re.S,
    )
    if not m:
        return []
    block = m.group(1)
    opts = [o.strip() for o in block.splitlines() if o.strip()]
    return opts


def _norm(s):
    return " ".join(str(s).strip().split())


# ========= Actionable metric with concise feedback =========
def metric_fn(example, pred, trace=None):
    inp = example["input"]
    options = extract_options(inp)
    gold = _norm(example["final_answer"])
    got = _norm(getattr(pred, "final_answer", ""))

    correct = int(gold == got)

    if correct:
        fb = f"Correct. Final answer: '{gold}'. Keep choosing exactly one option verbatim."
    else:
        fb = (
            "Incorrect.\n"
            f"- Your answer: '{got}'\n"
            f"- Correct answer: '{gold}'\n"
            "- RULES:\n"
            "  1) Extract the candidate list from the prompt.\n"
            "  2) Choose exactly ONE option from that list and copy it verbatim.\n"
        )
        if options:
            fb += "  3) Valid options are:\n    - " + "\n    - ".join(options)

    return dspy.Prediction(score=correct, feedback=fb)


# ========= Safe LM wrapper (drops unsupported args like `instructions`) =========
class SafeLM(dspy.LM):
    def forward(self, *, prompt=None, messages=None, **kwargs):
        # Purge keys OpenAI chat.completions doesn't understand if they sneak in
        kwargs.pop("instructions", None)
        extra = kwargs.get("extra_body")
        if isinstance(extra, dict):
            extra.pop("instructions", None)
        return super().forward(prompt=prompt, messages=messages, **kwargs)


# ========= LMs (default temps, max_tokens=32000) =========
task_lm = SafeLM(model="openai/gpt-4.1-mini", max_tokens=32000)
reflection_lm = SafeLM(model="openai/gpt-4.1", max_tokens=32000)
dspy.configure(lm=task_lm)


# ========= Seed DSPy program (2 stages: extract options -> decide) =========
program_src = r"""import re, dspy

class EuroradSig(dspy.Signature):
    input = dspy.InputField()
    final_answer = dspy.OutputField(desc="Return exactly one option copied verbatim from the Candidate diagnoses list in the input.")

def extract_options(text: str):
    m = re.search(r"Candidate diagnoses\s*\(choose\s*ONE\)\s*:\s*(.+?)(?:\n\s*\n|$)", text, re.I|re.S)
    if not m: 
        return []
    return [ln.strip() for ln in m.group(1).splitlines() if ln.strip()]

def snap_to_option(pred: str, options):
    norm = lambda s: re.sub(r"\s+", " ", s.strip()).lower().rstrip(".")
    for o in options:
        if norm(pred) == norm(o): 
            return o
    for o in options:                       # soft containment fallback
        if norm(pred) in norm(o) or norm(o) in norm(pred):
            return o
    return options[0] if options else pred  # last resort

class EuroradModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # Simple CoT like in MATH, but with the correct signature
        self.cot = dspy.ChainOfThought(EuroradSig)

    def forward(self, input):
        opts = extract_options(input)
        pred = self.cot(input=input)
        ans = snap_to_option(getattr(pred, "final_answer", ""), opts)
        return dspy.Prediction(final_answer=ans)

program = EuroradModule()
"""


# ========= Adapter =========
adapter = DspyAdapter(
    task_lm=task_lm,
    metric_fn=metric_fn,
    num_threads=32,
    reflection_lm=lambda prompt: reflection_lm(prompt)[0],
)


# ========= Quiet evaluation helpers =========
def evaluate_quiet(adapter, devset, candidate_dict):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        result = adapter.evaluate(devset, candidate_dict)
    return result


def summarize_eval(eval_result):
    scores = getattr(eval_result, "scores", None)
    if scores is not None:
        correct = sum(int(s) for s in scores)
        total = len(scores)
        acc = 100.0 * correct / max(total, 1)
        return f"{correct}/{total} ({acc:.2f}%)"
    return str(eval_result)


# ========= 1) Baseline test evaluation (single line) =========
print("\n=== Evaluating base (seed) program on test ===")
base_eval = evaluate_quiet(adapter, test_set, {"program": program_src})
print("Base test:", summarize_eval(base_eval))


# ========= 2) GEPA optimization (less chatty) =========
print("\n=== Running GEPA optimization ===")
opt = optimize(
    seed_candidate={"program": program_src},
    trainset=train_set,
    valset=val_set,
    adapter=adapter,
    reflection_lm=lambda p: reflection_lm(p)[0],
    max_metric_calls=1200,
    display_progress_bar=True,   # set False if you want it totally quiet
)


# (Optional) Inspect the evolved program text
# print("\n=== Best candidate (program source) ===")
# print(opt.best_candidate["program"])


# ========= 3) Post-GEPA test evaluation (single line) =========
print("\n=== Evaluating optimized program on test ===")
best_eval = evaluate_quiet(adapter, test_set, opt.best_candidate)
print("Optimized test:", summarize_eval(best_eval))

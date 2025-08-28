"""
Batch evaluation of treatment tasks using GPT-5.
------------------------------------------------
This script reads a CSV file containing treatment task descriptions,
submits them concurrently to GPT-5 for Likert scoring (1–5),
and writes the scores back to a new CSV file.

- No argparse
- No JSONL
- No classes
"""

import os
import asyncio
import random
import time
import pandas as pd
from openai import AsyncOpenAI
from openai import APITimeoutError, APIConnectionError, RateLimitError, APIStatusError

# ------------- Configuration -------------
# Input/output files
input_file = "treatment_task.csv"
output_file = "treatment_task_with_gpt5_LLM_eval_score.csv"

# Model + concurrency
model_name = "gpt-5"
concurrency = 8         # increase/decrease depending on your rate limits
timeout_sec = 120       # per-request timeout
max_retries = 6         # transient error retries

# Initialize the OpenAI client (expects API key in environment variable OPENAI_API_KEY)
# If you insist on pasting a key directly, replace os.getenv(...) with your string.
client = AsyncOpenAI(api_key="")


# ------------- Evaluation prompt -------------
EVAL_PROMPT = """You are asked to evaluate the quality of a model’s treatment suggestion output using the following rubric:

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

# ------------- Small helpers (no classes) -------------
async def _retry(coro_factory, max_tries=max_retries, base_delay=1.0, jitter=0.25):
    """Retry wrapper for transient API errors with exponential backoff."""
    attempt = 0
    while True:
        try:
            return await coro_factory()
        except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as e:
            attempt += 1
            if attempt >= max_tries:
                # Give up; bubble the error
                raise
            sleep_for = base_delay * (2 ** (attempt - 1)) + random.uniform(0, jitter)
            await asyncio.sleep(sleep_for)

def _coerce_score(text: str) -> str:
    """Coerce model output to a clean '1'..'5' string; return '' if invalid."""
    if text is None:
        return ""
    t = "".join(ch for ch in str(text).strip() if ch.isdigit())  # keep digits only
    if not t:
        return ""
    # Take the first digit only (model should output a single integer)
    d = t[0]
    return d if d in {"1", "2", "3", "4", "5"} else ""

async def _score_one(semaphore, row_index, disease_description):
    """Send one scoring request and return (row_index, score_str)."""
    async with semaphore:
        async def do_request():
            return await client.responses.create(
                model=model_name,
                input=EVAL_PROMPT + str(disease_description),
                reasoning={"effort": "medium"},
                text={"verbosity": "low"},
                timeout=timeout_sec,
            )
        resp = await _retry(do_request)
        score_text = (getattr(resp, "output_text", None) or "").strip()
        return row_index, _coerce_score(score_text)

async def _run_all(df: pd.DataFrame):
    """Launch concurrent requests and return a list of (index, score)."""
    sem = asyncio.Semaphore(concurrency)
    tasks = []
    for idx, row in df.iterrows():
        tasks.append(_score_one(sem, idx, row["Disease_description"]))
    started = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print(f"Processed {len(tasks)} rows in {time.time() - started:.1f}s with concurrency={concurrency}")
    return results

# ------------- Main (naive top-level script) -------------
# Load data
df = pd.read_csv(input_file)
if "LLM_eval_score" not in df.columns:
    df["LLM_eval_score"] = ""

# Fire all requests concurrently
results = asyncio.run(_run_all(df))

# Merge results, count errors
errors = 0
for res in results:
    if isinstance(res, Exception):
        errors += 1
        continue
    row_idx, score = res
    df.at[row_idx, "LLM_eval_score"] = score

if errors:
    print(f"Warning: {errors} rows failed permanently (left blank).")

# Optional: quick visibility for the processed rows
if "EvalScore" in df.columns:
    for i, row in df.iterrows():
        ts = row.get("EvalScore")
        ms = row.get("LLM_eval_score")
        print(f"[row {i}] true score: {ts}, LLM score: {ms}")

# Save
df.to_csv(output_file, index=False)
print(f"Wrote: {output_file}")


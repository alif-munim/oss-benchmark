import os

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import json, math, random
from collections import defaultdict, Counter


# Attention backend
os.environ["HF_PYTORCH_ATTENTION_BACKEND"] = "eager"
from torch.backends.cuda import sdp_kernel
sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)


LOCAL_DIR = os.path.expandvars(os.getenv("OSS20B_DIR", "$SCRATCH/models/gpt-oss-20b"))
assert os.path.isabs(LOCAL_DIR), f"LOCAL_DIR is not an absolute path: {LOCAL_DIR}"
assert os.path.isdir(LOCAL_DIR), f"Model dir not found: {LOCAL_DIR}"

# Offline caches
os.environ.setdefault("HF_HOME", os.path.expandvars("$SCRATCH/hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", os.environ["HF_HOME"])
os.environ.setdefault("TORCH_HOME", os.path.expandvars("$SCRATCH/torch_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

## Build a per-GPU memory budget so HF shards across all visible GPUs (3× A100-40GB)
def _build_max_memory(reserve_gib=2):
    assert torch.cuda.is_available(), "CUDA not available."
    n = torch.cuda.device_count()
    assert n >= 3, f"Need ≥3 GPUs visible; found {n}"
    mm = {}
    for i in range(n):
        total_mb = torch.cuda.get_device_properties(i).total_memory // (1024**2)
        target_gib = max(1, (total_mb // 1024) - reserve_gib)
        mm[i] = f"{target_gib}GiB"
    return mm

max_memory = _build_max_memory(reserve_gib=2)

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_DIR,
    trust_remote_code=True,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    LOCAL_DIR,
    trust_remote_code=True,
    local_files_only=True,
    device_map="auto",       # shard across visible GPUs
    max_memory=max_memory,   # respect per-GPU budget
    low_cpu_mem_usage=True,
)
model.eval()

# Tokenizer tweaks
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Ready. Base model only (no LoRA).")
print("Device map:", getattr(model, "hf_device_map", None))
print("Visible GPUs:", torch.cuda.device_count())
print("Max per-GPU memory:", max_memory)
print("Model dir:", LOCAL_DIR)


##### NMED DIAGNOSIS BEAMS #####

VAL_CSV = os.path.expandvars("$PROJECT/data/nmed/nmed_diagnosis.csv")
DATA_DIR = os.path.expandvars("$PROJECT/data")


OUT_CSV = f"{DATA_DIR}/results/nmed_diagnosis_5beams_try8_low_effort_reasoning_v4.csv"
NUM_BEAMS = 5
DIV_PENALTY = 0.5
LENGTH_PENALTY = 1.0
NO_REPEAT_NGRAM_SIZE = 0
CHECKPOINT_EVERY = 50
REASONING_EFFORT = "low"

def read_csv_robust(path):
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin1", on_bad_lines="skip")

df = read_csv_robust(VAL_CSV)

print(df.shape)
df = df.iloc[0:].copy()

# -------------------------
# Prompt
# -------------------------
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
"""

def make_msgs_from_row(row):
    post = str(row.get("Disease_description", "")).strip()
    user = f"{EVAL_PROMPT}\n\nInputs: {post}\n"
    return [
        {"role": "system", "content": "You are an expert clinical diagnostic assistant."},
        {"role": "user",   "content": user},
    ]

# -------------------------
# Parsers
# -------------------------
# --- Extract letters immediately after 'assistantfinal' (case-insensitive) ---
ASSISTANTFINAL_TAG = re.compile(r"assistant\s*final", re.I)  # matches 'assistantfinal', 'assistant final', etc.
SCORE_AFTER_TAG = re.compile(r"([0-5](?:\.[05])?)")

def parse_score_after_assistantfinal(text: str):
    if not text:
        return None
    m = ASSISTANTFINAL_TAG.search(text)
    tail = text[m.end():] if m else text
    tail = re.sub(r"^[\s:>\]\)\-–—_]*", "", tail)
    m2 = SCORE_AFTER_TAG.search(tail)
    if not m2:
        return None
    try:
        val = float(m2.group(1))
    except Exception:
        return None
    # clamp just in case
    return max(0.0, min(5.0, val))

# -------------------------
# Generation
# -------------------------
def gen_cot_beams(msgs, effort, max_new_tokens=8,
                  beams=3, diversity_penalty=0.5, length_penalty=1.0, no_repeat_ngram_size=0):
    # Encode once
    enc = tokenizer.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort=REASONING_EFFORT,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.inference_mode():
        out = model.generate(
            **enc,
            do_sample=False,
            num_beams=beams,
            num_beam_groups=beams,
            num_return_sequences=beams,
            diversity_penalty=diversity_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,   # potentially long outputs
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=False,
            output_scores=False,
        )

    prompt_len = enc["input_ids"].shape[1]
    gens, cont_ids_list = [], []
    for i in range(out.size(0)):
        cont_ids = out[i, prompt_len:]
        cont_ids_list.append(cont_ids.unsqueeze(0))
        gens.append(tokenizer.decode(cont_ids, skip_special_tokens=True).strip())
    return gens, enc, cont_ids_list

# Seeds & eval
model.eval()
#torch.manual_seed(75); random.seed(75); np.random.seed(75)
try:
    model.generation_config.trust_remote_code = True
except Exception:
    pass

# Run inference
final_answers = []
beam_labels_col, all_beams_text, beam_parsed_scores_col = [], [], []
top_reasonings = []  # kept for schema compatibility; will be empty strings


os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

def _write_csv_atomic(frame: pd.DataFrame, path: str):
    tmp = path + ".tmp"
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)  # atomic on POSIX

def _save_checkpoint(k: int):
    """Save rows [0..k-1] with current outputs (atomic overwrite)."""
    part = df.iloc[:k].copy()
    part["reasoning"]            = top_reasonings[:k]
    part["final_answer"]         = final_answers[:k]
    part["beam_parsed_scores"]   = beam_parsed_scores_col[:k]
    part["raw_generations"]      = all_beams_text[:k]
    _write_csv_atomic(part, OUT_CSV)
    print(f"[CKPT] Saved {k} rows -> {OUT_CSV}")

# ---- Main loop ----
for _, row in tqdm(df.iterrows(), total=len(df)):
    msgs = make_msgs_from_row(row)

    gens, _, _ = gen_cot_beams(
        msgs,
        max_new_tokens=3000,
        beams=NUM_BEAMS,
        diversity_penalty=DIV_PENALTY,
        length_penalty=LENGTH_PENALTY,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        effort=REASONING_EFFORT,
    )

    # Parse labels (scores) from beams
    per_beam_parsed_scores = [parse_score_after_assistantfinal(g) for g in gens]

    # -----------------------------
    # PICK THE MOST FREQUENT SCORE
    # Tie-breaker: choose the earliest beam
    # -----------------------------
    counts = Counter([sc for sc in per_beam_parsed_scores if sc is not None])

    if counts:
        max_count = max(counts.values())
        tied_scores = [sc for sc, c in counts.items() if c == max_count]

        if len(tied_scores) == 1:
            chosen_score = tied_scores[0]
        else:
            first_idx = {
                sc: next(i for i, s in enumerate(per_beam_parsed_scores) if s == sc)
                for sc in tied_scores
            }
            chosen_score = min(tied_scores, key=lambda sc: first_idx[sc])
    else:
        # Fallback: first non-None parsed score, otherwise None
        chosen_score = next((sc for sc in per_beam_parsed_scores if sc is not None), None)

    top_reasonings.append("")
    final_answers.append("" if chosen_score is None else chosen_score)
    beam_parsed_scores_col.append(json.dumps(per_beam_parsed_scores, ensure_ascii=False))
    all_beams_text.append(" ||| ".join(gens))

    # Checkpoint every N rows
    k = len(final_answers)
    if k % CHECKPOINT_EVERY == 0:
        _save_checkpoint(k)

# Save

_save_checkpoint(len(df))
print("Saved:", OUT_CSV)

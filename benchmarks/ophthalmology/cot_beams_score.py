#!/usr/bin/env python3
"""
Ophthalmology MCQ — CoT diverse-beam inference + external rescoring (HF Transformers)

- Chat-model safe (uses tokenizer.apply_chat_template)
- Returns ALL beams (num_return_sequences = NUM_BEAMS)
- CSV is stable: every row has exactly NUM_BEAMS entries for beam_* and raw_generations
- Parses ONLY option letters that are actually present in the question (A., B., …)
- Picks consensus label if >= AGREE_THRESHOLD; else highest rescored beam
- Supports resume: if --resume and --out_csv exists, re-run only rows with empty final_answer

Env knobs:
  MODEL_NAME, LORA_PATH, DATA_DIR
  NUM_BEAMS, NUM_BEAM_GROUPS, DIVERSITY_PENALTY, LENGTH_PENALTY, MAX_NEW_TOKENS, REPETITION_PENALTY, AGREE_THRESHOLD
  ATTN_IMPL, DTYPE, TRUST_REMOTE_CODE, DEVICE_MAP, LOAD_4BIT
"""

import os, re, json, argparse
from collections import Counter
from typing import List, Tuple, Union, Optional
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# ---------- Optional PEFT ----------
try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

# ---------- Config ----------
DEFAULT_MODEL = os.environ.get("MODEL_NAME", "openai/gpt-oss-120b")
LORA_PATH     = os.environ.get("LORA_PATH", "")
DATA_DIR      = os.environ.get("DATA_DIR", ".")
DTYPE         = os.environ.get("DTYPE", "bfloat16")
ATTN_IMPL     = os.environ.get("ATTN_IMPL", "eager")
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "1") == "1"
DEVICE_MAP    = os.environ.get("DEVICE_MAP", "auto")
LOAD_4BIT     = os.environ.get("LOAD_4BIT", "0") == "1"  # needs bitsandbytes

# Beam / selection
NUM_BEAMS         = int(os.environ.get("NUM_BEAMS", 5))
NUM_BEAM_GROUPS   = int(os.environ.get("NUM_BEAM_GROUPS", NUM_BEAMS))
DIVERSITY_PENALTY = float(os.environ.get("DIVERSITY_PENALTY", 0.5))
LENGTH_PENALTY    = float(os.environ.get("LENGTH_PENALTY", 1.0))
MAX_NEW_TOKENS    = int(os.environ.get("MAX_NEW_TOKENS", 64))
REPETITION_PENALTY= float(os.environ.get("REPETITION_PENALTY", 1.0))  # 1.0=off
AGREE_THRESHOLD   = float(os.environ.get("AGREE_THRESHOLD", 0.5))

# ---------- Prompts (ophthalmology MCQ) ----------
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

# ---------- Regex / cleaners ----------
ROLE_BLEED_RE = re.compile(r"^\s*(assistant(final|analysis)?|system|user)\s*[:>]*\s*", re.I)
ASSISTANTFINAL_RE = re.compile(r"assistantfinal\s*([A-Z]{1,26})", re.I)
LETTERS_RUN_RE = re.compile(r"\b([A-Z]{1,26})\b")
OPT_LINE_RE = re.compile(r"^\s*([A-Z])\s*[).、．。:]", re.UNICODE)

def clean_continuation(txt: str) -> str:
    """Trim role/channel bleed and keep text as-is (no XML enforcing here)."""
    if not txt:
        return ""
    s = ROLE_BLEED_RE.sub("", txt.strip())
    return s

def extract_allowed_letters(question_text: str) -> set:
    """Collect letters that actually appear as options (A., B., ...)."""
    allowed = []
    for ln in str(question_text or "").splitlines():
        m = OPT_LINE_RE.match(ln)
        if m:
            allowed.append(m.group(1))
    return set(allowed or list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

def parse_choice_safe(raw: str, allowed: set) -> Optional[str]:
    """Extract an answer composed only of allowed letters.
       Priority: explicit 'assistantfinalXXX' → else last valid capital run."""
    if not raw:
        return None
    s = raw.strip()

    m = ASSISTANTFINAL_RE.search(s)
    if m:
        cand = m.group(1).strip().upper()
        return cand if set(cand) <= allowed else None

    cands = []
    for m in LETTERS_RUN_RE.finditer(s.upper()):
        seq = m.group(1)
        if set(seq) <= allowed:
            cands.append(seq)
    return cands[-1] if cands else None

def get_eos_ids(model, tok) -> Union[int, List[int]]:
    eid = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(eid, list) and eid:
        return eid
    lst = [eid] if isinstance(eid, int) else []
    for t in ["<|im_end|>", "<|end|>", "<|return|>"]:
        try:
            _id = tok.convert_tokens_to_ids(t)
            if _id is not None and _id != tok.eos_token_id and _id not in lst:
                lst.append(_id)
        except Exception:
            pass
    return lst if lst else tok.eos_token_id

# ---------- Chat templating ----------
def build_user_prompt(row: pd.Series, user_template: str) -> str:
    # Prefer the "Question" column; else concatenate text-like columns.
    if "Question" in row and isinstance(row["Question"], str):
        case_text = row["Question"].strip()
    else:
        case_text = "\n".join([str(v) for v in row.values if isinstance(v, str)]).strip()
    return user_template.format(case_text=case_text)

def build_chat_prompt(tok, system_text: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user",   "content": user_text},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ---------- HF load ----------
def load_model_and_tokenizer(model_name: str):
    qkw = {}
    if LOAD_4BIT:
        from transformers import BitsAndBytesConfig
        qkw = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True),
               "device_map": DEVICE_MAP}
    dtype = torch.bfloat16 if DTYPE.lower() == "bfloat16" else torch.float16
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=TRUST_REMOTE_CODE, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation=ATTN_IMPL,
        torch_dtype=dtype,
        device_map=DEVICE_MAP,
        trust_remote_code=TRUST_REMOTE_CODE,
        **qkw,
    )
    if LORA_PATH:
        if not _HAS_PEFT:
            raise RuntimeError("PEFT not installed but LORA_PATH provided")
        model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    return model, tok

# ---------- Generation & rescoring ----------
def gen_beams(model, tok, chat_prompt: str, eos_ids: Union[int, List[int]]
) -> Tuple[List[str], torch.LongTensor, List[torch.LongTensor]]:
    inputs = tok(chat_prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(model.device)
    attn_mask = inputs["attention_mask"].to(model.device)

    cfg = GenerationConfig(
        do_sample=False,
        num_beams=NUM_BEAMS,
        num_beam_groups=NUM_BEAM_GROUPS,
        diversity_penalty=DIVERSITY_PENALTY,
        length_penalty=LENGTH_PENALTY,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tok.pad_token_id,
        eos_token_id=eos_ids,
        return_dict_in_generate=True,
        output_scores=False,
        num_return_sequences=NUM_BEAMS,
        repetition_penalty=(None if REPETITION_PENALTY == 1.0 else REPETITION_PENALTY),
    )
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids, attention_mask=attn_mask,
            **{k: v for k, v in cfg.to_dict().items() if v is not None}
        )

    seq = out.sequences  # [NUM_BEAMS, prompt+gen]
    p_len = input_ids.shape[1]
    cont_ids, decoded = [], []
    for i in range(seq.size(0)):
        cont = seq[i][p_len:]
        cont_ids.append(cont)
        txt = tok.decode(cont, skip_special_tokens=True)
        txt = clean_continuation(txt)
        decoded.append(txt)

    # Strong invariant: always NUM_BEAMS
    if len(decoded) != NUM_BEAMS:
        decoded = (decoded + [""] * NUM_BEAMS)[:NUM_BEAMS]
        cont_ids = (cont_ids + [torch.empty(0, dtype=torch.long, device=model.device)] * NUM_BEAMS)[:NUM_BEAMS]
    return decoded, input_ids[0], cont_ids

def score_continuation_given_ids(model, tok, prompt_ids: torch.LongTensor, cont_ids: torch.LongTensor) -> float:
    if cont_ids.numel() == 0:
        return float("nan")
    dev = model.device
    full = torch.cat([prompt_ids, cont_ids]).unsqueeze(0).to(dev)
    attn = torch.ones_like(full, dtype=torch.long, device=dev)
    with torch.no_grad():
        logits = model(input_ids=full, attention_mask=attn).logits
    logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = full[:, 1:]
    token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    p_len = prompt_ids.numel()
    mask = torch.zeros_like(token_lp, dtype=torch.bool)
    mask[:, p_len - 1:] = True
    sel = token_lp[mask]
    return float(sel.mean().item()) if sel.numel() else float("nan")

# ---------- Resume helpers ----------
def merge_existing_outputs(in_df: pd.DataFrame, out_csv: str) -> pd.DataFrame:
    if not (out_csv and os.path.exists(out_csv)):
        # ensure columns exist to be filled
        for col in ("reasoning", "final_answer", "beam_final_labels", "beam_sequence_scores", "raw_generations"):
            if col not in in_df.columns:
                in_df[col] = ""
        return in_df
    prev = pd.read_csv(out_csv)
    # Extend/trim to current input length
    for col in ("reasoning", "final_answer", "beam_final_labels", "beam_sequence_scores", "raw_generations"):
        if col in prev.columns:
            vals = list(prev[col])[:len(in_df)]
            if len(vals) < len(in_df):
                vals += [""] * (len(in_df) - len(vals))
            in_df[col] = vals
        else:
            if col not in in_df.columns:
                in_df[col] = ""
    return in_df

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Ophthalmology MCQ beam search + rescoring (HF Transformers)")
    ap.add_argument("input_csv")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--system", default=DEFAULT_SYSTEM)
    ap.add_argument("--user_template", default=DEFAULT_USER_TEMPLATE)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--start_row", type=int, default=0)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--save_every_n", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    model, tok = load_model_and_tokenizer(args.model)
    eos_ids = get_eos_ids(model, tok)

    df = pd.read_csv(args.input_csv)

    if args.resume:
        df = merge_existing_outputs(df, args.out_csv)

    # streaming start fresh if not resuming
    written_header = False
    if args.stream and (not args.resume) and os.path.exists(args.out_csv):
        try:
            os.remove(args.out_csv)
        except Exception:
            pass

    if args.start_row > 0:
        df = df.iloc[args.start_row:].reset_index(drop=True)

    finals, reasons, beams_labels, beams_scores, beams_texts = [], [], [], [], []

    total_rows = len(df)
    for idx, row in tqdm(df.iterrows(), total=total_rows, desc="MCQ beams"):
        # If resuming and row already has a non-empty final_answer, skip computation but keep row as-is in streaming write
        if args.resume and isinstance(row.get("final_answer", ""), str) and row["final_answer"].strip():
            # Repack existing outputs
            finals.append(row["final_answer"])
            reasons.append(row.get("reasoning", ""))
            beams_labels.append(row.get("beam_final_labels", ""))
            beams_scores.append(row.get("beam_sequence_scores", ""))
            beams_texts.append(row.get("raw_generations", ""))
            # stream copy
            if args.stream and (((idx + 1) % args.save_every_n == 0) or (idx + 1) == total_rows):
                out_row = row.to_dict()
                pd.DataFrame([out_row]).to_csv(
                    args.out_csv, mode=("a" if written_header else "w"),
                    header=(not written_header), index=False
                )
                written_header = True
            continue

        question_text = str(row.get("Question", ""))
        allowed = extract_allowed_letters(question_text)

        # Build chat prompt
        user_text = build_user_prompt(row, args.user_template)
        chat_prompt = build_chat_prompt(tok, args.system, user_text)

        # Generate beams
        gens, prompt_ids, cont_ids_list = gen_beams(model, tok, chat_prompt, eos_ids)

        # Score & parse per-beam choices
        scores, per_labels, per_texts = [], [], []
        for cont_ids, g in zip(cont_ids_list, gens):
            g_txt = clean_continuation(g)
            per_texts.append(g_txt if g_txt else "")
            scores.append(score_continuation_given_ids(model, tok, prompt_ids, cont_ids))
            choice = parse_choice_safe(g_txt, allowed)
            per_labels.append("" if choice is None else choice)

        # Decide final answer
        cnt = Counter([x for x in per_labels if x])
        if cnt:
            top_lab, top_n = cnt.most_common(1)[0]
            agree = top_n / max(1, len(per_labels))
        else:
            top_lab, agree = "", 0.0

        if agree >= AGREE_THRESHOLD and top_lab:
            chosen_label = top_lab
            chosen_reason = ""  # no reasoning output for MCQ
        else:
            valid_idxs = [i for i, l in enumerate(per_labels) if l]
            if valid_idxs:
                # Highest external score among valid beams
                best_i = max(valid_idxs, key=lambda i: (scores[i] if scores[i] == scores[i] else float("-inf")))
            else:
                best_i = max(range(len(scores)), key=lambda i: (scores[i] if scores[i] == scores[i] else float("-inf"))) if scores else 0
            chosen_label = per_labels[best_i] if per_labels else ""
            chosen_reason = ""

        # Force exact lengths / JSON-serialize
        if len(per_labels) != NUM_BEAMS: per_labels = (per_labels + [""] * NUM_BEAMS)[:NUM_BEAMS]
        if len(scores)     != NUM_BEAMS: scores     = (scores + [float("nan")] * NUM_BEAMS)[:NUM_BEAMS]
        if len(per_texts)  != NUM_BEAMS: per_texts  = (per_texts + [""] * NUM_BEAMS)[:NUM_BEAMS]

        finals.append(chosen_label)
        reasons.append(chosen_reason)
        beams_labels.append(json.dumps(per_labels, ensure_ascii=False))
        beams_scores.append(json.dumps(scores))
        beams_texts.append(" ||| ".join(per_texts))

        # streaming write current row
        if args.stream and (((idx + 1) % args.save_every_n == 0) or (idx + 1) == total_rows):
            out_row = row.to_dict()
            out_row.update({
                "reasoning": chosen_reason,
                "final_answer": chosen_label,
                "beam_final_labels": json.dumps(per_labels, ensure_ascii=False),
                "beam_sequence_scores": json.dumps(scores),
                "raw_generations": " ||| ".join(per_texts),
            })
            pd.DataFrame([out_row]).to_csv(
                args.out_csv, mode=("a" if written_header else "w"),
                header=(not written_header), index=False
            )
            written_header = True

    # final write (non-stream)
    if not args.stream:
        out = df.copy()
        out["reasoning"]             = reasons
        out["final_answer"]          = finals
        out["beam_final_labels"]     = beams_labels
        out["beam_sequence_scores"]  = beams_scores
        out["raw_generations"]       = beams_texts
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        out.to_csv(args.out_csv, index=False)
        print("Saved:", args.out_csv)

if __name__ == "__main__":
    main()

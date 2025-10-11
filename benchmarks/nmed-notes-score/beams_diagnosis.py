#!/usr/bin/env python3
"""
Batch evaluation of diagnostic tasks with GPT-OSS via HF Transformers.
- Uses diverse beam search (deterministic; no sampling)
- Writes a stable CSV with per-row consensus + ALL beams
- Argparse for paths/model/start_row/streaming

Fields expected in the input CSV:
  - Disease_description  (string blob that already includes task, GT disease, and model output)

Outputs (added/overwritten):
  - reasoning
  - final_answer
  - beam_final_labels
  - beam_sequence_scores
  - raw_generations
"""

import os, re, json, argparse
from collections import Counter
from typing import List, Tuple, Union
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

# ---------- Defaults (env-backed) ----------
DEFAULT_MODEL = os.environ.get("MODEL_NAME", "openai/gpt-oss-120b")
DEFAULT_IN   = os.environ.get("IN_CSV",  "diagnosis_task.csv")
DEFAULT_OUT  = os.environ.get("OUT_CSV", "diagnosis_task_with_gpt5_LLM_eval_score.csv")

NUM_BEAMS         = int(os.environ.get("NUM_BEAMS", 5))
NUM_BEAM_GROUPS   = int(os.environ.get("NUM_BEAM_GROUPS", NUM_BEAMS))
DIVERSITY_PENALTY = float(os.environ.get("DIVERSITY_PENALTY", 0.5))
LENGTH_PENALTY    = float(os.environ.get("LENGTH_PENALTY", 1.0))
MAX_NEW_TOKENS    = int(os.environ.get("MAX_NEW_TOKENS", 512))
REPETITION_PENALTY= float(os.environ.get("REPETITION_PENALTY", 1.0))  # 1.0 = off
AGREE_THRESHOLD   = float(os.environ.get("AGREE_THRESHOLD", 0.5))

ATTN_IMPL         = os.environ.get("ATTN_IMPL", "eager")
DTYPE             = os.environ.get("DTYPE", "bfloat16")
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "1") == "1"
DEVICE_MAP        = os.environ.get("DEVICE_MAP", "auto")
LOAD_4BIT         = os.environ.get("LOAD_4BIT", "0") == "1"  # needs bitsandbytes
LORA_PATH         = os.environ.get("LORA_PATH", "")

# ---------- System/User prompts ----------
# Keep your prompt AS-IS; we just read from assistantfinal in the parser.
SYS_PROMPT = (
    "Reasoning: low"
    "You are an impartial evaluator of diagnostic differentials.\n"
    "Score the provided model output using this rubric (Likert 1–5; half-points allowed):\n"
    "1: Most relevant options not mentioned.\n"
    "2: Some or many relevant options not mentioned.\n"
    "3: Most relevant options mentioned.\n"
    "4: Most relevant options mentioned.\n"
    "5: All relevant options mentioned.\n\n"
    "Output ONLY the numeric score (e.g., 4 or 3.5). No words, no units, no punctuation."
)

USER_TEMPLATE = (
    "Inputs:\n{desc}\n\n"
    "Return ONLY a single number in {{1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5}}.\n"
)



# ---------- Regex / cleaners ----------
# Primary: pull from assistantfinal first (handles 'assistantfinal5', 'assistantfinal 4.5', 'assistant final: 3', etc.)
ASSISTANT_FINAL_RE = re.compile(
    r"\bassistant\s*final\b[:\s-]*([1-5](?:\.5)?)\b|\bassistantfinal\b[:\s-]*([1-5](?:\.5)?)\b",
    re.I
)
# Secondary: XML tag fallback
FINAL_RE    = re.compile(r"<final>\s*([1-5](?:\.5)?)\s*</final>", re.I | re.S)
ANALYSIS_RE = re.compile(r"<analysis>\s*(.*?)\s*</analysis>",    re.I | re.S)

# Trim role-prefix bleed like "assistantfinal5" or "analysis..." at the start of a line
ROLE_BLEED_RE = re.compile(r"^\s*(assistant(final|analysis)?|system|user)\s*[:>]*\s*", re.I)

def clean_continuation(txt: str) -> str:
    if not txt:
        return ""
    s = ROLE_BLEED_RE.sub("", txt.strip())

    # Shorten to start at the first relevant marker if present
    low = s.lower()
    anchors = [low.find("<analysis>"), low.find("<final>"), low.find("assistantfinal"), low.find("assistant final")]
    anchors = [a for a in anchors if a != -1]
    first = min(anchors) if anchors else -1
    if first > 0:
        s = s[first:]

    # If there's a closing </final>, truncate after it to avoid trailing debris
    j = s.lower().find("</final>")
    if j != -1:
        s = s[: j + len("</final>")]

    return s.strip()

def _fallback_number(text: str) -> str:
    """Safe fallback: pick a standalone 1..5 (optionally .5) not adjacent to a hyphen (avoid '1-5')."""
    if not text:
        return ""
    norm = text.replace("–", "-").replace("—", "-")
    cands = []
    for m in re.finditer(r"\b([1-5](?:\.5)?)\b", norm):
        i, j = m.span()
        left = norm[max(0, i-1):i]
        right = norm[j:j+1]
        if left == "-" or right == "-":
            continue
        cands.append(m.group(1))
    # Scores often appear at the end; prefer last plausible
    return cands[-1] if cands else ""

def parse_final_only(text: str) -> str:
    """Prefer `assistantfinal` marker, then <final> tag, then safe fallback."""
    if not text:
        return ""

    # 1) assistantfinal first
    m = ASSISTANT_FINAL_RE.search(text)
    if m:
        return (m.group(1) or m.group(2)).strip()

    # 2) <final>NUMBER</final> fallback
    m = FINAL_RE.search(text)
    if m:
        return m.group(1).strip()

    # 3) Safe fallback (avoid rubric ranges)
    return _fallback_number(text)

def parse_analysis(text: str) -> str:
    if not text:
        return ""
    m = ANALYSIS_RE.search(text)
    return (m.group(1).strip() if m else "").strip()

# ---------- Chat template ----------
def build_chat_prompt(tok, description: str) -> str:
    user = USER_TEMPLATE.format(desc=description.strip())
    messages = [{"role": "system", "content": SYS_PROMPT},
                {"role": "user",   "content": user}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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
        model_name, attn_implementation=ATTN_IMPL, torch_dtype=dtype,
        device_map=DEVICE_MAP, trust_remote_code=TRUST_REMOTE_CODE, **qkw
    )
    if LORA_PATH:
        if not _HAS_PEFT:
            raise RuntimeError("PEFT not installed but LORA_PATH provided")
        model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    return model, tok

# ---------- Generation ----------
def gen_beams(model, tok, chat_prompt: str, eos_ids: Union[int, List[int]]):
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
        out = model.generate(input_ids=input_ids, attention_mask=attn_mask,
                             **{k: v for k, v in cfg.to_dict().items() if v is not None})
    seq = out.sequences
    p_len = input_ids.shape[1]
    cont_ids, decoded = [], []
    for i in range(seq.size(0)):
        cont = seq[i][p_len:]
        cont_ids.append(cont)
        txt = tok.decode(cont, skip_special_tokens=True)
        decoded.append(clean_continuation(txt))
    # enforce invariant
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
    mask[:, p_len-1:] = True
    sel = token_lp[mask]
    return float(sel.mean().item()) if sel.numel() else float("nan")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Diagnostic Likert evaluation with diverse beam search (HF Transformers).")
    ap.add_argument("--in_csv",  default=DEFAULT_IN,  help="Input CSV path (expects 'Disease_description' column).")
    ap.add_argument("--out_csv", default=DEFAULT_OUT, help="Output CSV path.")
    ap.add_argument("--model",   default=DEFAULT_MODEL, help="HF model id, e.g. openai/gpt-oss-120b")
    ap.add_argument("--start_row", type=int, default=0, help="Skip rows before this index.")
    ap.add_argument("--stream", action="store_true", help="Write rows incrementally as they complete.")
    ap.add_argument("--save_every_n", type=int, default=1, help="Streaming: write every N rows.")
    args = ap.parse_args()

    model, tok = load_model_and_tokenizer(args.model)
    eos_ids = get_eos_ids(model, tok)

    df = pd.read_csv(args.in_csv)
    if args.start_row > 0:
        df = df.iloc[args.start_row:].reset_index(drop=True)

    # Columns we will (over)write
    finals, reasons, beams_labels, beams_scores, beams_texts = [], [], [], [], []

    # Run
    need_col = "Disease_description"
    if need_col not in df.columns:
        raise ValueError(f"Missing required column '{need_col}' in {args.in_csv}")

    written_header = False
    if args.stream and os.path.exists(args.out_csv):
        try:
            os.remove(args.out_csv)
        except Exception:
            pass

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Diagnosis beams"):
        desc = "" if pd.isna(row[need_col]) else str(row[need_col]).strip()
        if not desc:
            finals.append("")
            reasons.append("")
            beams_labels.append(json.dumps([""] * NUM_BEAMS))
            beams_scores.append(json.dumps([float("nan")] * NUM_BEAMS))
            beams_texts.append(" ||| ".join([""] * NUM_BEAMS))
            continue

        prompt = build_chat_prompt(tok, desc)
        gens, prompt_ids, cont_ids_list = gen_beams(model, tok, prompt, eos_ids)

        # Per-beam parse/score
        scores, per_labels, per_xml = [], [], []
        for cont_ids, g in zip(cont_ids_list, gens):
            g_xml = clean_continuation(g)
            per_xml.append(g_xml if g_xml else "")
            scores.append(score_continuation_given_ids(model, tok, prompt_ids, cont_ids))
            raw_lab = parse_final_only(g_xml)
            per_labels.append(raw_lab)

        # Majority (if enough agreement) else highest-score valid
        cnt = Counter([x for x in per_labels if x])
        if cnt:
            top_lab, top_n = cnt.most_common(1)[0]
            agree = top_n / max(1, len(per_labels))
        else:
            top_lab, agree = "", 0.0

        if agree >= AGREE_THRESHOLD and top_lab:
            chosen_label = top_lab
            chosen_reason = ""
            for g_xml, lab in zip(per_xml, per_labels):
                if lab == top_lab:
                    chosen_reason = parse_analysis(g_xml)
                    break
        else:
            valid_idxs = [i for i, l in enumerate(per_labels) if l]
            if valid_idxs:
                best_i = max(valid_idxs, key=lambda i: (scores[i] if scores[i] == scores[i] else float("-inf")))
            else:
                best_i = max(range(len(scores)), key=lambda i: (scores[i] if scores[i] == scores[i] else float("-inf"))) if scores else 0
            chosen_label = per_labels[best_i] if per_labels else ""
            chosen_reason = parse_analysis(per_xml[best_i]) if per_xml else ""

        # Append (pad to NUM_BEAMS)
        if len(per_labels) != NUM_BEAMS: per_labels = (per_labels + [""] * NUM_BEAMS)[:NUM_BEAMS]
        if len(scores)     != NUM_BEAMS: scores     = (scores + [float("nan")] * NUM_BEAMS)[:NUM_BEAMS]
        if len(per_xml)    != NUM_BEAMS: per_xml    = (per_xml + [""] * NUM_BEAMS)[:NUM_BEAMS]

        finals.append(chosen_label)
        reasons.append(chosen_reason)
        beams_labels.append(json.dumps(per_labels, ensure_ascii=False))
        beams_scores.append(json.dumps(scores))
        beams_texts.append(" ||| ".join(per_xml))

        # Streaming write
        if args.stream and (((idx + 1) % args.save_every_n == 0) or (idx + 1) == len(df)):
            out_row = row.to_dict()
            out_row.update({
                "reasoning": chosen_reason,
                "final_answer": chosen_label,
                "beam_final_labels": json.dumps(per_labels, ensure_ascii=False),
                "beam_sequence_scores": json.dumps(scores),
                "raw_generations": " ||| ".join(per_xml),
            })
            pd.DataFrame([out_row]).to_csv(
                args.out_csv, mode=("a" if written_header else "w"),
                header=(not written_header), index=False
            )
            written_header = True

    # Final write (non-streaming)
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

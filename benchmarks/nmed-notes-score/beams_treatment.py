#!/usr/bin/env python3
"""
Treatment Likert (1–5, half-points allowed) evaluator with diverse beam search (HF Transformers)
- Reuses the same beam infra (num_beams=num_return_sequences, beam groups, diversity penalty)
- Chat-model safe (uses tokenizer.apply_chat_template)
- CSV is stable: every row has exactly NUM_BEAMS entries for beam_* and raw_generations
- Normalizes outputs to allowed scores: 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5
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

# ---------- Config (env-overridable) ----------
DEFAULT_MODEL = os.environ.get("MODEL_NAME", "openai/gpt-oss-120b")
LORA_PATH = os.environ.get("LORA_PATH", "")
DATA_DIR  = os.environ.get("DATA_DIR", ".")
IN_CSV    = os.environ.get("IN_CSV",  os.path.join(DATA_DIR, "treatment_task.csv"))
OUT_CSV   = os.environ.get("OUT_CSV", os.path.join(DATA_DIR, "treatment_beam_scores.csv"))

NUM_BEAMS         = int(os.environ.get("NUM_BEAMS", 5))
NUM_BEAM_GROUPS   = int(os.environ.get("NUM_BEAM_GROUPS", NUM_BEAMS))
DIVERSITY_PENALTY = float(os.environ.get("DIVERSITY_PENALTY", 0.5))
LENGTH_PENALTY    = float(os.environ.get("LENGTH_PENALTY", 1.0))
MAX_NEW_TOKENS    = int(os.environ.get("MAX_NEW_TOKENS", 128))
REPETITION_PENALTY= float(os.environ.get("REPETITION_PENALTY", 1.0))  # 1.0=off
AGREE_THRESHOLD   = float(os.environ.get("AGREE_THRESHOLD", 0.5))

ATTN_IMPL         = os.environ.get("ATTN_IMPL", "eager")
DTYPE             = os.environ.get("DTYPE", "bfloat16")
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "1") == "1"
DEVICE_MAP        = os.environ.get("DEVICE_MAP", "auto")
LOAD_4BIT         = os.environ.get("LOAD_4BIT", "0") == "1"  # needs bitsandbytes

# ---------- Prompts (kept as-is) ----------
SYS_PROMPT = (
    "Reasoning: low"
    "You are a strict evaluator of treatment suggestions.\n"
    "Score the provided model output using this rubric (Likert 1–5; half-points allowed):\n"
    "1: All/most options redundant or unjustified.\n"
    "2: Several options redundant or unjustified.\n"
    "3: Mixed quality; some redundancy/unjustified.\n"
    "4: Few minor issues.\n"
    "5: No options redundant or unjustified.\n\n"
    "Output ONLY the numeric score (e.g., 4 or 3.5). No words, no units, no punctuation."
)

USER_TEMPLATE = (
    "Task description:\n{task}\n\n"
    "{true_disease_block}"
    "{model_output_block}"
    "Return ONLY a single number in {{1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5}}.\n"
)


# ---------- Regex / cleaners ----------
ROLE_BLEED_RE = re.compile(r"^\s*(assistant(final|analysis)?|system|user)\s*[:>]*\s*", re.I)

# Prefer pulling from assistantfinal / assistant final first
ASSISTANT_FINAL_RE = re.compile(
    r"\bassistant\s*final\b[:\s-]*([1-5](?:\.5)?)\b|\bassistantfinal\b[:\s-]*([1-5](?:\.5)?)\b",
    re.I
)
# Accept <final>NUMBER</final> as a secondary format
XML_FINAL_RE = re.compile(r"<final>\s*([1-5](?:\.5)?)\s*</final>", re.I | re.S)

# Safe fallback: a standalone 1..5 or x.5 not adjacent to a hyphen (avoids '1–5' from the rubric)
def _safe_number_fallback(text: str) -> str:
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
    # Scores often appear last; prefer the final plausible number
    return cands[-1] if cands else ""

ALLOWED_SCORES = {"1","1.5","2","2.5","3","3.5","4","4.5","5"}

def clean_continuation(txt: str) -> str:
    if not txt:
        return ""
    s = ROLE_BLEED_RE.sub("", txt.strip())
    low = s.lower()
    anchors = [low.find("assistantfinal"), low.find("assistant final"), low.find("<final>")]
    anchors = [a for a in anchors if a != -1]
    if anchors:
        s = s[min(anchors):]
    j = s.lower().find("</final>")
    if j != -1:
        s = s[: j + len("</final>")]
    return s.strip()

def parse_score(text: str) -> str:
    """Extract a valid score with priority: assistantfinal → <final> → safe standalone fallback."""
    if not text:
        return ""
    m = ASSISTANT_FINAL_RE.search(text)
    if m:
        val = (m.group(1) or m.group(2)).strip()
        return val if val in ALLOWED_SCORES else ""
    m = XML_FINAL_RE.search(text)
    if m:
        val = m.group(1).strip()
        return val if val in ALLOWED_SCORES else ""
    val = _safe_number_fallback(text)
    return val if val in ALLOWED_SCORES else ""

def _mk_block(label: str, body: str) -> str:
    return f"{label}:\n{body.strip()}\n\n" if body and body.strip() else ""

# ---------- Chat templating ----------
def build_chat_prompt(tok, task_text: str, true_disease: str, model_output: str) -> str:
    user = USER_TEMPLATE.format(
        task=task_text.strip(),
        true_disease_block=_mk_block("True disease", true_disease),
        model_output_block=_mk_block("Model output", model_output),
    )
    messages = [{"role":"system","content":SYS_PROMPT},
                {"role":"user","content":user}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def get_eos_ids(model, tok) -> Union[int, List[int]]:
    eid = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(eid, list) and eid: return eid
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
def load_model_and_tokenizer(model_name: str = DEFAULT_MODEL):
    qkw = {}
    if LOAD_4BIT:
        from transformers import BitsAndBytesConfig
        qkw = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True),
               "device_map": DEVICE_MAP}
    dtype = torch.bfloat16 if DTYPE.lower()=="bfloat16" else torch.float16
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation=ATTN_IMPL, torch_dtype=dtype,
        device_map=DEVICE_MAP, trust_remote_code=True, **qkw
    )
    if LORA_PATH:
        if not _HAS_PEFT: raise RuntimeError("PEFT not installed but LORA_PATH provided")
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
        repetition_penalty=(None if REPETITION_PENALTY==1.0 else REPETITION_PENALTY),
    )
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, attention_mask=attn_mask,
                             **{k:v for k,v in cfg.to_dict().items() if v is not None})

    seq = out.sequences  # [NUM_BEAMS, prompt+gen]
    p_len = input_ids.shape[1]
    cont_ids, decoded = [], []
    for i in range(seq.size(0)):
        cont = seq[i][p_len:]
        cont_ids.append(cont)
        txt = tok.decode(cont, skip_special_tokens=True)
        txt = clean_continuation(txt)
        decoded.append(txt)

    if len(decoded) != NUM_BEAMS:
        decoded   = (decoded + [""]*NUM_BEAMS)[:NUM_BEAMS]
        cont_ids  = (cont_ids + [torch.empty(0, dtype=torch.long, device=model.device)]*NUM_BEAMS)[:NUM_BEAMS]
    return decoded, input_ids[0], cont_ids

def score_continuation_given_ids(model, tok, prompt_ids: torch.LongTensor, cont_ids: torch.LongTensor) -> float:
    if cont_ids.numel()==0: return float("nan")
    dev = model.device
    full = torch.cat([prompt_ids, cont_ids]).unsqueeze(0).to(dev)
    attn = torch.ones_like(full, dtype=torch.long, device=dev)
    with torch.no_grad():
        logits = model(input_ids=full, attention_mask=attn).logits
    logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets  = full[:, 1:]
    token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    p_len = prompt_ids.numel()
    mask = torch.zeros_like(token_lp, dtype=torch.bool)
    mask[:, p_len-1:] = True
    sel = token_lp[mask]
    return float(sel.mean().item()) if sel.numel() else float("nan")

# ---------- Column helpers ----------
def pick_task_text(row: pd.Series) -> str:
    for key in ("TaskDescription", "task_description", "Task", "Description", "Disease_description", "Case", "Prompt", "Input"):
        if key in row and isinstance(row[key], str) and row[key].strip():
            return str(row[key])
    return "\n".join([str(v) for v in row.values if isinstance(v, str)]).strip()

def pick_true_disease(row: pd.Series) -> str:
    for key in ("TrueDisease", "true_disease", "True", "Reference", "Reference_Disease"):
        if key in row and isinstance(row[key], str) and row[key].strip():
            return str(row[key])
    return ""

def pick_model_output(row: pd.Series) -> str:
    for key in ("ModelOutput", "model_output", "LLM_output", "Generated", "Response", "Model_answer", "ModelAnswer"):
        if key in row and isinstance(row[key], str) and row[key].strip():
            return str(row[key])
    return ""

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Treatment Likert evaluator with diverse beam search (HF Transformers).")
    ap.add_argument("--in_csv", default=IN_CSV)
    ap.add_argument("--out_csv", default=OUT_CSV)
    ap.add_argument("--model",  default=DEFAULT_MODEL)
    ap.add_argument("--start_row", type=int, default=0)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--save_every_n", type=int, default=1)
    args = ap.parse_args()

    model, tok = load_model_and_tokenizer(args.model)
    eos_ids = get_eos_ids(model, tok)
    df = pd.read_csv(args.in_csv)

    written_header = False
    if args.stream and os.path.exists(args.out_csv):
        try: os.remove(args.out_csv)
        except Exception: pass

    if args.start_row > 0:
        df = df.iloc[args.start_row:].reset_index(drop=True)

    finals, reasons, beams_labels, beams_scores, beams_texts = [], [], [], [], []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Treatment beam scoring"):
        task_text   = pick_task_text(row)
        true_dis    = pick_true_disease(row)
        model_out   = pick_model_output(row)

        if not task_text:
            finals.append("")
            reasons.append("")
            beams_labels.append(json.dumps([""]*NUM_BEAMS, ensure_ascii=False))
            beams_scores.append(json.dumps([float("nan")]*NUM_BEAMS))
            beams_texts.append(" ||| ".join([""]*NUM_BEAMS))
            continue

        prompt = build_chat_prompt(tok, task_text, true_dis, model_out)
        gens, prompt_ids, cont_ids_list = gen_beams(model, tok, prompt, eos_ids)

        seq_scores, per_scores, per_raw = [], [], []
        for cont_ids, g in zip(cont_ids_list, gens):
            g_txt = clean_continuation(g)
            per_raw.append(g_txt if g_txt else "")
            seq_scores.append(score_continuation_given_ids(model, tok, prompt_ids, cont_ids))
            per_scores.append(parse_score(g_txt))

        cnt = Counter([x for x in per_scores if x])
        if cnt:
            top_val, top_n = cnt.most_common(1)[0]
            agree = top_n / max(1, len(per_scores))
        else:
            top_val, agree = "", 0.0

        if agree >= AGREE_THRESHOLD and top_val:
            chosen_score = top_val
            chosen_reason = ""
        else:
            valid_idxs = [i for i,s in enumerate(per_scores) if s]
            if valid_idxs:
                best_i = max(valid_idxs, key=lambda i: (seq_scores[i] if seq_scores[i]==seq_scores[i] else float("-inf")))
            else:
                best_i = max(range(len(seq_scores)), key=lambda i: (seq_scores[i] if seq_scores[i]==seq_scores[i] else float("-inf"))) if seq_scores else 0
            chosen_score  = per_scores[best_i] if per_scores else ""
            chosen_reason = ""

        finals.append(chosen_score)
        reasons.append(chosen_reason)

        if len(per_scores) != NUM_BEAMS: per_scores = (per_scores + [""]*NUM_BEAMS)[:NUM_BEAMS]
        if len(seq_scores) != NUM_BEAMS: seq_scores = (seq_scores + [float("nan")]*NUM_BEAMS)[:NUM_BEAMS]
        if len(per_raw)    != NUM_BEAMS: per_raw    = (per_raw + [""]*NUM_BEAMS)[:NUM_BEAMS]

        beams_labels.append(json.dumps(per_scores, ensure_ascii=False))
        beams_scores.append(json.dumps(seq_scores))
        beams_texts.append(" ||| ".join(per_raw))

        if args.stream and (((idx+1) % args.save_every_n == 0) or (idx+1)==len(df)):
            out_row = row.to_dict()
            out_row.update({
                "reasoning": chosen_reason,
                "final_answer": chosen_score,
                "LLM_eval_score": chosen_score,
                "beam_final_labels": json.dumps(per_scores, ensure_ascii=False),
                "beam_sequence_scores": json.dumps(seq_scores),
                "raw_generations": " ||| ".join(per_raw),
            })
            pd.DataFrame([out_row]).to_csv(
                args.out_csv, mode=("a" if written_header else "w"),
                header=(not written_header), index=False
            )
            written_header = True

    if not args.stream:
        out = df.copy()
        out["reasoning"]             = reasons
        out["final_answer"]          = finals
        out["LLM_eval_score"]        = finals
        out["beam_final_labels"]     = beams_labels
        out["beam_sequence_scores"]  = beams_scores
        out["raw_generations"]       = beams_texts
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        out.to_csv(args.out_csv, index=False)
        print("Saved:", args.out_csv)

if __name__ == "__main__":
    main()

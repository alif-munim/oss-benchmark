#!/usr/bin/env python3
"""
EURORAD CoT diverse-beam inference + external rescoring (HF Transformers)
- Chat-model safe (uses tokenizer.apply_chat_template)
- Returns ALL beams (num_return_sequences=NUM_BEAMS)
- CSV is stable: every row has exactly NUM_BEAMS entries for beam_* and raw_generations
- Cleans role bleed, trims to first XML block, normalizes <final> to candidate list
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

# ---------- Config ----------
DEFAULT_MODEL = os.environ.get("MODEL_NAME", "openai/gpt-oss-120b")
LORA_PATH = os.environ.get("LORA_PATH", "")
DATA_DIR  = os.environ.get("DATA_DIR", ".")
IN_CSV    = os.environ.get("IN_CSV",  os.path.join(DATA_DIR, "eurorad_val.csv"))
OUT_CSV   = os.environ.get("OUT_CSV", os.path.join(DATA_DIR, "val_preds_beams_hf.csv"))

NUM_BEAMS         = int(os.environ.get("NUM_BEAMS", 5))
NUM_BEAM_GROUPS   = int(os.environ.get("NUM_BEAM_GROUPS", NUM_BEAMS))
DIVERSITY_PENALTY = float(os.environ.get("DIVERSITY_PENALTY", 0.5))
LENGTH_PENALTY    = float(os.environ.get("LENGTH_PENALTY", 1.0))
MAX_NEW_TOKENS    = int(os.environ.get("MAX_NEW_TOKENS", 1024))
REPETITION_PENALTY= float(os.environ.get("REPETITION_PENALTY", 1.0))  # 1.0=off
AGREE_THRESHOLD   = float(os.environ.get("AGREE_THRESHOLD", 0.5))

ATTN_IMPL         = os.environ.get("ATTN_IMPL", "eager")
DTYPE             = os.environ.get("DTYPE", "bfloat16")
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "1") == "1"
DEVICE_MAP        = os.environ.get("DEVICE_MAP", "auto")
LOAD_4BIT         = os.environ.get("LOAD_4BIT", "0") == "1"  # needs bitsandbytes

# ---------- Prompts ----------
SYS_PROMPT = (
    "You are a careful clinical reasoning assistant for radiology cases. "
    "Given a case description and a finite list of possible diagnoses, "
    "think step by step, then return a single final answer that matches "
    "exactly one label from the provided candidate list."
)
USER_TEMPLATE = (
    "Case description:\n{post}\n\n"
    "Candidate diagnoses (copy exactly one):\n{cands}\n\n"
    "Respond in the following strict XML format:\n"
    "<analysis>YOUR DETAILED REASONING HERE</analysis>\n"
    "<final>ONE LABEL COPIED VERBATIM FROM THE LIST</final>\n"
)

# ---------- Regex / cleaners ----------
FINAL_RE    = re.compile(r"<final>\s*(.*?)\s*</final>",    re.I | re.S)
ANALYSIS_RE = re.compile(r"<analysis>\s*(.*?)\s*</analysis>", re.I | re.S)
ROLE_BLEED_RE = re.compile(r"^\s*(assistant(final|analysis)?|system|user)\s*[:>]*\s*", re.I)
_PLACEH = "ONE LABEL COPIED VERBATIM"

def clean_continuation(txt: str) -> str:
    if not txt: return ""
    s = ROLE_BLEED_RE.sub("", txt.strip())
    # keep from first XML tag onward
    p1, p2 = s.lower().find("<analysis>"), s.lower().find("<final>")
    first = min([i for i in (p1, p2) if i != -1] or [-1])
    if first > 0: s = s[first:]
    # keep only first XML block (until first </final>)
    j = s.lower().find("</final>")
    if j != -1: s = s[: j + len("</final>")]
    return s.strip()

def is_placeholder(s: str) -> bool:
    u = re.sub(r"\s+", " ", (s or "").upper()).strip()
    return u.startswith("YOUR DETAILED REASONING") or (_PLACEH in u)

def parse_final_only(text: str) -> str:
    if not text: return ""
    m = FINAL_RE.search(text)
    val = (m.group(1) if m else "").strip()
    if not val:
        # fallback to last non-empty line
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        val = lines[-1] if lines else ""
    return "" if is_placeholder(val) else val

def parse_analysis(text: str) -> str:
    if not text: return ""
    m = ANALYSIS_RE.search(text)
    val = (m.group(1) if m else text).strip()
    return "" if is_placeholder(val) else val

def _norm(x: str) -> str:
    return re.sub(r"\s+", " ", (x or "").strip().lower()
                  .replace("–","-").replace("—","-").replace("-","-"))

def normalize_to_candidate(label: str, candidates: List[str]) -> str:
    """Exact/normalized match; if not found, allow unique prefix match; else ''."""
    if not label: return ""
    if label in candidates: return label
    cand_norm = {_norm(c): c for c in candidates}
    key = _norm(label)
    if key in cand_norm: return cand_norm[key]
    # unique prefix match (normalized)
    pref = [c for n,c in cand_norm.items() if n.startswith(key) or key.startswith(n)]
    if len(pref) == 1: return pref[0]
    # very short/ambiguous -> drop
    return ""

# ---------- Chat templating ----------
def build_chat_prompt(tok, post: str, candidates: List[str]) -> str:
    user = USER_TEMPLATE.format(post=post.strip(), cands="\n".join(candidates))
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
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=TRUST_REMOTE_CODE, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation=ATTN_IMPL, torch_dtype=dtype,
        device_map=DEVICE_MAP, trust_remote_code=TRUST_REMOTE_CODE, **qkw
    )
    if LORA_PATH:
        if not _HAS_PEFT: raise RuntimeError("PEFT not installed but LORA_PATH provided")
        model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    return model, tok

# ---------- Generation & rescoring ----------
def gen_cot_beams(model, tok, chat_prompt: str, eos_ids: Union[int, List[int]]
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
        txt = clean_continuation(txt)           # drop bleed/preamble
        decoded.append(txt)

    # strong invariant: always NUM_BEAMS
    if len(decoded) != NUM_BEAMS:
        decoded = (decoded + [""]*NUM_BEAMS)[:NUM_BEAMS]
        cont_ids = (cont_ids + [torch.empty(0, dtype=torch.long, device=model.device)]*NUM_BEAMS)[:NUM_BEAMS]
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

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
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

    # streaming start fresh
    written_header = False
    if args.stream and os.path.exists(args.out_csv):
        try: os.remove(args.out_csv)
        except Exception: pass

    if args.start_row > 0:
        df = df.iloc[args.start_row:].reset_index(drop=True)

    req = ["PostDescription","DifferentialDiagnosisList"]
    miss = [c for c in req if c not in df.columns]
    if miss: raise ValueError(f"Missing required columns: {miss}")

    finals, reasons, beams_labels, beams_scores, beams_texts = [], [], [], [], []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="CoT beams"):
        post = ("" if pd.isna(row["PostDescription"]) else str(row["PostDescription"]))
        cand_raw = ("" if pd.isna(row["DifferentialDiagnosisList"]) else str(row["DifferentialDiagnosisList"]))
        candidates = [c.strip() for c in re.split(r"\n|,", cand_raw) if c.strip()]

        if not post or not candidates:
            finals.append("")
            reasons.append("")
            beams_labels.append(json.dumps([""]*NUM_BEAMS, ensure_ascii=False))
            beams_scores.append(json.dumps([float("nan")]*NUM_BEAMS))
            beams_texts.append(" ||| ".join([""]*NUM_BEAMS))
            continue

        prompt = build_chat_prompt(tok, post, candidates)
        gens, prompt_ids, cont_ids_list = gen_cot_beams(model, tok, prompt, eos_ids)

        # per-beam: score + label/analysis (strictly from cleaned XML)
        scores, per_labels, per_xml = [], [], []
        for cont_ids, g in zip(cont_ids_list, gens):
            # ensure only first XML block is kept in CSV output
            g_xml = clean_continuation(g)
            per_xml.append(g_xml if g_xml else "")
            scores.append(score_continuation_given_ids(model, tok, prompt_ids, cont_ids))
            raw_lab = parse_final_only(g_xml)
            per_labels.append(normalize_to_candidate(raw_lab, candidates))

        # consensus vs. best score
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
            # choose highest external score among beams that produced a valid candidate
            valid_idxs = [i for i,l in enumerate(per_labels) if l]
            if valid_idxs:
                best_i = max(valid_idxs, key=lambda i: (scores[i] if scores[i]==scores[i] else float("-inf")))
            else:
                # fallback to globally best score even if label empty
                best_i = max(range(len(scores)), key=lambda i: (scores[i] if scores[i]==scores[i] else float("-inf"))) if scores else 0
            chosen_label  = per_labels[best_i] if per_labels else ""
            chosen_reason = parse_analysis(per_xml[best_i]) if per_xml else ""

        # append row outputs (force exact lengths)
        finals.append(chosen_label)
        reasons.append(chosen_reason)

        if len(per_labels) != NUM_BEAMS: per_labels = (per_labels + [""]*NUM_BEAMS)[:NUM_BEAMS]
        if len(scores)     != NUM_BEAMS: scores     = (scores + [float("nan")]*NUM_BEAMS)[:NUM_BEAMS]
        if len(per_xml)    != NUM_BEAMS: per_xml    = (per_xml + [""]*NUM_BEAMS)[:NUM_BEAMS]

        beams_labels.append(json.dumps(per_labels, ensure_ascii=False))
        beams_scores.append(json.dumps(scores))
        beams_texts.append(" ||| ".join(per_xml))

        # streaming write
        if args.stream and (((idx+1) % args.save_every_n == 0) or (idx+1)==len(df)):
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

    # final write
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

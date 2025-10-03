#!/usr/bin/env python3
# benchmarks/eurorad/eval_finetune.py
#
# Evaluate a locally fine-tuned GPT-OSS model with a LoRA adapter checkpoint or the base model.
# - Matches training format: expects model to emit <analysis>...</analysis><final>...</final>
# - Batched generation (configurable --batch_size)
# - Sampling (not greedy) with the model's default temperature
# - Streams GOLD + full raw model output when --stream
# - Saves raw model reasoning (analysis block) in CSV
# - Robust EOS: do NOT stop on <|end|>; optionally stop on </final> or <|return|> if they are single tokens
# - Robust extraction when <final> is missing (fallback to last-contained option, fuzzy, etc.)
# - Prints tokenizer/model/generation configs (temperature, top_p, etc.) before inference
#
# CSV columns expected:
#   case_id, OriginalDescription, PostDescription, DifferentialDiagnosisList, FinalDiagnosis
#
# Outputs (added if missing):
#   model_answer_raw, model_reasoning_raw, model_answer, match_type, correct, options
#
# Example:
#   python benchmarks/eurorad/eval_finetune.py \
#     --model_dir outputs/checkpoint-552 \
#     --base_model openai/gpt-oss-120b \
#     --input_csv eurorad_test.csv \
#     --results results \
#     --batch_size 1 --max_new_tokens 8192 --stream

import re, argparse, unicodedata, difflib, sys
from pathlib import Path
from typing import Optional, Tuple, List, Any
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

# ====== Speed toggles (safe on A100s) ======
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

# ---------- Prompts (mirror training setup) ----------
DEFAULT_SYSTEM = "You are an expert radiologist."

# Training-style instructions inside the user turn; model should emit XML-ish blocks.
DEFAULT_USER_TEMPLATE = (
    "You are an expert radiologist. You will receive a case and a finite list of candidate diagnoses. "
    "Reason step by step, then pick exactly one diagnosis copied verbatim from the list. "
    "Use your standard radiology knowledge to interpret the given findings (don't invent new findings).\n\n"
    "PostDescription / Clinical History + Imaging Summary:\n{case_text}\n\n"
    "DifferentialDiagnosisList (comma-separated options):\n{options_csv}\n\n"
    "Response rules:\n"
    "1) Provide <analysis> content and order: A) Key findings; B) Candidate pass/fail with 1–2 pros/cons each; "
    "C) Ranking with tie-breaks (pathognomonic findings win; exact anatomic site/side; specific subtype over umbrella; sanity check).\n"
    "2) In <final>, copy the chosen label verbatim from the list. Do not include any other text.\n\n"
    "Return exactly in two blocks and nothing else:\n"
    "<analysis> ... </analysis><final> ... </final>"
)

# ---------- Regex & utilities ----------
# XML-style blocks (training) and channel-style (just in case)
FINAL_XML_RE = re.compile(r"<final>(.*?)</final>", re.DOTALL | re.IGNORECASE)
ANALYSIS_XML_RE = re.compile(r"<analysis>(.*?)</analysis>", re.DOTALL | re.IGNORECASE)

FINAL_CH_RE = re.compile(r"<\|channel\|>final<\|message\|>(.*)$", re.DOTALL)
ANALYSIS_CH_RE = re.compile(
    r"<\|channel\|>analysis<\|message\|>(.*?)(?=(?:<\|channel\|>final<\|message\|>)|$)",
    re.DOTALL,
)

CLEAN_TAG_RE = re.compile(r"<\|[^|>]+?\|>")  # strips chat special tags (not XML)
XML_TAG_RE = re.compile(r"</?(analysis|final)\s*>", re.IGNORECASE)

def extract_final(text: str) -> str:
    if not text:
        return ""
    m = FINAL_XML_RE.search(text)
    if m:
        return m.group(1).strip()
    m = FINAL_CH_RE.search(text)
    return (m.group(1).strip() if m else "")

def extract_reasoning(text: str) -> str:
    if not text:
        return ""
    m = ANALYSIS_XML_RE.search(text)
    if m:
        return m.group(1).strip()
    m = ANALYSIS_CH_RE.search(text)
    return (m.group(1).strip() if m else "")

def drop_analysis_block(text: str) -> str:
    if not text:
        return text
    return ANALYSIS_XML_RE.sub("", ANALYSIS_CH_RE.sub("", text))

def strip_special(s: str) -> str:
    # Remove both chat tags and XML analysis/final tags for mapping
    s = CLEAN_TAG_RE.sub("", s or "")
    s = XML_TAG_RE.sub("", s)
    return s.strip()

def build_options_list(s: str) -> List[str]:
    opts = [o.strip() for o in (s or "").split(",") if o.strip()]
    seen, out = set(), []
    for o in opts:
        if o not in seen:
            seen.add(o)
            out.append(o)
    return out

def norm_text(s: str) -> str:
    t = unicodedata.normalize("NFKC", s or "")
    t = t.replace("–", "-").replace("—", "-")
    t = " ".join(t.strip().split())
    return t.lower()

def pick_last_contained(clean_text: str, options: List[str]) -> Tuple[str, str]:
    ntext = norm_text(clean_text)
    last_hit, last_pos = None, -1
    for o in options:
        p = ntext.rfind(norm_text(o))
        if p > last_pos and p != -1:
            last_pos, last_hit = p, o
    if last_hit is None:
        return "", "no_match"
    return last_hit, "contains-last"

def map_to_option(raw_answer: str, options: List[str]) -> Tuple[str, str]:
    """Robust mapping:
       1) exact
       2) normalized exact
       3) containment (clean & normalized)
       4) fuzzy (raw, then normalized)
    """
    if not raw_answer or not options:
        return "", "no_match"

    raw = raw_answer.strip()
    if raw in options:
        return raw, "exact"

    norm2opt = {norm_text(o): o for o in options}
    nr = norm_text(raw)
    if nr in norm2opt:
        return norm2opt[nr], "normalized"

    clean = strip_special(raw)
    nclean = norm_text(clean)
    for o in options:
        if o in clean:
            return o, "contains"
        if norm_text(o) in nclean:
            return o, "contains-normalized"

    cand = difflib.get_close_matches(raw, options, n=1, cutoff=0.8)
    if cand:
        return cand[0], "fuzzy"

    norm_opts = list(norm2opt.keys())
    candn = difflib.get_close_matches(nr, norm_opts, n=1, cutoff=0.9)
    if candn:
        return norm2opt[candn[0]], "fuzzy"

    return "", "no_match"

def build_user_prompt(row: pd.Series, user_template: str = DEFAULT_USER_TEMPLATE) -> str:
    case_id = str(row.get("case_id", "")).strip()
    desc = str(row.get("PostDescription") or row.get("OriginalDescription") or "").strip()
    options = build_options_list(row.get("DifferentialDiagnosisList", ""))
    options_csv = ", ".join(options) if options else ""
    return user_template.format(case_id=case_id, case_text=desc, options_csv=options_csv)

def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s)

def out_path(input_csv: Path, model_dir: Path, results_dir: Path, max_out: Optional[int]) -> Path:
    tag = sanitize(model_dir.name)
    parts = [input_csv.stem, tag, "local"]
    if max_out:
        parts.append(f"max{int(max_out)}")
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / ("_".join(parts) + ".csv")

def needs_rerun(raw: str, mapped: str) -> bool:
    if isinstance(mapped, str) and mapped.strip():
        return False
    if not raw or not str(raw).strip():
        return True
    if str(raw).strip().upper().startswith("ERROR:"):
        return True
    return True

# ---------- Token helpers ----------
def _single_token_id(tokenizer, literal: str):
    """Return token id if `literal` encodes to exactly one token; else None."""
    if not literal:
        return None
    try:
        tid = tokenizer.convert_tokens_to_ids(literal)
        if isinstance(tid, int) and tid >= 0:
            return tid
    except Exception:
        pass
    try:
        ids = tokenizer.encode(literal, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    except Exception:
        pass
    return None

def _decode_token_list(tokenizer, ids: List[int]) -> List[str]:
    out = []
    for t in ids:
        try:
            s = tokenizer.decode([t], skip_special_tokens=False)
        except Exception:
            s = ""
        out.append(s)
    return out

# ---------- Model loading ----------
def load_local_model(model_dir: Path, base_model_hint: Optional[str] = "openai/gpt-oss-120b"):
    # Detect whether model_dir is a local folder (adapter/merged) or a HF repo id
    is_local = isinstance(model_dir, Path) and model_dir.exists()
    model_id_or_path = str(model_dir) if not is_local else model_dir

    # Tokenizer source:
    # - If this is an adapter folder, we need the base tokenizer.
    # - Otherwise use the same id/path we’re loading the model from.
    is_adapter = is_local and (
        (model_dir / "adapter_config.json").exists() or (model_dir / "adapter_model.safetensors").exists()
    )
    tok_src = base_model_hint if is_adapter else model_id_or_path

    tokenizer = AutoTokenizer.from_pretrained(str(tok_src), use_fast=True)
    tokenizer.padding_side = "left"  # decoder-only: left padding recommended
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_to_load = base_model_hint if is_adapter else model_id_or_path
    model = AutoModelForCausalLM.from_pretrained(
        str(base_to_load),
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    if is_adapter:
        model = PeftModel.from_pretrained(model, str(model_dir), device_map="auto")
        try:
            model = model.merge_and_unload()
        except Exception:
            pass

    # Build generation config
    gen_cfg = GenerationConfig.from_model_config(model.config)

    # Ensure PAD
    if tokenizer.pad_token_id is not None:
        gen_cfg.pad_token_id = tokenizer.pad_token_id

    # Robust EOS: start with base EOS; optionally add '</final>' or '<|return|>' if single tokens.
    # IMPORTANT: do NOT add '<|end|>' (turn terminator) — it would truncate before <final>.
    eos_ids = set()
    if isinstance(gen_cfg.eos_token_id, int) and gen_cfg.eos_token_id >= 0:
        eos_ids.add(int(gen_cfg.eos_token_id))
    elif isinstance(gen_cfg.eos_token_id, (list, tuple)):
        for x in gen_cfg.eos_token_id:
            if isinstance(x, int) and x >= 0:
                eos_ids.add(int(x))
    elif tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))

    for lit in ("</final>", "<|return|>"):
        tid = _single_token_id(tokenizer, lit)
        if tid is not None:
            eos_ids.add(tid)

    gen_cfg.eos_token_id = sorted(eos_ids) if eos_ids else tokenizer.eos_token_id

    model.generation_config = gen_cfg
    model.config.pad_token_id = gen_cfg.pad_token_id
    model.config.eos_token_id = gen_cfg.eos_token_id
    model.config.use_cache = True
    model.eval()
    return tokenizer, model

# ---------- Chat rendering (use the tokenizer's chat template) ----------
def render_chat(tokenizer, system_text: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

# ---------- Pretty-print effective run config ----------
def _get(cfg: Any, name: str, fallback=None):
    return getattr(cfg, name, fallback)

def print_run_config(tokenizer, model, gen_kwargs, batch_size, max_new_tokens, max_input_len):
    print("\n=== Run config (inference) ===")
    # Tokenizer
    print("Tokenizer:")
    print(f"  name_or_path: {getattr(tokenizer, 'name_or_path', 'unknown')}")
    print(f"  padding_side: {tokenizer.padding_side}")
    print(f"  pad_token_id: {tokenizer.pad_token_id} ({tokenizer.decode([tokenizer.pad_token_id]) if tokenizer.pad_token_id is not None else 'None'})")
    print(f"  eos_token_id (base): {tokenizer.eos_token_id} ({tokenizer.decode([tokenizer.eos_token_id]) if tokenizer.eos_token_id is not None else 'None'})")

    # Model
    print("\nModel:")
    print(f"  name_or_path: {getattr(model, 'name_or_path', 'unknown')}")
    print(f"  dtype: {next((p.dtype for p in model.parameters() if p.requires_grad or True), torch.bfloat16)}")
    dm = getattr(model, "hf_device_map", None)
    print(f"  device_map: {'auto' if dm is None else 'custom'}")
    if dm:
        # Print a small summary (first few entries) to avoid noise
        items = list(dm.items())
        head = ", ".join(f"{k}:{v}" for k, v in items[:5])
        tail = "" if len(items) <= 5 else f", ... (+{len(items)-5} more)"
        print(f"    {head}{tail}")

    # GenerationConfig (model defaults) vs effective gen_kwargs
    gc = getattr(model, "generation_config", GenerationConfig())
    def eff(name, default=None):
        # Prefer explicit gen_kwargs, else model.generation_config, else default
        return gen_kwargs.get(name, _get(gc, name, default))

    print("\nSampling / Generation:")
    print(f"  do_sample: {eff('do_sample', False)}")
    print(f"  temperature: {eff('temperature', None)}")
    print(f"  top_p: {eff('top_p', None)}")
    print(f"  top_k: {eff('top_k', None)}")
    print(f"  typical_p: {eff('typical_p', None)}")
    print(f"  repetition_penalty: {eff('repetition_penalty', None)}")
    print(f"  no_repeat_ngram_size: {eff('no_repeat_ngram_size', None)}")
    print(f"  num_beams: {eff('num_beams', 1)}")
    print(f"  length_penalty: {eff('length_penalty', None)}")
    print(f"  min_new_tokens: {eff('min_new_tokens', None)}")
    print(f"  max_new_tokens: {max_new_tokens}")

    # EOS set
    eos_ids = eff("eos_token_id", None)
    eos_list = eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids] if eos_ids is not None else []
    eos_decoded = _decode_token_list(tokenizer, [e for e in eos_list if isinstance(e, int)])
    print(f"  eos_token_id (effective): {eos_list}  decoded={eos_decoded}")
    print(f"  pad_token_id (effective): {eff('pad_token_id', None)}")

    # Data path params
    print("\nBatching / Limits:")
    print(f"  batch_size: {batch_size}")
    print(f"  max_input_len (prompt tokens): {max_input_len}")
    print("=== End run config ===\n", flush=True)

# ---------- Runner ----------
def main():
    ap = argparse.ArgumentParser(description="Evaluate a locally fine-tuned GPT-OSS model on a radiology CSV (batched, streams, robust extraction).")
    ap.add_argument("--model_dir", type=str, required=True, help="Path to LoRA adapter directory, merged model dir, or HF repo id (e.g., openai/gpt-oss-120b).")
    ap.add_argument("--base_model", type=str, default="openai/gpt-oss-120b", help="Base model to load if model_dir is an adapter.")
    ap.add_argument("--input_csv", type=str, required=True)
    ap.add_argument("--results", type=str, default="results")
    ap.add_argument("--output_csv", type=str, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--system", type=str, default=DEFAULT_SYSTEM)
    ap.add_argument("--user_template", type=str, default=DEFAULT_USER_TEMPLATE)
    ap.add_argument("--limit", type=int, default=None, help="Only run first N rows.")
    ap.add_argument("--stream", action="store_true", help="Print GOLD + full raw model output + mapping for each example.")
    ap.add_argument("--batch_size", type=int, default=1, help="Generation batch size (1 is safest for very long decode).")
    ap.add_argument("--max_input_len", type=int, default=2048, help="Max prompt tokens (truncate for safety).")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    input_csv = Path(args.input_csv)
    results_dir = Path(args.results)

    if args.output_csv:
        out_csv = Path(args.output_csv)
    else:
        out_csv = out_path(input_csv, model_dir, results_dir, args.max_new_tokens)
    print(f"Output: {out_csv}")

    # Load data
    df = pd.read_csv(input_csv)
    if args.limit:
        df = df.head(int(args.limit))

    # Ensure output columns
    for col in ("model_answer_raw", "model_reasoning_raw", "model_answer", "match_type", "correct", "options"):
        if col not in df.columns:
            df[col] = ""

    # Resume merge
    if args.resume and out_csv.exists() and input_csv.resolve() != out_csv.resolve():
        prev = pd.read_csv(out_csv)
        for col in ("model_answer_raw", "model_reasoning_raw", "model_answer", "match_type", "correct", "options"):
            if col in prev.columns:
                vals = list(prev[col])[:len(df)]
                if len(vals) < len(df): vals += [""]*(len(df)-len(vals))
                df[col] = vals

    # Build todo list
    indices = list(df.index)
    todo = [i for i in indices if needs_rerun(str(df.at[i,"model_answer_raw"]), str(df.at[i,"model_answer"]))]
    print(f"resume: {len(indices)-len(todo)} done / {len(indices)} total, {len(todo)} to run")

    if not todo:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        if "FinalDiagnosis" in df.columns:
            gold = df["FinalDiagnosis"].astype(str).apply(norm_text)
            pred = df["model_answer"].astype(str).apply(norm_text)
            acc = (gold == pred).mean() if len(df) else 0.0
            print(f"Accuracy: {acc:.3f}")
        print(f"Saved (unchanged): {out_csv}")
        return

    # Load model
    print("Loading model…")
    tokenizer, model = load_local_model(model_dir, base_model_hint=args.base_model)

    # Generation kwargs (sampling; temperature from model default)
    default_temp = float(getattr(model.generation_config, "temperature", 1.0) or 1.0)
    gen_kwargs = dict(
        do_sample=True,
        temperature=default_temp,
        num_beams=1,
        max_new_tokens=int(args.max_new_tokens),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=model.generation_config.eos_token_id,
        use_cache=True,
    )

    # Print effective configs once before inference
    print_run_config(tokenizer, model, gen_kwargs, batch_size=max(1, int(args.batch_size)),
                     max_new_tokens=int(args.max_new_tokens), max_input_len=int(args.max_input_len))

    # Batched inference
    todo_idx = todo
    bsz = max(1, int(args.batch_size))

    with torch.inference_mode():
        for start in tqdm(range(0, len(todo_idx), bsz), desc="Evaluating (batched)"):
            batch_ids = todo_idx[start:start+bsz]
            rows = [df.loc[i] for i in batch_ids]

            # Build prompts and options
            user_prompts = [build_user_prompt(r, args.user_template) for r in rows]
            options_lists = [build_options_list(r.get("DifferentialDiagnosisList", "")) for r in rows]

            # Render chat prompts -> batch tokenize (truncate for safety)
            rendered = [render_chat(tokenizer, args.system, up) for up in user_prompts]
            enc = tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(args.max_input_len),
            ).to(model.device)

            # Generate (sampling)
            out = model.generate(**enc, **gen_kwargs)

            # Slice continuations & decode
            for j, row_idx in enumerate(batch_ids):
                # left padding => input length = sum(attention_mask)
                input_len = int(enc["attention_mask"][j].sum().item()) if "attention_mask" in enc else int((enc["input_ids"][j] != tokenizer.pad_token_id).sum().item())
                cont_ids = out[j, input_len:]

                raw_full = tokenizer.decode(cont_ids, skip_special_tokens=False).strip()
                raw_final = extract_final(raw_full)
                raw_reason = extract_reasoning(raw_full)

                options = options_lists[j]

                # Prefer explicit <final> if present; else drop analysis and map
                if raw_final:
                    mapped_region = raw_final
                else:
                    mapped_region = drop_analysis_block(raw_full)

                mapped, mtype = map_to_option(mapped_region, options)
                if not mapped:
                    # Fallback: pick the *last* contained option in the whole text (often after reasoning)
                    mapped, mtype = pick_last_contained(strip_special(raw_full), options)

                # Write back
                df.at[row_idx, "model_answer_raw"] = raw_full
                df.at[row_idx, "model_reasoning_raw"] = raw_reason
                df.at[row_idx, "model_answer"] = mapped
                df.at[row_idx, "match_type"] = mtype
                df.at[row_idx, "options"] = " | ".join(options)

                if "FinalDiagnosis" in df.columns and mapped:
                    gold = norm_text(str(df.loc[row_idx, "FinalDiagnosis"]))
                    df.at[row_idx, "correct"] = int(norm_text(mapped) == gold)
                else:
                    df.at[row_idx, "correct"] = ""

                # Streaming print
                if args.stream:
                    cid = rows[j].get("case_id", row_idx)
                    gold_txt = str(rows[j].get("FinalDiagnosis", "")).strip()
                    print(f"\n[Case {cid}] GOLD: {gold_txt}", flush=True)
                    print(f"[Case {cid}] RAW (verbatim):\n{raw_full}\n", flush=True)
                    if raw_final:
                        print(f"[Case {cid}] FINAL (used for mapping):\n{raw_final}\n", flush=True)
                    print(f"[Case {cid}] mapped: {mapped}  ({mtype})", flush=True)

            # incremental save
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv, index=False)

    # Final save + accuracy
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    if "FinalDiagnosis" in df.columns:
        mask = df["model_answer"].astype(str).str.len() > 0
        if mask.any():
            gold = df.loc[mask, "FinalDiagnosis"].astype(str).apply(norm_text)
            pred = df.loc[mask, "model_answer"].astype(str).apply(norm_text)
            acc = (gold == pred).mean() if len(gold) else 0.0
            print(f"Accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()

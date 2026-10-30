"""
Standalone KV-cache judge experiment (merged kv_cache.py + kv_cache_vllm.py).

Shuffles the personas, runs the first BATCH_SIZE with no prefix cache and the
next BATCH_SIZE reusing the cached system prompt, then compares the two.

  python judge_v1.py                          # vLLM backend (default)
  python judge_v1.py --backend hf             # HF transformers backend

Needs only the judge prompt (.jinja2) and the JD / persona CSVs as inputs.
"""
import os
# Keep the vLLM engine in this process so reset_prefix_cache() and the torch
# memory stats in _print_comparison see the real engine. Must be set before vllm is imported.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import argparse
import copy
import json
import re
import time
import torch
import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from jinja2.exceptions import TemplateError
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

MODEL_ID       = "Qwen/Qwen3-4B"
DATA_PATH      = "/home/meghss/Project_Biases_in_Job_Descriptions/CSV_Dataset_Files/Sample_job_descriptions_balanced_300.csv"
PERSONAS_PATH  = "/home/meghss/Project_Biases_in_Job_Descriptions/personas_300/Lama_3.1_8B.csv"
PROMPT_PATH    = "/home/meghss/Project_Biases_in_Job_Descriptions/Judge/judge_prompt_opus_5.5.jinja2"
RESULTS_DIR    = "/home/meghss/Project_Biases_in_Job_Descriptions/Output_Files"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
DTYPES         = {"float16": torch.float16, "bfloat16": torch.bfloat16, "auto": "auto"}
MAX_NEW_TOKENS = 4096   # 11-category judge JSON; worked examples are 1.6-2k tokens
# Qwen3 recommended non-thinking sampling settings
TEMPERATURE    = 0.7
TOP_P          = 0.8
TOP_K          = 20
BATCH_SIZE     = 100
SEED           = 42

# Set in __main__ from the command line.
SYSTEM_PROMPT  = None
RESULTS_PATH   = None
HF_DTYPE       = torch.float16 if torch.cuda.is_available() else torch.float32

# ---------------------------------------------------------------------------
# Prompt and data
# ---------------------------------------------------------------------------

def load_system_prompt(path: str = PROMPT_PATH) -> str:
    env = Environment(loader=FileSystemLoader(os.path.dirname(path)), undefined=StrictUndefined)
    return env.get_template(os.path.basename(path)).render().strip()


def build_user_input(jd: str, persona: str, religious_exemption: bool = False) -> str:
    """User turn in the format the judge prompt expects (Section 1)."""
    return (
        f"<job_description>\n{jd}\n</job_description>\n"
        f"<generated_persona>\n{persona}\n</generated_persona>\n"
        f"<religious_exemption_preapproved>{str(religious_exemption).lower()}"
        f"</religious_exemption_preapproved>"
    )


def load_jd_snippets():
    df = pd.read_csv(DATA_PATH)
    df["jd_id"] = [f"job_{i}" for i in range(len(df))]
    return df


def load_personas(path):
    job_ids = load_jd_snippets()
    df = pd.read_csv(path)
    df_merged = pd.merge(df, job_ids, on="jd_id", how="inner")
    personas = df_merged[["jd_id","run","description","persona","searched_role"]]

    return personas


def build_full_prompt(tokenizer, user_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_prefix_only(tokenizer) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TemplateError:
        # Some templates (e.g. Qwen3.5+) refuse a system-only conversation.
        # Use the text every full prompt shares, cut after its last special
        # token so the cut lands on a token boundary.
        common = os.path.commonprefix([build_full_prompt(tokenizer, "A"),
                                       build_full_prompt(tokenizer, "B")])
        specials = [t.content for t in tokenizer.added_tokens_decoder.values() if t.special]
        end = max((common.rfind(t) + len(t) for t in specials if t in common), default=0)
        if end == 0:
            raise
        return common[:end]

# ---------------------------------------------------------------------------
# HF transformers backend
# ---------------------------------------------------------------------------

def load_hf(model_id: str = MODEL_ID):
    print(f"Loading {model_id} on {DEVICE} ({HF_DTYPE})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=HF_DTYPE,
        device_map="auto",
    )
    model.eval()
    print(f"Loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B\n")
    return model, tokenizer


def compute_prefix_cache_hf(model, tokenizer) -> tuple:

    prefix_text = build_prefix_only(tokenizer)
    prefix_ids  = tokenizer(prefix_text, return_tensors="pt").input_ids.to(DEVICE)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        out = model(input_ids=prefix_ids, use_cache=True, return_dict=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    kv      = out.past_key_values
    mem_mb  = _kv_memory_mb_hf(kv)
    n_tok   = prefix_ids.shape[-1]

    print(f"  Prefix cached: {n_tok} tokens | "
          f"{mem_mb:.1f} MB | computed in {t1-t0:.3f}s")
    return kv, n_tok, t1 - t0, mem_mb


def clone_cache(kv) -> DynamicCache:
    """
    Deep-copy the prefix DynamicCache before each generate() call.
    Without this, the cache grows with each generation and subsequent
    calls see a polluted prefix.

    transformers>=4.56 replaced DynamicCache.key_cache/value_cache lists
    with a list of per-layer objects (cache.layers[i].keys/.values), so a
    plain copy.deepcopy is the version-agnostic way to clone it.
    """
    if isinstance(kv, DynamicCache):
        return copy.deepcopy(kv)

    fresh = DynamicCache()
    for k, v in kv:
        fresh.update(k.clone(), v.clone(), layer_idx=len(fresh.layers))
    return fresh


def _kv_memory_mb_hf(kv) -> float:
    total = 0
    if isinstance(kv, DynamicCache):
        for layer in kv.layers:
            total += layer.keys.nelement() * layer.keys.element_size()
            total += layer.values.nelement() * layer.values.element_size()
    else:
        for layer in kv:
            for t in layer:
                total += t.nelement() * t.element_size()
    return total / 1024 ** 2


def infer_no_cache(model, tokenizer, user_text: str) -> dict:
    prompt    = build_full_prompt(tokenizer, user_text)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    n_in      = input_ids.shape[-1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n_out   = out.shape[-1] - n_in
    decoded = tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()

    return {
        "input_tokens":      n_in,
        "output_tokens":     n_out,
        "tokens_computed":   n_in,       # recomputed everything
        "tokens_saved":      0,
        "time_s":            elapsed,
        "prefill_tok_per_s": n_in / elapsed,
        "output":            decoded,
    }


def infer_with_cache(model, tokenizer, user_text: str,
                     prefix_kv, prefix_len: int) -> dict:
    full_prompt   = build_full_prompt(tokenizer, user_text)
    prefix_text   = build_prefix_only(tokenizer)
    user_turn     = full_prompt[len(prefix_text):]   # only the new tokens

    full_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(DEVICE)
    user_ids = tokenizer(user_turn,   return_tensors="pt").input_ids.to(DEVICE)

    n_in_total = full_ids.shape[-1]    # for fair token comparison
    n_user     = user_ids.shape[-1]    # what we actually compute

    cached_kv = clone_cache(prefix_kv)

    # generate() needs an attention_mask spanning the *full* sequence
    # (cached prefix + new user turn) when past_key_values is passed in
    # directly — without it, cache-length/position bookkeeping breaks.
    attention_mask = torch.ones((1, prefix_len + n_user),
                                 dtype=torch.long, device=DEVICE)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        out = model.generate(
            user_ids,
            attention_mask=attention_mask,
            past_key_values=cached_kv,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n_out   = out.shape[-1] - user_ids.shape[-1]
    decoded = tokenizer.decode(out[0][n_user:], skip_special_tokens=True).strip()

    return {
        "input_tokens":      n_in_total,
        "output_tokens":     n_out,
        "tokens_computed":   n_user,          # only the user turn was computed
        "tokens_saved":      prefix_len,      # prefix was reused
        "time_s":            elapsed,
        "prefill_tok_per_s": n_user / elapsed,
        "output":            decoded,
    }


def run_experiment_hf(model, tokenizer):
    df = load_personas(PERSONAS_PATH).sample(frac=1, random_state=SEED).reset_index(drop=True)
    batch_nc = df.iloc[:BATCH_SIZE]
    batch_wc = df.iloc[BATCH_SIZE:2 * BATCH_SIZE]
    print(f"Loaded {len(df)} personas from {os.path.basename(PERSONAS_PATH)}, shuffled (seed={SEED}). "
          f"No-cache batch: {len(batch_nc)} | cached batch: {len(batch_wc)}\n")

    print("=" * 65)
    print("  Computing prefix KV cache (one-time cost)")
    print("=" * 65)
    prefix_kv, prefix_len, prefix_time, kv_mb = compute_prefix_cache_hf(model, tokenizer)

    # Warm-up (not recorded) so CUDA init overhead doesn't penalise batch 1.
    # Uses a persona outside both batches.
    warm = df.iloc[-1]
    infer_no_cache(model, tokenizer, build_user_input(warm["description"], warm["persona"]))

    nc_rows = _run_batch("No Cache", batch_nc,
                         lambda text: infer_no_cache(model, tokenizer, text))
    wc_rows = _run_batch("KV Cache", batch_wc,
                         lambda text: infer_with_cache(model, tokenizer, text,
                                                       prefix_kv, prefix_len))

    pd.DataFrame(nc_rows + wc_rows).to_csv(RESULTS_PATH, index=False)
    print(f"\nPer-persona results saved to {RESULTS_PATH}")

    _print_comparison(nc_rows, wc_rows, prefix_len, prefix_time, kv_mb)

# ---------------------------------------------------------------------------
# vLLM backend (prefix caching)
# ---------------------------------------------------------------------------

def load_vllm(model_id: str = MODEL_ID, tensor_parallel_size: int = 1, dtype: str = "float16",
              max_model_len: int | None = None, language_model_only: bool = False,
              disable_custom_all_reduce: bool = False, gpu_memory_utilization: float = 0.90):
    from vllm import LLM
    print(f"Loading {model_id} with vLLM (tp={tensor_parallel_size}, {dtype})...")
    # Prefix caching is on; the no-cache batch flushes it before every call.
    llm = LLM(
        model=model_id,
        dtype=dtype,                   # float16 default matches the HF experiment
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        language_model_only=language_model_only,
        disable_custom_all_reduce=disable_custom_all_reduce,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
    )
    print("Loaded.\n")
    return llm, llm.get_tokenizer()


def _sampling(max_tokens: int = MAX_NEW_TOKENS):
    from vllm import SamplingParams
    return SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, max_tokens=max_tokens)


def _tokens(tokenizer, text: str) -> list:
    # Chat templates already add BOS (e.g. Llama's <|begin_of_text|>); don't add a second.
    return tokenizer(text, add_special_tokens=False).input_ids


def _generate(llm, tokenizer, prompt: str):
    t0 = time.perf_counter()
    out = llm.generate([{"prompt_token_ids": _tokens(tokenizer, prompt)}], _sampling(), use_tqdm=False)[0]
    return out, time.perf_counter() - t0


def _kv_memory_mb_vllm(llm, n_tokens: int) -> float:
    # Attention KV only; hybrid models (Qwen3.5+) also keep a fixed-size linear-attention state.
    cfg = llm.llm_engine.model_config.hf_text_config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    layer_types = getattr(cfg, "layer_types", None)
    n_attn = layer_types.count("full_attention") if layer_types else cfg.num_hidden_layers
    per_token = 2 * n_attn * cfg.num_key_value_heads * head_dim * 2  # K+V, 2 bytes
    return per_token * n_tokens / 1024 ** 2


def compute_prefix_cache_vllm(llm, tokenizer) -> tuple:
    """Prime vLLM's prefix cache with the system prompt (one-time cost)."""
    llm.reset_prefix_cache()
    prefix_text = build_prefix_only(tokenizer)
    prefix_ids  = _tokens(tokenizer, prefix_text)
    n_tok = len(prefix_ids)

    t0 = time.perf_counter()
    llm.generate([{"prompt_token_ids": prefix_ids}], _sampling(max_tokens=1), use_tqdm=False)
    elapsed = time.perf_counter() - t0

    mem_mb = _kv_memory_mb_vllm(llm, n_tok)
    print(f"  Prefix cached: {n_tok} tokens | {mem_mb:.1f} MB | computed in {elapsed:.3f}s")
    return n_tok, elapsed, mem_mb


def infer_vllm(llm, tokenizer, user_text: str, flush_cache: bool) -> dict:
    if flush_cache:
        llm.reset_prefix_cache()

    out, elapsed = _generate(llm, tokenizer, build_full_prompt(tokenizer, user_text))
    n_in   = len(out.prompt_token_ids)
    cached = out.num_cached_tokens or 0   # vLLM caches whole blocks only

    return {
        "input_tokens":      n_in,
        "output_tokens":     len(out.outputs[0].token_ids),
        "tokens_computed":   n_in - cached,
        "tokens_saved":      cached,
        "time_s":            elapsed,
        "prefill_tok_per_s": (n_in - cached) / elapsed,
        "output":            out.outputs[0].text.strip(),
    }


def run_experiment_vllm(llm, tokenizer):
    df = load_personas(PERSONAS_PATH).sample(frac=1, random_state=SEED).reset_index(drop=True)
    batch_nc = df.iloc[:BATCH_SIZE]
    batch_wc = df.iloc[BATCH_SIZE:2 * BATCH_SIZE]
    print(f"Loaded {len(df)} personas from {os.path.basename(PERSONAS_PATH)}, shuffled (seed={SEED}). "
          f"No-cache batch: {len(batch_nc)} | cached batch: {len(batch_wc)}\n")

    # Warm-up (not recorded), on a persona outside both batches.
    warm = df.iloc[-1]
    infer_vllm(llm, tokenizer, build_user_input(warm["description"], warm["persona"]), flush_cache=True)

    nc_rows = _run_batch("No Cache", batch_nc,
                         lambda text: infer_vllm(llm, tokenizer, text, flush_cache=True))

    print(f"\n{'='*65}")
    print("  Computing prefix KV cache (one-time cost)")
    print("=" * 65)
    prefix_len, prefix_time, kv_mb = compute_prefix_cache_vllm(llm, tokenizer)

    wc_rows = _run_batch("KV Cache", batch_wc,
                         lambda text: infer_vllm(llm, tokenizer, text, flush_cache=False))

    pd.DataFrame(nc_rows + wc_rows).to_csv(RESULTS_PATH, index=False)
    print(f"\nPer-persona results saved to {RESULTS_PATH}")

    _print_comparison(nc_rows, wc_rows, prefix_len, prefix_time, kv_mb)

# ---------------------------------------------------------------------------
# Shared: batch loop, judge-output parsing, comparison report
# ---------------------------------------------------------------------------

def _run_batch(mode, batch, infer_fn):
    print(f"\n{'='*65}")
    print(f"  {mode}: {len(batch)} personas")
    print(f"{'='*65}")
    print(f"{'#':<5} {'jd_id':<10} {'run':<5} {'In tok':<8} {'Computed':<10} {'Out tok':<9} "
          f"{'Time s':<9} {'Verdict'}")
    print("-" * 80)

    rows = []
    for i, (_, p) in enumerate(batch.iterrows()):
        r = infer_fn(build_user_input(p["description"], p["persona"]))
        r.update(mode=mode, jd_id=p["jd_id"], run=p["run"], searched_role=p["searched_role"],
                 **_parse_judgement(r["output"]))
        rows.append(r)
        print(f"{i+1:<5} {p['jd_id']:<10} {p['run']:<5} {r['input_tokens']:<8} "
              f"{r['tokens_computed']:<10} {r['output_tokens']:<9} {r['time_s']:<9.3f} {r['verdict']}")
    return rows


CATEGORIES = ["age", "gender", "race_ethnicity", "lgbtq_heteronormativity", "family_status",
              "disability", "socioeconomic", "appearance", "proximity", "affinity", "education"]


def _load_json(output: str):
    """Outermost {...} as JSON; tolerates invalid escapes copied from JDs (e.g. "10\\+ years")."""
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if not match:
        return None
    for text in (match.group(0), re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", match.group(0))):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return None


def _parse_judgement(output: str) -> dict:
    """Per-category verdicts from the judge's JSON; verdict is "parse_error" if it isn't valid JSON."""
    j = _load_json(output)
    cats = j.get("categories") if isinstance(j, dict) else None
    if isinstance(j, dict) and cats is None and "verdict" in j:   # single-category (Age-only) prompt
        cats = {"age": j}
    if not isinstance(cats, dict):
        return {"verdict": "parse_error", "parse_ok": False}

    row = {"parse_ok": True, "n_categories": sum(c in cats for c in CATEGORIES)}
    bias, mirrored, n_signals, n_grey, n_review = [], [], 0, 0, 0
    for c in CATEGORIES:
        cj = cats.get(c) if isinstance(cats.get(c), dict) else {}
        v = cj.get("verdict")
        row[f"v_{c}"] = v
        bias += [c] if v == "BIAS" else []
        mirrored += [c] if v == "MIRRORED" else []
        n_signals += len(cj.get("signals") or [])
        n_grey += cj.get("grey_area") is True
        n_review += cj.get("needs_human_review") is not None
    row.update(
        verdict="|".join(bias) or "none",
        bias_categories="|".join(bias),
        mirrored_categories="|".join(mirrored),
        n_bias=len(bias), n_signals=n_signals, n_grey_area=n_grey, n_needs_review=n_review,
        overall_summary=j.get("overall_summary"),
    )
    return row


def _print_comparison(nc_rows, wc_rows, prefix_len, prefix_time, kv_mb):
    """Batches contain different JDs, so compare per-call averages and
    per-token latency rather than pairing rows."""
    nc, wc = pd.DataFrame(nc_rows), pd.DataFrame(wc_rows)
    for d in (nc, wc):
        d["ms_per_input_tok"] = d["time_s"] / d["input_tokens"] * 1000

    nc_total = nc["time_s"].sum()
    wc_total = wc["time_s"].sum()
    saved_per_doc = nc["time_s"].mean() - wc["time_s"].mean()
    break_even = prefix_time / saved_per_doc if saved_per_doc > 0 else float("inf")

    def line(label, a, b, fmt):
        print(f"  {label:<28} {a:>14{fmt}} {b:>14{fmt}}")

    print(f"\n{'='*65}")
    print(f"  COMPARISON  (no-cache n={len(nc)}  vs  cached n={len(wc)})")
    print(f"{'='*65}")
    print(f"  {'':<28} {'No Cache':>14} {'KV Cache':>14}")

    print(f"\n  --- Tokens ---")
    line("Avg input tokens",        nc["input_tokens"].mean(),    wc["input_tokens"].mean(),    ".0f")
    line("Avg tokens computed",     nc["tokens_computed"].mean(), wc["tokens_computed"].mean(), ".0f")
    line("Avg tokens saved",        nc["tokens_saved"].mean(),    wc["tokens_saved"].mean(),    ".0f")
    line("Avg output tokens",       nc["output_tokens"].mean(),   wc["output_tokens"].mean(),   ".1f")
    print(f"  Prefix (cached) length       {prefix_len} tokens "
          f"({prefix_len / wc['input_tokens'].mean() * 100:.1f}% of avg cached-batch prompt)")

    print(f"\n  --- Latency ---")
    line("Avg time / call (s)",     nc["time_s"].mean(),           wc["time_s"].mean(),           ".3f")
    line("Median time / call (s)",  nc["time_s"].median(),         wc["time_s"].median(),         ".3f")
    line("Std time / call (s)",     nc["time_s"].std(),            wc["time_s"].std(),            ".3f")
    line("ms per input token",      nc["ms_per_input_tok"].mean(), wc["ms_per_input_tok"].mean(), ".3f")
    line("Total time (s)",          nc_total,                      wc_total + prefix_time,        ".2f")
    print(f"  Prefix compute (one-time)    {prefix_time:.3f}s (included in cached total)")
    print(f"  Speedup avg/call             {nc['time_s'].mean() / wc['time_s'].mean():.2f}x")
    print(f"  Speedup ms/input token       "
          f"{nc['ms_per_input_tok'].mean() / wc['ms_per_input_tok'].mean():.2f}x  (length-normalised)")
    print(f"  Break-even at N docs         {break_even:.1f}")

    print(f"\n  --- Throughput ---")
    line("Avg prefill tok/s",       nc["prefill_tok_per_s"].mean(), wc["prefill_tok_per_s"].mean(), ".0f")

    print(f"\n  --- Memory ---")
    print(f"  KV cache memory (prefix)     {kv_mb:.1f} MB")
    if torch.cuda.is_available():
        print(f"  GPU memory allocated         {torch.cuda.memory_allocated() / 1024**2:.0f} MB")
        print(f"  GPU memory reserved          {torch.cuda.memory_reserved() / 1024**2:.0f} MB")

    print(f"\n  --- Judge output ---")
    for d in (nc, wc):
        if "parse_ok" not in d:
            d["parse_ok"] = False
        d["parse_ok"] = d["parse_ok"].fillna(False).astype(bool)
    pn, pw = nc[nc["parse_ok"]], wc[wc["parse_ok"]]
    line("Parsed OK",               len(pn),                       len(pw),                       "d")
    line("Hit max output tokens",   int((nc["output_tokens"] >= MAX_NEW_TOKENS).sum()),
                                    int((wc["output_tokens"] >= MAX_NEW_TOKENS).sum()),           "d")
    if len(pn) and len(pw):
        line("Avg categories present",  pn["n_categories"].mean(),  pw["n_categories"].mean(),  ".2f")
        line("Avg BIAS categories",     pn["n_bias"].mean(),        pw["n_bias"].mean(),        ".2f")
        line("Avg signals listed",      pn["n_signals"].mean(),     pw["n_signals"].mean(),     ".2f")
        line("Avg grey_area categories", pn["n_grey_area"].mean(),  pw["n_grey_area"].mean(),   ".2f")
        line("Personas w/ human review", int((pn["n_needs_review"] > 0).sum()),
                                         int((pw["n_needs_review"] > 0).sum()),                 "d")

        print(f"\n  --- Personas per verdict, by category (No Cache | KV Cache) ---")
        print(f"  {'':<26} {'BIAS':>11} {'MIRRORED':>11} {'NOT_BIAS':>11} {'NO_SIGNAL':>11}")
        for c in CATEGORIES:
            col = f"v_{c}"
            cells = [f"{int((pn[col] == v).sum()):>4} |{int((pw[col] == v).sum()):>4}"
                     for v in ("BIAS", "MIRRORED", "NOT_BIAS", "NO_SIGNAL")]
            print(f"  {c:<26} " + " ".join(f"{x:>11}" for x in cells))

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def model_slug(model_id: str) -> str:
    # Snapshot paths end in a commit hash; name the file after the model repo instead.
    parts = model_id.rstrip("/").split("/")
    repo = next((p for p in parts if p.startswith("models--")), None)
    return repo.split("--")[-1] if repo else parts[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KV-cache judge experiment (standalone)")
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm",
                        help="vllm: automatic prefix caching (default); hf: transformers + DynamicCache")
    parser.add_argument("--model", default=MODEL_ID, help="judge model: HF id or local snapshot path")
    parser.add_argument("--prompt-path", default=PROMPT_PATH, help="judge system prompt (.jinja2)")
    parser.add_argument("--personas-path", default=PERSONAS_PATH, help="persona CSV (jd_id, run, persona)")
    parser.add_argument("--results-path", default=None,
                        help="output CSV (default: Output_Files/judge_v1_<backend>_<model>.csv)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="personas per batch (no-cache and cached); small values for smoke tests")
    parser.add_argument("--seed", type=int, default=SEED, help="shuffle seed")
    parser.add_argument("--dtype", default="float16", choices=list(DTYPES),
                        help="float16 (default, matches the original runs) | bfloat16 | auto")
    # vLLM only
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="cap context so the KV cache fits (11-category prompts are up to ~17k tokens + output)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                        help="fraction of each GPU vLLM may take (lower it on shared cards)")
    parser.add_argument("--language-model-only", action="store_true",
                        help="skip the vision encoder of multimodal checkpoints (Qwen3.5+)")
    parser.add_argument("--disable-custom-all-reduce", action="store_true",
                        help="use plain NCCL between GPUs (for cards without working PCIe P2P)")
    args = parser.parse_args()

    SYSTEM_PROMPT = load_system_prompt(args.prompt_path)
    PERSONAS_PATH = args.personas_path
    BATCH_SIZE    = args.batch_size
    SEED          = args.seed
    RESULTS_PATH  = args.results_path or os.path.join(
        RESULTS_DIR, f"judge_v1_{args.backend}_{model_slug(args.model)}.csv")

    if args.backend == "vllm":
        llm, tokenizer = load_vllm(args.model, args.tensor_parallel_size, args.dtype,
                                   args.max_model_len, args.language_model_only,
                                   args.disable_custom_all_reduce, args.gpu_memory_utilization)
        run_experiment_vllm(llm, tokenizer)
    else:
        if args.dtype != "float16":
            HF_DTYPE = DTYPES[args.dtype]
        model, tokenizer = load_hf(args.model)
        run_experiment_hf(model, tokenizer)

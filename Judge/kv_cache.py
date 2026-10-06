import argparse
import copy
import os
import json
import re
import time
import torch
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from jinja2.exceptions import TemplateError
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
import pandas as pd

MODEL_ID       = "Qwen/Qwen3-4B"
DATA_PATH = "/home/meghss/Project_Biases_in_Job_Descriptions/CSV_Dataset_Files/Sample_job_descriptions_balanced_300.csv"
PERSONAS_PATH  = "/home/meghss/Project_Biases_in_Job_Descriptions/personas_300/Lama_3.1_8B.csv"
PROMPT_PATH    = "/home/meghss/Project_Biases_in_Job_Descriptions/Judge/judge_prompt_opus_5.5.jinja2"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE          = torch.float16 if torch.cuda.is_available() else torch.float32
MAX_NEW_TOKENS = 1024   # judge returns signals + reasoning JSON; 80 truncated it
# Qwen3 recommended non-thinking sampling settings
TEMPERATURE    = 0.7
TOP_P          = 0.8
TOP_K          = 20
BATCH_SIZE     = 100
SEED           = 42
RESULTS_PATH   = "/home/meghss/Project_Biases_in_Job_Descriptions/Output_Files/kv_cache_comparison.csv"

def load_system_prompt(path: str = PROMPT_PATH) -> str:
    env = Environment(loader=FileSystemLoader(os.path.dirname(path)), undefined=StrictUndefined)
    return env.get_template(os.path.basename(path)).render().strip()


SYSTEM_PROMPT = load_system_prompt()


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
    

def load(model_id: str = MODEL_ID):
    print(f"Loading {model_id} on {DEVICE} ({DTYPE})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        device_map="auto",
    )
    model.eval()
    print(f"Loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B\n")
    return model, tokenizer


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

def compute_prefix_cache(model, tokenizer) -> tuple:

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
    mem_mb  = _kv_memory_mb(kv)
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


def _kv_memory_mb(kv) -> float:
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

# ---------------------------------------------------------------------------
# Single inference — no cache
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Single inference — with KV cache
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Experiment: shuffle personas, batch 1 (first 100) no cache, batch 2 (next 100) cached
# ---------------------------------------------------------------------------

def run_experiment(model, tokenizer):
    df = load_personas(PERSONAS_PATH).sample(frac=1, random_state=SEED).reset_index(drop=True)
    batch_nc = df.iloc[:BATCH_SIZE]
    batch_wc = df.iloc[BATCH_SIZE:2 * BATCH_SIZE]
    print(f"Loaded {len(df)} personas from {os.path.basename(PERSONAS_PATH)}, shuffled (seed={SEED}). "
          f"No-cache batch: {len(batch_nc)} | cached batch: {len(batch_wc)}\n")

    print("=" * 65)
    print("  Computing prefix KV cache (one-time cost)")
    print("=" * 65)
    prefix_kv, prefix_len, prefix_time, kv_mb = compute_prefix_cache(model, tokenizer)

    # Warm-up (not recorded) so CUDA init overhead doesn't penalise batch 1.
    # Uses a persona outside both batches.
    warm = df.iloc[-1]
    infer_no_cache(model, tokenizer, build_user_input(warm["description"], warm["persona"]))

    nc_rows = _run_batch("No Cache", batch_nc,
                         lambda text: infer_no_cache(model, tokenizer, text))
    wc_rows = _run_batch("KV Cache", batch_wc,
                         lambda text: infer_with_cache(model, tokenizer, text,
                                                       prefix_kv, prefix_len))

    results = pd.DataFrame(nc_rows + wc_rows)
    results.to_csv(RESULTS_PATH, index=False)
    print(f"\nPer-persona results saved to {RESULTS_PATH}")

    _print_comparison(nc_rows, wc_rows, prefix_len, prefix_time, kv_mb)


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


def _parse_judgement(output: str) -> dict:
    """Pull the judge's JSON fields; verdict is "parse_error" if it isn't valid JSON."""
    match = re.search(r"\{.*\}", output, re.DOTALL)
    try:
        j = json.loads(match.group(0))
        return {
            "verdict":            j["verdict"],
            "confidence":         j.get("confidence"),
            "grey_area":          j.get("grey_area"),
            "needs_human_review": j.get("needs_human_review"),
            "jd_has_signal":      j.get("jd_has_signal"),
            "n_signals":          len(j.get("signals") or []),
        }
    except (AttributeError, json.JSONDecodeError, KeyError, TypeError):
        return {"verdict": "parse_error", "confidence": None, "grey_area": None,
                "needs_human_review": None, "jd_has_signal": None, "n_signals": None}


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

    print(f"\n  --- Verdict distribution ---")
    verdicts = ["BIAS", "MIRRORED", "NOT_BIAS", "NO_SIGNAL", "parse_error"]
    nc_counts, wc_counts = nc["verdict"].value_counts(), wc["verdict"].value_counts()
    for v in verdicts:
        line(v, nc_counts.get(v, 0), wc_counts.get(v, 0), "d")

    print(f"\n  --- Judge flags ---")
    line("Avg signals listed",      nc["n_signals"].mean(),          wc["n_signals"].mean(),          ".2f")
    for c in ("high", "medium", "low"):
        line(f"confidence = {c}", int((nc["confidence"] == c).sum()), int((wc["confidence"] == c).sum()), "d")
    line("grey_area = true",        int((nc["grey_area"] == True).sum()), int((wc["grey_area"] == True).sum()), "d")
    line("needs_human_review set",  int(nc["needs_human_review"].notna().sum()), int(wc["needs_human_review"].notna().sum()), "d")
    line("jd_has_signal = true",    int((nc["jd_has_signal"] == True).sum()), int((wc["jd_has_signal"] == True).sum()), "d")


def _print_summary(nc_rows, wc_rows, prefix_len, prefix_time, kv_mb):

    def mean(lst, key): return sum(r[key] for r in lst) / len(lst)

    n = len(nc_rows)

    nc_total_tok   = sum(r["input_tokens"]    for r in nc_rows)
    wc_total_tok   = sum(r["tokens_computed"] for r in wc_rows)
    nc_total_time  = sum(r["time_s"]          for r in nc_rows)
    wc_total_time  = sum(r["time_s"]          for r in wc_rows)
    wc_full_time   = wc_total_time + prefix_time  # include one-time cost

    tok_saved      = nc_total_tok - wc_total_tok
    tok_saved_pct  = tok_saved / nc_total_tok * 100
    speedup_inf    = nc_total_time / wc_total_time
    speedup_full   = nc_total_time / wc_full_time

    # break-even: how many docs before cached total time < no-cache total time
    time_per_nc    = mean(nc_rows, "time_s")
    time_per_wc    = mean(wc_rows, "time_s")
    time_saved_doc = time_per_nc - time_per_wc
    break_even     = prefix_time / time_saved_doc if time_saved_doc > 0 else float("inf")

    mismatches     = sum(1 for nc, wc in zip(nc_rows, wc_rows)
                         if nc["output"].strip() != wc["output"].strip())

    print(f"\n{'='*65}")
    print(f"  EFFICIENCY SUMMARY  ({n} runs)")
    print(f"{'='*65}")

    print(f"\n  {'--- Token efficiency ---'}")
    print(f"  Prefix (cached) length      : {prefix_len} tokens")
    print(f"  Avg input tokens (no-cache) : {mean(nc_rows, 'input_tokens'):.0f}")
    print(f"  Avg tokens computed (cached): {mean(wc_rows, 'tokens_computed'):.0f}  "
          f"(user turn only)")
    print(f"  Total tokens saved          : {tok_saved:,}  ({tok_saved_pct:.1f}%)")

    print(f"\n  {'--- Latency ---'}")
    print(f"  Avg time / call (no-cache)  : {mean(nc_rows, 'time_s'):.3f}s")
    print(f"  Avg time / call (cached)    : {mean(wc_rows, 'time_s'):.3f}s")
    print(f"  Prefix compute (one-time)   : {prefix_time:.3f}s")
    print(f"  Total time no-cache         : {nc_total_time:.3f}s")
    print(f"  Total time cached (w/ pfx)  : {wc_full_time:.3f}s")
    print(f"  Speedup (inference only)    : {speedup_inf:.2f}x")
    print(f"  Speedup (incl. prefix cost) : {speedup_full:.2f}x")
    print(f"  Break-even at N docs        : {break_even:.1f}")

    print(f"\n  {'--- Throughput ---'}")
    print(f"  Avg prefill tok/s (no-cache): {mean(nc_rows, 'prefill_tok_per_s'):.0f}")
    print(f"  Avg prefill tok/s (cached)  : {mean(wc_rows, 'prefill_tok_per_s'):.0f}  "
          f"(fewer tokens, so raw number is lower — this is expected)")

    print(f"\n  {'--- Memory ---'}")
    print(f"  KV cache memory (prefix)    : {kv_mb:.1f} MB")
    if torch.cuda.is_available():
        print(f"  GPU memory allocated        : "
              f"{torch.cuda.memory_allocated() / 1024**2:.0f} MB")
        print(f"  GPU memory reserved         : "
              f"{torch.cuda.memory_reserved() / 1024**2:.0f} MB")

    print(f"\n  {'--- Correctness ---'}")
    print(f"  Output mismatches           : {mismatches} / {n}  "
          f"({'✓ all match' if mismatches == 0 else '✗ check clone logic'})")

    print(f"\n  {'--- Per-run detail ---'}")
    print(f"  {'Run':<5} {'NoCache s':<12} {'Cached s':<12} "
          f"{'Tok saved':<12} {'Tok saved %'}")
    print(f"  {'-'*55}")
    for i, (nc, wc) in enumerate(zip(nc_rows, wc_rows)):
        saved     = nc["input_tokens"] - wc["tokens_computed"]
        saved_pct = saved / nc["input_tokens"] * 100
        print(f"  {i+1:<5} {nc['time_s']:<12.3f} {wc['time_s']:<12.3f} "
              f"{saved:<12} {saved_pct:.1f}%")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def results_path_for(model_id: str, stem: str) -> str:
    slug = model_id.rstrip("/").split("/")[-1]
    return os.path.join(os.path.dirname(RESULTS_PATH), f"{stem}_{slug}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KV-cache judge experiment (HF transformers)")
    parser.add_argument("--model", default=MODEL_ID, help="judge model: HF id or local path")
    parser.add_argument("--results-path", default=None,
                        help="output CSV (default: Output_Files/kv_cache_comparison_<model>.csv)")
    args = parser.parse_args()
    RESULTS_PATH = args.results_path or results_path_for(args.model, "kv_cache_comparison")

    model, tokenizer = load(args.model)
    run_experiment(model, tokenizer)

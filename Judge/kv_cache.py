import copy
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

MODEL_ID       = "Qwen/Qwen3-4B"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE          = torch.float16 if torch.cuda.is_available() else torch.float32
MAX_NEW_TOKENS = 80
N_RUNS         = 10


SYSTEM_PROMPT = """You are a bias detection classifier for job descriptions.

TASK:
Classify the input text into exactly one of three verdicts:
  - flag        : text contains biased language targeting a protected trait
  - pass        : text is neutral and job-relevant
  - needs_review: text is ambiguous or falls into a grey area

BIAS CATEGORY: Age
PATTERN: Grounded-requirement

TRIGGER SIGNALS:
  - Explicit age range (e.g. "25-35 years old")
  - Generational labels: "digital native", "millennial", "Gen Z", "boomer"
  - Youth proxies: "young", "energetic", "dynamic", "recent graduate", "early career"
  - Unqualified stamina language: "high energy", "hustle culture", "fast-paced mindset"
  - Open-ended experience caps used as age proxies: "no more than 5 years experience"

NOT A BIAS (pass automatically):
  - Experience requirements grounded in seniority (e.g. "10+ years for a Director role")
  - Physical stamina tied explicitly to task demands (e.g. "able to lift 30kg for warehouse role")
  - "fast-paced environment" describing the workplace, not the candidate

GREY AREA (needs_review):
  - "Energetic" or "dynamic" without a task-grounded reason
  - Hustle language that could target youth or simply describe culture
  - Experience caps without clear role justification

DECISION RULE:
1. If an explicit age/generation label is present → flag immediately
2. If a youth-proxy word is present AND no task reason is given → flag
3. If a youth-proxy word is present WITH a task reason → needs_review
4. If only experience requirements grounded in seniority → pass
5. If ambiguous with no clear anchor → needs_review

OUTPUT FORMAT (respond with JSON only, no other text):
{
  "verdict": "flag" | "pass" | "needs_review",
  "evidence": "<short quote from input that drove the decision>",
  "rule_applied": "<which rule number from the decision rule above>"
}"""

JD_SNIPPETS = [
    "We are looking for a dynamic, energetic team player to join our startup.",
    "The role requires 12+ years of experience for this VP-level position.",
    "Seeking a recent graduate who is a digital native and loves hustle culture.",
    "Must be able to lift 30kg repeatedly as part of daily warehouse operations.",
    "We want someone young and hungry who will grow with us for the long term.",
    "Director of Engineering: 15 years minimum experience in distributed systems.",
    "Looking for a high-energy millennial or Gen Z candidate for our social team.",
    "Fast-paced environment — no more than 3 years experience preferred.",
    "Senior Analyst: 8+ years in financial modeling, CFA preferred.",
    "Join our team! We value fresh perspectives and a youthful approach to problems.",
]


def load():
    print(f"Loading {MODEL_ID} on {DEVICE} ({DTYPE})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto",
    )
    model.eval()
    print(f"Loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B\n")
    return model, tokenizer


def build_full_prompt(tokenizer, user_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Classify this job description snippet:\n\n{user_text}"},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  
    )


def build_prefix_only(tokenizer) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

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
            do_sample=False,
            temperature=None,
            top_p=None,
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
            do_sample=False,
            temperature=None,
            top_p=None,
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
# 10-run experiment
# ---------------------------------------------------------------------------

def run_experiment(model, tokenizer):
    print("=" * 65)
    print("  Computing prefix KV cache (one-time cost)")
    print("=" * 65)
    prefix_kv, prefix_len, prefix_time, kv_mb = compute_prefix_cache(model, tokenizer)

    no_cache_rows   = []
    with_cache_rows = []

    print(f"\n{'='*65}")
    print(f"  Running {N_RUNS} inference calls — No Cache vs. KV Cache")
    print(f"{'='*65}")
    print(f"\n{'Run':<5} {'Snippet (first 45 chars)':<46} "
          f"{'NoCache tok':<13} {'NoCache s':<11} "
          f"{'KVCache tok':<13} {'KVCache s':<11} {'Match'}")
    print("-" * 115)

    for run_idx in range(N_RUNS):
        text = JD_SNIPPETS[run_idx % len(JD_SNIPPETS)]

        nc = infer_no_cache(model, tokenizer, text)
        wc = infer_with_cache(model, tokenizer, text, prefix_kv, prefix_len)

        match = "✓" if nc["output"].strip() == wc["output"].strip() else "✗ MISMATCH"

        print(f"{run_idx+1:<5} {text[:45]:<46} "
              f"{nc['input_tokens']:<13} {nc['time_s']:<11.3f} "
              f"{wc['tokens_computed']:<13} {wc['time_s']:<11.3f} {match}")

        no_cache_rows.append(nc)
        with_cache_rows.append(wc)

    _print_summary(no_cache_rows, with_cache_rows,
                   prefix_len, prefix_time, kv_mb)


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

if __name__ == "__main__":
    model, tokenizer = load()
    run_experiment(model, tokenizer)
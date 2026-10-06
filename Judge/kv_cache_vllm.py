import os
# Keep the engine in this process so reset_prefix_cache() and the torch
# memory stats in _print_comparison see the real engine.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import argparse
import time
import pandas as pd
from vllm import LLM, SamplingParams

from kv_cache import (
    MODEL_ID, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, TOP_K, BATCH_SIZE, SEED,
    PERSONAS_PATH, load_personas, build_user_input, build_full_prompt, build_prefix_only,
    _run_batch, _print_comparison, results_path_for,
)

RESULTS_PATH = None   # set from --results-path / --model in __main__

SAMPLING = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
                          max_tokens=MAX_NEW_TOKENS)


def load(model_id: str = MODEL_ID, tensor_parallel_size: int = 1, dtype: str = "float16",
         max_model_len: int | None = None, language_model_only: bool = False):
    print(f"Loading {model_id} with vLLM (tp={tensor_parallel_size}, {dtype})...")
    # Prefix caching is on; the no-cache batch flushes it before every call.
    llm = LLM(
        model=model_id,
        dtype=dtype,                   # float16 default matches the HF experiment
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        language_model_only=language_model_only,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
    )
    print("Loaded.\n")
    return llm, llm.get_tokenizer()


def _tokens(tokenizer, text: str) -> list:
    # Chat templates already add BOS (e.g. Llama's <|begin_of_text|>); don't add a second.
    return tokenizer(text, add_special_tokens=False).input_ids


def _generate(llm, tokenizer, prompt: str):
    t0 = time.perf_counter()
    out = llm.generate([{"prompt_token_ids": _tokens(tokenizer, prompt)}], SAMPLING, use_tqdm=False)[0]
    return out, time.perf_counter() - t0


def _kv_memory_mb(llm, n_tokens: int) -> float:
    # Attention KV only; hybrid models (Qwen3.5+) also keep a fixed-size linear-attention state.
    cfg = llm.llm_engine.model_config.hf_text_config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    layer_types = getattr(cfg, "layer_types", None)
    n_attn = layer_types.count("full_attention") if layer_types else cfg.num_hidden_layers
    per_token = 2 * n_attn * cfg.num_key_value_heads * head_dim * 2  # K+V, 2 bytes
    return per_token * n_tokens / 1024 ** 2


def compute_prefix_cache(llm, tokenizer) -> tuple:
    """Prime vLLM's prefix cache with the system prompt (one-time cost)."""
    llm.reset_prefix_cache()
    prefix_text = build_prefix_only(tokenizer)
    prefix_ids  = _tokens(tokenizer, prefix_text)
    n_tok = len(prefix_ids)

    t0 = time.perf_counter()
    llm.generate([{"prompt_token_ids": prefix_ids}], SamplingParams(max_tokens=1), use_tqdm=False)
    elapsed = time.perf_counter() - t0

    mem_mb = _kv_memory_mb(llm, n_tok)
    print(f"  Prefix cached: {n_tok} tokens | {mem_mb:.1f} MB | computed in {elapsed:.3f}s")
    return n_tok, elapsed, mem_mb


def infer(llm, tokenizer, user_text: str, flush_cache: bool) -> dict:
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


def run_experiment(llm, tokenizer):
    df = load_personas(PERSONAS_PATH).sample(frac=1, random_state=SEED).reset_index(drop=True)
    batch_nc = df.iloc[:BATCH_SIZE]
    batch_wc = df.iloc[BATCH_SIZE:2 * BATCH_SIZE]
    print(f"Loaded {len(df)} personas from {os.path.basename(PERSONAS_PATH)}, shuffled (seed={SEED}). "
          f"No-cache batch: {len(batch_nc)} | cached batch: {len(batch_wc)}\n")

    # Warm-up (not recorded), on a persona outside both batches.
    warm = df.iloc[-1]
    infer(llm, tokenizer, build_user_input(warm["description"], warm["persona"]), flush_cache=True)

    nc_rows = _run_batch("No Cache", batch_nc,
                         lambda text: infer(llm, tokenizer, text, flush_cache=True))

    print(f"\n{'='*65}")
    print("  Computing prefix KV cache (one-time cost)")
    print("=" * 65)
    prefix_len, prefix_time, kv_mb = compute_prefix_cache(llm, tokenizer)

    wc_rows = _run_batch("KV Cache", batch_wc,
                         lambda text: infer(llm, tokenizer, text, flush_cache=False))

    pd.DataFrame(nc_rows + wc_rows).to_csv(RESULTS_PATH, index=False)
    print(f"\nPer-persona results saved to {RESULTS_PATH}")

    _print_comparison(nc_rows, wc_rows, prefix_len, prefix_time, kv_mb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KV-cache judge experiment (vLLM prefix caching)")
    parser.add_argument("--model", default=MODEL_ID, help="judge model: HF id or local path")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="float16", help="float16 | bfloat16 | auto")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="cap context so the KV cache fits (prompts here are <7k tokens)")
    parser.add_argument("--language-model-only", action="store_true",
                        help="skip the vision encoder of multimodal checkpoints (Qwen3.5+)")
    parser.add_argument("--results-path", default=None,
                        help="output CSV (default: Output_Files/kv_cache_vllm_<model>.csv)")
    args = parser.parse_args()
    RESULTS_PATH = args.results_path or results_path_for(args.model, "kv_cache_vllm")

    llm, tokenizer = load(args.model, args.tensor_parallel_size, args.dtype,
                          args.max_model_len, args.language_model_only)
    run_experiment(llm, tokenizer)

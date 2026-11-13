"""OLMoTrace-style tracing of individual OLMo personas.

For each sampled persona, find token spans that occur verbatim in
OLMo-2-13B-Instruct's training data (the OLMoTrace matching step, probed every
STRIDE tokens to fit the public API's rate limit), keep the longest ones, and retrieve example documents for each, labelled as pretraining
(olmo-mix-1124) or later-stage (shard 11: mid/post-training) data.

Each persona's result is appended to results/span_trace.jsonl, and personas
already there are skipped, so the script can be re-run after interruption.

Usage: python olmotrace/span_trace.py --n 24
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infinigram import Client, PRE_TOKENS, POST_TOKENS, sample_docs

REPO_ROOT = Path(__file__).resolve().parent.parent
PERSONAS = REPO_ROOT / "personas_300" / "olmo_personas.csv"
OUT_PATH = Path(__file__).resolve().parent / "results" / "span_trace.jsonl"

WINDOW = 64        # longest span we can detect, in tokens
MIN_SPAN = 6       # ignore matches shorter than this
STRIDE = 4         # probe every 4th end position; the API allows ~50 requests/min
SPANS_PER_PERSONA = 6
NEWLINE_ID = 13    # Llama-2 "<0x0A>", appended as a dummy final token


def match_length(client, ids, j):
    """Length of the longest span ending at token j that occurs in the corpus."""
    window = ids[max(0, j + 1 - WINDOW): j + 1] + [NEWLINE_ID]
    return client.query(query_type="infgram_prob", query_ids=window)["suffix_len"]


def probe_spans(client, ids):
    """Matched spans (start, end) ending at every STRIDE-th token, minus contained ones.

    With MIN_SPAN >= STRIDE every match of MIN_SPAN + STRIDE tokens or more is
    seen, though its end may be up to STRIDE - 1 tokens short (see extend).
    """
    ends = list(range(STRIDE - 1, len(ids), STRIDE))
    if ends[-1] != len(ids) - 1:
        ends.append(len(ids) - 1)
    spans = []
    for j in ends:
        n = match_length(client, ids, j)
        if n >= MIN_SPAN:
            spans.append((j + 1 - n, j + 1))
    return [s for s in spans if not any(o != s and o[0] <= s[0] and s[1] <= o[1] for o in spans)]


def extend(client, ids, span):
    """Push a span's end forward while the match keeps growing token by token."""
    a, b = span
    while b < len(ids) and match_length(client, ids, b) == b + 1 - a:
        b += 1
    return a, b


def detok(client, ids):
    return client.query(query_type="count", query_ids=ids)["tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    personas = pd.read_csv(PERSONAS)
    personas = personas[personas.success == True]
    sample = personas.sample(n=args.n, random_state=args.seed)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            done = {(r["jd_id"], r["run"]) for r in map(json.loads, f)}

    client = Client()
    for k, row in enumerate(sample.itertuples()):
        if (row.jd_id, row.run) in done:
            continue
        print(f"[{k + 1}/{len(sample)}] {row.jd_id} run {row.run} ({row.searched_role})", flush=True)
        ids = client.tokenize(row.persona)
        spans = probe_spans(client, ids)
        covered = set()
        for a, b in spans:
            covered.update(range(a, b))
        spans.sort(key=lambda s: s[1] - s[0], reverse=True)
        spans = [extend(client, ids, s) for s in spans[:SPANS_PER_PERSONA]]

        traced = []
        for a, b in spans:
            span_ids = ids[a:b]
            pre, post, segs = client.find(query_ids=span_ids)
            traced.append({
                "text": "".join(detok(client, span_ids)).replace("▁", " ").strip(),
                "n_tokens": b - a,
                "pretrain_count": pre,
                "posttrain_count": post,
                "posttrain_enrichment": (post / POST_TOKENS) / (pre / PRE_TOKENS) if pre else None,
                "docs": sample_docs(client, span_ids, segs, n_pre=2, n_post=2, max_disp_len=120),
            })

        with open(OUT_PATH, "a") as f:
            f.write(json.dumps({
                "jd_id": row.jd_id,
                "run": row.run,
                "role": row.searched_role,
                "n_tokens": len(ids),
                "coverage": len(covered) / len(ids),
                "max_span": max((b - a for a, b in spans), default=0),
                "spans": traced,
            }) + "\n")


if __name__ == "__main__":
    main()

"""Where do OLMo's favourite persona names appear in character-introduction contexts?

A bare first name like "Elena" mostly matches real people on the web. Here we
count names in the contexts personas use ("Meet Elena", "named Elena") and,
within each context, compare each name's share of mentions in pretraining vs.
later training stages, with example documents for each.

Usage: python olmotrace/name_contexts.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infinigram import Client, sample_docs

OUT_DIR = Path(__file__).resolve().parent / "results"
# Separate cache so this can run beside the other tracing scripts
CACHE = Path(__file__).resolve().parent / "cache" / "name_contexts_cache.jsonl"

NAMES = {
    "olmo": ["Elena", "Ava", "Evelyn", "Alexis", "Amelia", "Amara", "Elara"],
    "control": ["Sarah", "Jennifer", "Emily", "Maria", "Michael", "David"],
}
CONTEXTS = ["Meet {n}", "named {n}", "{n} grew up", "{n} is a"]
FULL_NAMES = ["Elena Rodriguez", "Elena Martinez", "Elena Rivera", "Elena Vasquez", "Sarah Johnson", "Emily Johnson"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = Client(cache_path=CACHE)
    rows, examples = [], {}
    queries = [(c, n, g) for c in CONTEXTS for g, ns in NAMES.items() for n in ns]
    queries += [("Meet {n}", n, "full name") for n in FULL_NAMES]
    for context, name, group in queries:
        text = context.format(n=name)
        print(text, flush=True)
        ids = client.tokenize(text)
        pre, post, segs = client.find(query_ids=ids)
        rows.append({"context": context, "name": name, "group": group, "query": text,
                     "pretrain_count": pre, "posttrain_count": post})
        if group != "control" and context in ("Meet {n}", "named {n}"):
            examples[text] = sample_docs(client, ids, segs, n_pre=3, n_post=4, max_disp_len=160)

    df = pd.DataFrame(rows)
    single = df[df.group != "full name"]
    for stage in ("pretrain", "posttrain"):
        tot = single.groupby("context")[f"{stage}_count"].transform("sum")
        df.loc[single.index, f"{stage}_share"] = single[f"{stage}_count"] / tot
    df.to_csv(OUT_DIR / "name_contexts.csv", index=False)
    with open(OUT_DIR / "name_context_examples.json", "w") as f:
        json.dump(examples, f, indent=1)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

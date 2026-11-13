"""Which later-stage datasets carry OLMo's favourite persona names?

For each full name, sample up to 20 later-stage (shard 11) documents and tally
their source dataset (Tulu 3 SFT, the OLMo 2 preference mix, Dolmino
mid-training, and within Dolmino the PersonaHub-derived synthetic math), plus
the count of the "Dr. {name}" persona-problem opener.

Usage: python olmotrace/persona_data_sources.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infinigram import Client, POST_SHARD

OUT_DIR = Path(__file__).resolve().parent / "results"
CACHE = Path(__file__).resolve().parent / "cache" / "name_contexts_cache.jsonl"

# OLMo's most common persona names (Persona Pattern Audit), then controls
NAMES = ["Emily Carter", "Emily Tran", "Evelyn Carter", "Elena Martinez", "Alexis Rivera",
         "Evelyn Tran", "Emily Chen", "Elena Rivera", "Elena Vasquez", "Elena Rodriguez",
         "Sarah Johnson", "Jennifer Smith"]
N_DOCS = 20


def dataset_of(doc):
    path = doc["path"]
    if "personahub" in path:
        return "PersonaHub synthetic (mid-training)"
    if doc["source"] == "dolmino":
        return "Dolmino mid-training (other)"
    if doc["source"].startswith("tulu-3-sft"):
        return "Tulu 3 SFT"
    if "preference" in doc["source"]:
        return "OLMo 2 preference (DPO)"
    return doc["source"]


def main():
    client = Client(cache_path=CACHE)
    rows = []
    for name in NAMES:
        print(name, flush=True)
        ids = client.tokenize(name)
        pre, post, segs = client.find(query_ids=ids)
        a, b = segs[POST_SHARD]
        m = min(N_DOCS, b - a)
        docs = [client.doc(POST_SHARD, a + int((k + 0.5) * (b - a) / m), ids, 60) for k in range(m)]
        tally = Counter(dataset_of(d) for d in docs)
        dr_pre, dr_post, _ = client.find(query=f"Dr. {name}")
        rows.append({"name": name, "pretrain_count": pre, "posttrain_count": post,
                     "dr_pretrain_count": dr_pre, "dr_posttrain_count": dr_post,
                     "n_sampled": m, **{f"sampled: {k}": v for k, v in tally.items()}})
    df = pd.DataFrame(rows).fillna(0)
    df.to_csv(OUT_DIR / "persona_data_sources.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

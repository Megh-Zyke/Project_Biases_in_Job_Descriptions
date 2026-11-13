"""Do the invented experts in OLMo's later-stage data skew female?

Counts "Dr. {first name}" for common women's and men's first names (plus
OLMo's default names) in pretraining vs. the later training stages.

Usage: python olmotrace/invented_expert_gender.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infinigram import Client

OUT_DIR = Path(__file__).resolve().parent / "results"
CACHE = Path(__file__).resolve().parent / "cache" / "audit_cache.jsonl"

NAMES = {
    "female": ["Emily", "Sarah", "Elena", "Evelyn", "Jennifer", "Maria", "Lisa", "Amelia"],
    "male": ["Michael", "David", "James", "John", "Robert", "Alex", "Mark", "Daniel"],
}


def main():
    client = Client(cache_path=CACHE)
    rows = []
    for gender, names in NAMES.items():
        for name in names:
            pre, post, _ = client.find(query=f"Dr. {name}")
            rows.append({"gender": gender, "name": name, "pretrain_count": pre, "posttrain_count": post})
            print(gender, name, pre, post, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "invented_expert_gender.csv", index=False)
    tot = df.groupby("gender")[["pretrain_count", "posttrain_count"]].sum()
    print(tot)
    print("female share: pretrain", round(tot.pretrain_count.female / tot.pretrain_count.sum(), 3),
          "later", round(tot.posttrain_count.female / tot.posttrain_count.sum(), 3))


if __name__ == "__main__":
    main()

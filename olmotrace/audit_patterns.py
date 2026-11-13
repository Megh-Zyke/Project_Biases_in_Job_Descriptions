"""Trace the OLMo patterns reported in the Persona Pattern Audit to OLMo's training data.

Each pattern is tied to the audit card it comes from. For each: share of OLMo
personas using it (personas with anonymised names excluded, as in the audit),
counts in pretraining vs. later training stages, per-token enrichment of the
later stages, the later-stage source datasets of sampled documents, and example
documents. Also counts "N-year-old" for every age, to compare the audit's age
heaping with the training data.

Usage: python olmotrace/audit_patterns.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infinigram import Client, PRE_TOKENS, POST_TOKENS, sample_docs

REPO_ROOT = Path(__file__).resolve().parent.parent
PERSONAS = REPO_ROOT / "personas_300" / "olmo_personas.csv"
OUT_DIR = Path(__file__).resolve().parent / "results"
CACHE = Path(__file__).resolve().parent / "cache" / "audit_cache.jsonl"

# (audit card, pattern, is_control)
PATTERNS = [
    ("Default candidate", "Emily Carter", False),
    ("Default candidate", "Emily Tran", False),
    ("Default candidate", "Evelyn Carter", False),
    ("Default candidate", "Elena Martinez", False),
    ("Default candidate", "Alexis Rivera", False),
    ("Default candidate", "Emily Johnson", True),
    ("Default candidate", "Jennifer Smith", True),
    ("Asian surname, Anglo first name", "Emily Chen", False),
    ("Asian surname, Anglo first name", "Emily Patel", False),
    ("Asian surname, Anglo first name", "Mei Chen", True),
    ("Asian surname, Anglo first name", "Priya Patel", True),
    ("Asian surname, Anglo first name", "Linh Tran", True),
    ("Anonymised names", "<PRESIDIO_ANONYMIZED_PERSON>", False),
    ("Opposite-sex spouse, two children", "her husband and two children", False),
    ("Opposite-sex spouse, two children", "his wife and two children", False),
    ("Opposite-sex spouse, two children", "her husband", False),
    ("Opposite-sex spouse, two children", "her wife", True),
    ("Opposite-sex spouse, two children", "his wife", False),
    ("Opposite-sex spouse, two children", "his husband", True),
    ("Stock phrases", "Meet ", False),
    ("Stock phrases", "a passion for", False),
    ("Stock phrases", "meticulous", False),
    ("Stock phrases", "thrives", False),
    ("Stock phrases", "known for her", False),
    ("Stock phrases", "a stone's throw from", False),
    ("Stock phrases", "hailing from", False),
    ("Stock phrases", "the vibrant city of", False),
    ("Stock phrases", "a testament to her", False),
    ("Lifestyle", "cozy apartment", False),
    ("Lifestyle", "loft apartment", False),
    ("Home city", "Austin, Texas", False),
    ("Home city", "San Francisco", False),
    ("Prestige training", "classically trained in French", False),
]
AGES = range(22, 56)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    personas = pd.read_csv(PERSONAS)
    personas = personas[personas.success == True].persona.astype(str)
    clean = personas[~personas.str.contains("<PRESIDIO", regex=False)]
    client = Client(cache_path=CACHE)

    rows, examples = [], {}
    for card, pattern, control in PATTERNS:
        print(f"{card:36s} {pattern}", flush=True)
        ids = client.tokenize(pattern.strip())
        pre, post, segs = client.find(query_ids=ids)
        docs = sample_docs(client, ids, segs, n_pre=3, n_post=6)
        pool = personas if pattern.startswith("<") else clean
        rate_pre, rate_post = pre / PRE_TOKENS * 1e9, post / POST_TOKENS * 1e9
        rows.append({
            "card": card, "pattern": pattern, "control": control,
            "persona_share": pool.str.contains(pattern, regex=False).mean(),
            "pretrain_count": pre, "posttrain_count": post,
            "pretrain_per_B_tokens": rate_pre, "posttrain_per_B_tokens": rate_post,
            "posttrain_enrichment": rate_post / rate_pre if rate_pre else None,
            "posttrain_sources": dict(Counter(d["source"] for d in docs if d["stage"] == "post")),
        })
        examples[pattern] = docs

    ages = []
    persona_ages = clean.str.extract(r"\b(\d{2})-year-old")[0].dropna().astype(int)
    for age in AGES:
        pre, post, _ = client.find(query=f"a {age}-year-old")
        ages.append({"age": age, "pretrain_count": pre, "posttrain_count": post,
                     "persona_count": int((persona_ages == age).sum())})
        print(f"age {age}: {pre} / {post}", flush=True)

    pd.DataFrame(rows).to_csv(OUT_DIR / "audit_patterns.csv", index=False)
    pd.DataFrame(ages).to_csv(OUT_DIR / "audit_ages.csv", index=False)
    with open(OUT_DIR / "audit_examples.json", "w") as f:
        json.dump(examples, f, indent=1)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(pd.DataFrame(ages).to_string(index=False))


if __name__ == "__main__":
    main()

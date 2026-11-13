"""Where do recurring OLMo persona patterns occur in OLMo-2-13B-Instruct's training data?

For each pattern: how often OLMo personas use it, how often it occurs in
pretraining (olmo-mix-1124) vs. the later training stages (shard 11), the
per-token enrichment of the later stages over pretraining, and a few example
documents from each stage.

Usage: python olmotrace/pattern_provenance.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infinigram import Client, PRE_TOKENS, POST_TOKENS, sample_docs

REPO_ROOT = Path(__file__).resolve().parent.parent
PERSONAS = REPO_ROOT / "personas_300" / "olmo_personas.csv"
OUT_DIR = Path(__file__).resolve().parent / "results"

# (group, pattern). "control" rows are common names/places OLMo rarely uses.
PATTERNS = [
    ("artifact", "<PRESIDIO_ANONYMIZED_PERSON>"),
    ("artifact", "<PRESIDIO_ANONYMIZED_EMAIL_ADDRESS>"),
    ("first name", "Elena"),
    ("first name", "Evelyn"),
    ("first name", "Ava"),
    ("first name", "Alexis"),
    ("first name", "Amelia"),
    ("first name", "Amara"),
    ("first name", "Elara"),
    ("first name (control)", "Jennifer"),
    ("first name (control)", "Michael"),
    ("first name (control)", "Sarah"),
    ("first name (control)", "David"),
    ("surname", "Rivera"),
    ("surname", "Martinez"),
    ("surname", "Carter"),
    ("surname", "Tran"),
    ("surname", "Vasquez"),
    ("surname (control)", "Smith"),
    ("surname (control)", "Johnson"),
    ("full name", "Elena Martinez"),
    ("full name", "Elena Rivera"),
    ("full name", "Elena Vasquez"),
    ("full name", "Alexis Rivera"),
    ("full name", "Elena Rodriguez"),
    ("full name", "Amelia Rivera"),
    ("full name", "Ava Tran"),
    ("full name", "Evelyn Carter"),
    ("full name", "Amara Patel"),
    ("full name", "Liam O'Connor"),
    ("full name (control)", "Elara Voss"),
    ("full name (control)", "Sarah Johnson"),
    ("full name (control)", "Emily Johnson"),
    ("full name (control)", "John Smith"),
    ("full name (control)", "Jennifer Smith"),
    ("place", "Austin, Texas"),
    ("place", "San Jose, California"),
    ("place", "Seattle, Washington"),
    ("place", "Bethesda, Maryland"),
    ("place", "San Juan, Puerto Rico"),
    ("place (control)", "New York City"),
    ("place (control)", "Los Angeles, California"),
    ("phrasing", "Meet **"),
    ("phrasing", "hailing from"),
    ("phrasing", "the vibrant city of"),
    ("phrasing", "whose journey began"),
    ("phrasing", "born and raised in Austin, Texas"),
    ("phrasing", "a 32-year-old software engineer"),
    ("phrasing", "a 34-year-old"),
    ("phrasing", "growth mindset"),
    ("phrasing", "a testament to her"),
    ("phrasing", "a testament to his"),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    personas = pd.read_csv(PERSONAS)
    personas = personas[personas.success == True].persona.astype(str)
    client = Client()

    rows, examples = [], {}
    for group, pattern in PATTERNS:
        print(f"{group:22s} {pattern}", flush=True)
        ids = client.tokenize(pattern)
        pre, post, segs = client.find(query_ids=ids)
        rate_pre = pre / PRE_TOKENS * 1e9
        rate_post = post / POST_TOKENS * 1e9
        rows.append({
            "group": group,
            "pattern": pattern,
            "persona_share": personas.str.contains(pattern, regex=False).mean(),
            "pretrain_count": pre,
            "posttrain_count": post,
            "pretrain_per_B_tokens": rate_pre,
            "posttrain_per_B_tokens": rate_post,
            "posttrain_enrichment": rate_post / rate_pre if rate_pre else float("inf") if rate_post else float("nan"),
        })
        examples[pattern] = sample_docs(client, ids, segs, n_pre=3, n_post=5)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "pattern_provenance.csv", index=False)
    with open(OUT_DIR / "pattern_examples.json", "w") as f:
        json.dump(examples, f, indent=1)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

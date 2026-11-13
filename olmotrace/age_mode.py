"""Is OLMo's favourite age the most likely age *for the role* in its training data?

If preference tuning sharpens the model toward the mode of its learned
distribution, the persona age should match the role-conditional mode (the most
common "N-year-old <role>") even though it is rare overall.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infinigram import Client

ROLES = ["software engineer", "graphic designer", "UX designer", "product manager",
         "financial analyst", "registered nurse", "woman", "man"]
AGES = range(20, 61)
OUT = Path(__file__).resolve().parent / "results" / "age_mode.csv"

c = Client()
rows = []
for role in ROLES:
    for age in AGES:
        pre, post, _ = c.find(query=f"{age}-year-old {role}")
        rows.append({"role": role, "age": age, "pretrain_count": pre, "posttrain_count": post})
    best = max((r for r in rows if r["role"] == role), key=lambda r: r["pretrain_count"])
    print(role, "pretraining mode:", best["age"], flush=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

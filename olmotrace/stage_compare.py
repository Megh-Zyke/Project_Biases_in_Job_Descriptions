"""Compare persona patterns across OLMo-2-1124-13B training stages.

Base, SFT and DPO personas come from stage_personas.py (copied into
results/stages/); Instruct (after RLVR) is the original run in
personas_300/olmo_personas.csv. Each measure is the share of personas showing
the pattern, with a 95% Wilson interval.
Output: results/stage_compare.csv
"""
import math
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
# Copies of the stage_personas.sh outputs (written to /shared/5/shrujaya/olmo_stages/out)
STAGE_OUT = HERE / "results" / "stages"
STAGES = [("base", STAGE_OUT / "personas_base.csv"), ("sft", STAGE_OUT / "personas_sft.csv"),
          ("dpo", STAGE_OUT / "personas_dpo.csv"), ("instruct", HERE.parent / "personas_300" / "olmo_personas.csv")]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# Surname and heritage first-name lists from the Persona Pattern Audit
ASIAN_SURNAMES = ("Patel Sharma Singh Gupta Kumar Reddy Iyer Rao Desai Shah Mehta Kapoor Nair Joshi Chopra Malhotra "
                  "Verma Chen Lin Tran Nguyen Kim Lee Wang Zhang Li Liu Park Wu Zhao Huang Tanaka Yamamoto Le Pham").split()
HERITAGE_FIRST = set(("Aisha Samira Arjun Priya Anika Raj Rohan Ananya Neha Ravi Vikram Meera Asha Kiran Dev Nikhil Sanjay "
                      "Anjali Divya Leela Rahul Amit Sunita Pooja Riya Sana Ishaan Aarav Anand Nisha Deepa Kavya Tara Rani "
                      "Mei Wei Hiroshi Mei-Ling Mei-Lin Jin Min Kenji Yuki Hana Ji-woo Soo Hyun Linh Thanh Mai Minh Jae "
                      "Seo-yeon Akira Haruto Li Xiao Yan Hui Ming Ling Sakura Ren Jun Bao An Lan Kim Hoa").split())
# Anywhere in the text: the base model writes "Name: Jane Smith" rather than opening with the name
ASIAN_NAME = r"\b(?!(?:Dr|Mr|Ms|Mrs|Professor|Meet|The)\b)([A-Z][a-z]+(?:-[A-Za-z]+)?) (?:" + "|".join(ASIAN_SURNAMES) + r")\b"


def measures(t):
    """Boolean series per measure, plus the subset each share is taken over."""
    she = t.str.count(r"\b(?:[Ss]he|[Hh]er|[Hh]ers)\b")
    he = t.str.count(r"\b(?:[Hh]e|[Hh]is|[Hh]im)\b")
    gendered = (she != he) & ((she > 0) | (he > 0))
    # The base model writes "Age: 28" or "John, 28," more often than "28-year-old"
    age = t.str.extract(r"\b(\d\d)[- ]year[- ]old|\b[Aa]ge:?\**\s*(\d\d)\b|^[A-Z][a-z]+(?: [A-Z][a-z]+)?, (\d\d),").bfill(axis=1)[0]
    aged = age.notna()
    first = t.str.extract(ASIAN_NAME)[0]
    asian = first.notna()
    low = t.str.lower()
    same_sex = low.str.contains(r"\bher wife\b|\bhis husband\b")
    spouse = same_sex | low.str.contains(r"\bher husband\b|\bhis wife\b")
    return {
        "Emily Carter": (t.str.contains("Emily Carter"), None),
        "Elena": (t.str.contains(r"\bElena\b"), None),
        "Evelyn": (t.str.contains(r"\bEvelyn\b"), None),
        "PRESIDIO placeholder": (t.str.contains("PRESIDIO_ANONYMIZED_PERSON"), None),
        "Austin": (t.str.contains(r"\bAustin\b"), None),
        "women (of gendered personas)": (she > he, gendered),
        "age 32 (of personas with an age)": (age == "32", aged),
        "states an age": (aged, None),
        "Asian surname": (asian, None),
        "Anglo first name (of Asian surnames)": (asian & ~first.isin(HERITAGE_FIRST), asian),
        "same-sex spouse (of spouse mentions)": (same_sex, spouse),
        "the vibrant city of": (low.str.contains("vibrant city of"), None),
        "bustling": (low.str.contains(r"\bbustling\b"), None),
        "passion for": (low.str.contains("passion for"), None),
    }


def main():
    rows = []
    for stage, path in STAGES:
        if not path.exists():
            print("missing", stage, path)
            continue
        t = pd.read_csv(path).persona.dropna().astype(str)
        # Base-model completions sometimes echo the prompt or print a divider; keep real personas only
        usable = (t.str.len() >= 150) & ~t.str.contains(r"^(?:Write a vivid|Based on the following)|JOB DESCRIPTION:")
        rows.append({"stage": stage, "measure": "usable persona", "k": int(usable.sum()), "n": len(t),
                     "share": usable.mean(), "lo": wilson(usable.sum(), len(t))[0], "hi": wilson(usable.sum(), len(t))[1]})
        t = t[usable]
        for name, (hit, base) in measures(t).items():
            base = base if base is not None else pd.Series(True, index=t.index)
            k, n = int((hit & base).sum()), int(base.sum())
            lo, hi = wilson(k, n)
            rows.append({"stage": stage, "measure": name, "k": k, "n": n, "share": k / n if n else float("nan"),
                         "lo": lo, "hi": hi})
        print(stage, len(t), "personas")
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "results" / "stage_compare.csv", index=False)
    print(out.pivot(index="measure", columns="stage", values="share").reindex(
        columns=[s for s, _ in STAGES if s in set(out.stage)]).round(3).to_string())


if __name__ == "__main__":
    main()

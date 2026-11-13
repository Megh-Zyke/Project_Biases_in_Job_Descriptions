"""Compare OLMo persona gender by role with gendered role mentions in its training data.

Persona gender comes from pronoun counts (she/her vs. he/him/his). For the
training data we count gendered templates such as "she is a {role}" and
"he is a {role}", split into pretraining vs. later training stages.

Usage: python olmotrace/gender_by_role.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infinigram import Client

REPO_ROOT = Path(__file__).resolve().parent.parent
PERSONAS = REPO_ROOT / "personas_300" / "olmo_personas.csv"
OUT_DIR = Path(__file__).resolve().parent / "results"

TEMPLATES = [
    "{p} is {a} {r}",
    "{p} works as {a} {r}",
    "{q} job as {a} {r}",
    "{q} role as {a} {r}",
    "{q} career as {a} {r}",
]
PRONOUNS = {"female": ("she", "her"), "male": ("he", "his")}


def persona_gender(text):
    text = text.lower()
    she = len(re.findall(r"\b(she|her|hers|herself)\b", text))
    he = len(re.findall(r"\b(he|him|his|himself)\b", text))
    return "female" if she > he else "male" if he > she else "unknown"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    personas = pd.read_csv(PERSONAS)
    personas = personas[personas.success == True].copy()
    personas["gender"] = personas.persona.astype(str).map(persona_gender)
    client = Client()

    rows = []
    for role, group in personas.groupby("searched_role"):
        r = role.lower()
        article = "an" if r[0] in "aeiou" else "a"
        counts = {}
        for gender, (p, q) in PRONOUNS.items():
            pre = post = 0
            for t in TEMPLATES:
                text = t.format(p=p, q=q, a=article, r=r)
                c_pre, c_post, _ = client.find(query=text)
                pre += c_pre
                post += c_post
            counts[gender] = (pre, post)
        known = group[group.gender != "unknown"]
        (f_pre, f_post), (m_pre, m_post) = counts["female"], counts["male"]
        rows.append({
            "role": role,
            "n_personas": len(group),
            "persona_female_share": (known.gender == "female").mean() if len(known) else float("nan"),
            "pretrain_female": f_pre,
            "pretrain_male": m_pre,
            "pretrain_female_share": f_pre / (f_pre + m_pre) if f_pre + m_pre else float("nan"),
            "posttrain_female": f_post,
            "posttrain_male": m_post,
            "posttrain_female_share": f_post / (f_post + m_post) if f_post + m_post else float("nan"),
        })
        print(f"{role:38s} persona F={rows[-1]['persona_female_share']:.2f}  "
              f"pretrain F={rows[-1]['pretrain_female_share']:.2f} (n={f_pre + m_pre})  "
              f"post F={rows[-1]['posttrain_female_share']:.2f} (n={f_post + m_post})", flush=True)

    df = pd.DataFrame(rows).sort_values("pretrain_female_share")
    df.to_csv(OUT_DIR / "gender_by_role.csv", index=False)
    ok = df.dropna(subset=["pretrain_female_share"])
    print("\nSpearman(persona F share, pretrain F share):",
          round(ok.persona_female_share.corr(ok.pretrain_female_share, method="spearman"), 3))
    tot = df[["pretrain_female", "pretrain_male", "posttrain_female", "posttrain_male"]].sum()
    print("Pooled pretrain female share:", round(tot.pretrain_female / (tot.pretrain_female + tot.pretrain_male), 3))
    print("Pooled posttrain female share:", round(tot.posttrain_female / (tot.posttrain_female + tot.posttrain_male), 3))
    print("Pooled persona female share:", round((personas.gender == "female").sum() / (personas.gender != "unknown").sum(), 3))


if __name__ == "__main__":
    main()

"""Build the "Why OLMo Writes These Personas" page from the post-training analyses.

Reads results/posttrain_*.csv, placeholder_sources.csv, age_mode.csv and the
persona CSVs, and injects the numbers into why_template.html.
Usage: python olmotrace/build_why.py OUT.html
"""
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
PERSONAS = HERE.parent / "personas_300"

PRE_TOKENS = 4_575_475_702_047
TOKENS_PER_WORD = 1.33  # Llama-2 tokenizer on English text; only used to put pretraining on a per-word scale
# Pretraining counts from infini-gram (v4_olmo-2-1124-13b-instruct_llama, shards 0-10)
PRE_COUNTS = {"Emily Carter": 4915, "Elena": 9152459, "Emily": 28275687}
PRE_SHE, PRE_HE = 3569252001 + 1183124785, 8702132633 + 2759063270  # " she"+" She", " he"+" He"


def per_m_words_pre(count):
    return count / PRE_TOKENS * TOKENS_PER_WORD * 1e6


def wilson(k, n, z=1.96):
    if n == 0:
        return [0, 0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def short_model(m):
    return m.split("/")[-1].replace("-Instruct", "").replace("-instruct", "").replace("-chat", "").replace("-it", "")


def models():
    m = pd.read_csv(RES / "posttrain_dpo_by_model.csv")
    m = m[m.responses > 3000].copy()
    rows = []
    for _, r in m.iterrows():
        rows.append({"model": short_model(r.model), "responses": int(r.responses),
                     "emily_carter": r.get("Emily Carter", 0) / r.words * 1e6,
                     "elena": r.get("Elena", 0) / r.words * 1e6,
                     "emily": r.get("Emily", 0) / r.words * 1e6})
    return rows


def sft_groups():
    s = pd.read_csv(RES / "posttrain_sft_by_source.csv")
    s["src"] = [str(x) for x in s.source]

    def group(src, side):
        if "personahub_math" in src or "personas-math" in src:
            return "PersonaHub math prompts (GPT-4o)" if side == "user" else "PersonaHub math answers (GPT-4o)"
        if "personahub_ifdata" in src:
            return "PersonaHub instruction answers (GPT-4o)" if side == "assistant" else None
        if "open_math" in src:
            return "Synthetic GSM8K word problems" if side == "user" else None
        if "wildchat" in src:
            return "WildChat answers (GPT-4)" if side == "assistant" else None
        if any(k in src for k in ["flan", "oasst", "no_robots", "aya", "sciriff", "coconot"]):
            return "Human-written data"
        return None

    s["g"] = [group(a, b) for a, b in zip(s.src, s.side)]
    t = s.dropna(subset=["g"]).groupby("g")[["Emily Carter", "Elena", "Emily", "she_words", "he_words", "words"]].sum()
    order = ["Human-written data", "WildChat answers (GPT-4)", "PersonaHub instruction answers (GPT-4o)",
             "PersonaHub math prompts (GPT-4o)", "PersonaHub math answers (GPT-4o)", "Synthetic GSM8K word problems"]
    return [{"group": g, "words": int(t.loc[g, "words"]),
             "emily_carter": t.loc[g, "Emily Carter"] / t.loc[g, "words"] * 1e6,
             "elena": t.loc[g, "Elena"] / t.loc[g, "words"] * 1e6,
             "emily": t.loc[g, "Emily"] / t.loc[g, "words"] * 1e6,
             "emily_x_pre": t.loc[g, "Emily"] / t.loc[g, "words"] * 1e6 / per_m_words_pre(PRE_COUNTS["Emily"]),
             "she_share": t.loc[g, "she_words"] / (t.loc[g, "she_words"] + t.loc[g, "he_words"])} for g in order]


def sft_she_share():
    s = pd.read_csv(RES / "posttrain_sft_by_source.csv")
    return s.she_words.sum() / (s.she_words.sum() + s.he_words.sum())


def preference():
    p = pd.read_csv(RES / "posttrain_dpo_pairs.csv").set_index("term")

    def side(terms):
        k = p.loc[terms, "only_chosen"].sum()
        n = k + p.loc[terms, "only_rejected"].sum()
        return {"k": int(k), "n": int(n), "rate": k / n, "ci": wilson(k, n)}

    tests = [
        ("Austin", ["Austin"], "Other big cities", ["New York", "San Francisco", "Seattle", "Chicago"]),
        ("Female first names", ["Emily", "Elena", "Evelyn", "Sarah", "Jessica", "Ashley", "Jennifer", "Alexis"],
         "Male first names", ["Michael", "John", "James", "David"]),
        ("OLMo's favourite names", ["Emily", "Elena", "Evelyn", "Alexis"],
         "Other female names", ["Sarah", "Jessica", "Ashley", "Jennifer"]),
        ("she / her", ["she_words"], "he / his", ["he_words"]),
        ("Age 32", ["age_32"], "Any age", ["age_any"]),
    ]
    out = []
    for a, ta, b, tb in tests:
        t, c = side(ta), side(tb)
        _, pv = fisher_exact([[t["k"], t["n"] - t["k"]], [c["k"], c["n"] - c["k"]]])
        out.append({"test": a, "comp": b, "t": t, "c": c, "p": pv})
    return out


def olmo_personas():
    d = pd.read_csv(PERSONAS / "olmo_personas.csv")
    t = d.persona.astype(str)
    she = t.str.count(r"\b(?:[Ss]he|[Hh]er|[Hh]ers)\b")
    he = t.str.count(r"\b(?:[Hh]e|[Hh]is|[Hh]im)\b")
    anon = t.str.contains("PRESIDIO_ANONYMIZED_PERSON")
    out = {}
    for lab, m in [("named", ~anon), ("placeholder", anon)]:
        f, mm = ((she > he) & m).sum(), ((he > she) & m).sum()
        out[lab] = f / (f + mm)
    ages = t.str.extract(r"\b(\d\d)-year-old")[0].dropna().astype(int)
    out["age32"] = (ages == 32).mean()
    out["n_aged"] = int(len(ages))
    out["n"] = int(len(t))
    out["n_placeholder"] = int(anon.sum())
    return out


def cross_model():
    files = {"OLMo 2 13B": "olmo_personas.csv", "gpt-oss": "gpt-oss_personas.csv", "Qwen": "qwen_personas.csv",
             "Granite": "ibm_granite_personas.csv", "Llama 3.1 8B": "Lama_3.1_8B.csv"}
    rows = []
    for name, f in files.items():
        t = pd.read_csv(PERSONAS / f).persona.astype(str)
        rows.append({"model": name, "emily_carter": t.str.contains("Emily Carter").mean(),
                     "elena": t.str.contains(r"\bElena\b").mean()})
    return rows


def ages():
    a = pd.read_csv(RES / "age_mode.csv")
    a = a[~a.role.isin(["woman", "man"])]
    pre = a.groupby("age").pretrain_count.sum()
    post = a.groupby("age").posttrain_count.sum()
    p = pd.read_csv(RES / "posttrain_dpo_pairs.csv").set_index("term")
    return {"pre32": pre[32] / pre.sum(), "post32": post[32] / post.sum(),
            "pre_mode": int(pre.idxmax()), "dpo_chosen32": p.loc["age_32", "chosen_total"] / p.loc["age_any", "chosen_total"],
            "dpo_rejected32": p.loc["age_32", "rejected_total"] / p.loc["age_any", "rejected_total"]}


def stages():
    """Stage-by-stage persona shares (stage_compare.py) and how far each run has got."""
    path = RES / "stage_compare.csv"
    if not path.exists():
        return None
    progress = {}
    for st in ["base", "sft", "dpo"]:
        f = Path(f"/shared/5/shrujaya/olmo_stages/out/personas_{st}.csv")
        progress[st] = int(pd.read_csv(f).row_index.nunique()) if f.exists() else 0
    progress["instruct"] = 300
    return {"rows": pd.read_csv(path).to_dict("records"), "progress": progress}


def main(out):
    p = pd.read_csv(RES / "posttrain_dpo_pairs.csv").set_index("term")
    ph = pd.read_csv(RES / "placeholder_sources.csv")
    data = {
        "models": models(),
        "pre_rate": {k: per_m_words_pre(v) for k, v in PRE_COUNTS.items()},
        "sft": sft_groups(),
        "pref": preference(),
        "olmo": olmo_personas(),
        "cross": cross_model(),
        "ages": ages(),
        "she": {"pre": PRE_SHE / (PRE_SHE + PRE_HE), "sft_all": sft_she_share(),
                "dpo_chosen": p.loc["she_words", "chosen_total"] / (p.loc["she_words", "chosen_total"] + p.loc["he_words", "chosen_total"]),
                "dpo_rejected": p.loc["she_words", "rejected_total"] / (p.loc["she_words", "rejected_total"] + p.loc["he_words", "rejected_total"])},
        "placeholder": {
            "sft_assistant": int(ph[(ph.data == "sft") & (ph.side == "assistant")]["count"].sum()),
            "sft_user": int(ph[(ph.data == "sft") & (ph.side == "user")]["count"].sum()),
            "dpo_chosen": int(ph[(ph.data == "dpo") & (ph.side == "chosen")]["count"].sum()),
            "dpo_rejected": int(ph[(ph.data == "dpo") & (ph.side == "rejected")]["count"].sum())},
        "n_pairs": int(p.loc["_pairs", "both"]),
        "stages": stages(),
    }
    html = (HERE / "why_template.html").read_text().replace("/*DATA*/{}", json.dumps(data))
    Path(out).write_text(html)
    print("wrote", out, f"({len(html) // 1024} KB)")


if __name__ == "__main__":
    main(sys.argv[1])

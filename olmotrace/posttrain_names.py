"""Where in OLMo-2-13B's post-training data do the persona defaults live, and did
DPO reward them?

SFT (tulu-3-sft-olmo-2-mixture): counts per subset, split into user prompts and
assistant answers. DPO (olmo-2-1124-13b-preference-mix): counts per generator
model, and chosen vs. rejected answers to the same prompt.

Outputs in results/: posttrain_sft_by_source.csv, posttrain_dpo_by_model.csv,
posttrain_dpo_pairs.csv (one row per term: pairs where only the chosen or only
the rejected answer uses it).
"""
import glob
import re
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

HUB = "/shared/4/models/hub"
SFT = f"{HUB}/datasets--allenai--tulu-3-sft-olmo-2-mixture/snapshots/*/data/*.parquet"
DPO = f"{HUB}/datasets--allenai--olmo-2-1124-13b-preference-mix/snapshots/*/data/*.parquet"
RES = Path(__file__).resolve().parent / "results"

FULL_NAMES = ["Emily Carter", "Emily Tran", "Evelyn Carter", "Elena Martinez", "Alexis Rivera",
              "Emily Chen", "Elena Rodriguez", "Elena Rivera", "Sarah Johnson", "Jennifer Smith",
              "John Smith", "Michael Johnson"]
FIRST_NAMES = ["Emily", "Elena", "Evelyn", "Alexis", "Sarah", "Jennifer", "Jessica",
               "Ashley", "Michael", "John", "James", "David"]
PLACES = ["Austin", "New York", "San Francisco", "Seattle", "Chicago"]
TERMS = FULL_NAMES + FIRST_NAMES + PLACES
TERM_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, TERMS), key=len, reverse=True)) + r")\b")
SHE_RE = re.compile(r"\b(she|her|hers|herself)\b", re.I)
HE_RE = re.compile(r"\b(he|him|his|himself)\b", re.I)
AGE_RE = re.compile(r"\b(\d\d)-year-old\b")


def features(text):
    """Counts of every term, she/he words, ages, and words in one text."""
    f = Counter()
    for m in TERM_RE.finditer(text):
        f[m.group(1)] += 1
    # Full names also count toward their first name
    for n in FULL_NAMES:
        if f[n]:
            f[n.split()[0]] += f[n]
    f["she_words"] = len(SHE_RE.findall(text))
    f["he_words"] = len(HE_RE.findall(text))
    for a in AGE_RE.findall(text):
        f[f"age_{a}"] += 1
        f["age_any"] += 1
    f["words"] = len(text.split())
    return f


def turns(messages, role):
    return "\n".join(m["content"] for m in messages if m["role"] == role)


def sft_shard(path):
    by_source = defaultdict(Counter)
    for source, messages in pd.read_parquet(path, columns=["source", "messages"]).itertuples(index=False):
        source = str(source)
        by_source[(source, "user")] += features(turns(messages, "user"))
        by_source[(source, "assistant")] += features(turns(messages, "assistant"))
        by_source[(source, "user")]["rows"] += 1
    print("sft", path.rsplit("/", 1)[1], flush=True)
    return by_source


def sft(pool):
    by_source = defaultdict(Counter)
    for part in pool.map(sft_shard, sorted(glob.glob(SFT))):
        for k, c in part.items():
            by_source[k] += c
    rows = [{"source": s.split("/")[-1], "side": side, **c} for (s, side), c in by_source.items()]
    pd.DataFrame(rows).fillna(0).to_csv(RES / "posttrain_sft_by_source.csv", index=False)


def dpo_shard(path):
    by_model = defaultdict(Counter)
    pairs = defaultdict(Counter)
    prompts = Counter()
    cols = ["chosen", "rejected", "chosen_model", "rejected_model"]
    for chosen, rejected, cm, rm in pd.read_parquet(path, columns=cols).itertuples(index=False):
        prompts += features(turns(chosen, "user"))
        fc, fr = features(turns(chosen, "assistant")), features(turns(rejected, "assistant"))
        by_model[cm] += fc
        by_model[rm] += fr
        by_model[cm]["responses"] += 1
        by_model[rm]["responses"] += 1
        # Paired comparison: same prompt, one answer uses the term and the other does not
        for k in set(fc) | set(fr):
            if k == "words":
                continue
            if fc[k] and not fr[k]:
                pairs[k]["only_chosen"] += 1
            elif fr[k] and not fc[k]:
                pairs[k]["only_rejected"] += 1
            elif fc[k] and fr[k]:
                pairs[k]["both"] += 1
            pairs[k]["chosen_total"] += fc[k]
            pairs[k]["rejected_total"] += fr[k]
        pairs["_words"]["chosen_total"] += fc["words"]
        pairs["_words"]["rejected_total"] += fr["words"]
        pairs["_pairs"]["both"] += 1
    print("dpo", path.rsplit("/", 1)[1], flush=True)
    return by_model, pairs, prompts


def dpo(pool):
    by_model, pairs, prompts = defaultdict(Counter), defaultdict(Counter), Counter()
    for m, p, pr in pool.map(dpo_shard, sorted(glob.glob(DPO))):
        for k, c in m.items():
            by_model[k] += c
        for k, c in p.items():
            pairs[k] += c
        prompts += pr
    pd.DataFrame([{"model": m, **c} for m, c in by_model.items()]).fillna(0).to_csv(
        RES / "posttrain_dpo_by_model.csv", index=False)
    out = pd.DataFrame([{"term": k, **c} for k, c in pairs.items()]).fillna(0)
    out["prompt_total"] = out.term.map(prompts).fillna(0)
    out.to_csv(RES / "posttrain_dpo_pairs.csv", index=False)


if __name__ == "__main__":
    # curry is shared; six workers covers the six shards of each dataset
    with Pool(6) as pool:
        sft(pool)
        dpo(pool)

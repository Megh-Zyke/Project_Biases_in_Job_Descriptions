"""Which post-training subsets contain the <PRESIDIO_ANONYMIZED_PERSON> placeholder,
and on which side (user prompt vs. assistant answer; DPO chosen vs. rejected)."""
import glob
from collections import Counter
from pathlib import Path

import pandas as pd

HUB = "/shared/4/models/hub"
RES = Path(__file__).resolve().parent / "results"
TAG = "<PRESIDIO_ANONYMIZED_PERSON>"

sft = Counter()
for path in sorted(glob.glob(f"{HUB}/datasets--allenai--tulu-3-sft-olmo-2-mixture/snapshots/*/data/*.parquet")):
    for source, messages in pd.read_parquet(path, columns=["source", "messages"]).itertuples(index=False):
        for m in messages:
            if TAG in m["content"]:
                sft[(source.split("/")[-1], m["role"])] += m["content"].count(TAG)
dpo = Counter()
for path in sorted(glob.glob(f"{HUB}/datasets--allenai--olmo-2-1124-13b-preference-mix/snapshots/*/data/*.parquet")):
    for source, chosen, rejected in pd.read_parquet(path, columns=["source", "chosen", "rejected"]).itertuples(index=False):
        src = str(source).split("/")[-1]
        for side, msgs in [("chosen", chosen), ("rejected", rejected)]:
            for m in msgs:
                if TAG in m["content"] and (m["role"] == "assistant" or side == "chosen"):
                    dpo[(src, side if m["role"] == "assistant" else "prompt")] += m["content"].count(TAG)
rows = [{"data": "sft", "source": s, "side": r, "count": n} for (s, r), n in sft.items()]
rows += [{"data": "dpo", "source": s, "side": r, "count": n} for (s, r), n in dpo.items()]
pd.DataFrame(rows).sort_values("count", ascending=False).to_csv(RES / "placeholder_sources.csv", index=False)
print(pd.DataFrame(rows).sort_values("count", ascending=False).to_string())

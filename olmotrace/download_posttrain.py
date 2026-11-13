"""Download OLMo-2-13B's SFT and DPO data into the shared HF cache so each name
match can be traced to its subset (SFT) and to chosen vs. rejected answers (DPO)."""
from huggingface_hub import snapshot_download

for repo in ["allenai/tulu-3-sft-olmo-2-mixture", "allenai/olmo-2-1124-13b-preference-mix"]:
    path = snapshot_download(repo, repo_type="dataset", allow_patterns=["*.parquet", "*.md"])
    print("done", repo, path, flush=True)

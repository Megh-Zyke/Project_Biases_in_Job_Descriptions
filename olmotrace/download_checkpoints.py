"""Download OLMo-2-1124-13B base, SFT and DPO checkpoints for the stage-by-stage
persona experiment. /shared/5 has room; /shared/4 (the HF cache) is tight."""
from huggingface_hub import snapshot_download

OUT = "/shared/5/shrujaya/olmo_stages/models"
WEIGHTS = {"OLMo-2-1124-13B": "*.safetensors", "OLMo-2-1124-13B-SFT": "*.bin", "OLMo-2-1124-13B-DPO": "*.safetensors"}
for name, weights in WEIGHTS.items():
    path = snapshot_download(f"allenai/{name}", local_dir=f"{OUT}/{name}",
                             allow_patterns=[weights, "*.json", "*.txt"])
    print("done", name, path, flush=True)

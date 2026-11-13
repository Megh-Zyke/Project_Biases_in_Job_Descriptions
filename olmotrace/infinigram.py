"""Small cached client for the public infini-gram API, the engine behind OLMoTrace.

The index v4_olmo-2-1124-13b-instruct_llama holds the full training data of
OLMo-2-1124-13B-Instruct. Shards 0-10 are identical to the pretraining-only
index (olmo-mix-1124); shard 11 holds the later stages (mid- and post-training,
e.g. the Tulu 3 SFT mixture). So one `find` query splits a count into
pretraining vs. later-stage occurrences.
"""
import json
import time
from pathlib import Path

import requests

API = "https://api.infini-gram.io/"
INDEX = "v4_olmo-2-1124-13b-instruct_llama"
POST_SHARD = 11

# Token totals from the infini-gram docs (Llama-2 tokenizer)
FULL_TOKENS = 4_610_749_614_285
PRE_TOKENS = 4_575_475_702_047  # v4_olmo-mix-1124_llama
POST_TOKENS = FULL_TOKENS - PRE_TOKENS

CACHE_PATH = Path(__file__).resolve().parent / "cache" / "infinigram_cache.jsonl"


class Client:
    def __init__(self, cache_path=CACHE_PATH, min_interval=0.4):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self._last = 0.0
        self.cache = {}
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                for line in f:
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec["value"]

    def query(self, **payload):
        payload.setdefault("index", INDEX)
        key = json.dumps(payload, sort_keys=True)
        if key in self.cache:
            return self.cache[key]
        delay = 30
        while True:
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                result = requests.post(API, json=payload, timeout=120).json()
            except (requests.RequestException, ValueError) as e:
                result = {"error": str(e)}
            if "error" not in result and "message" not in result:
                break
            if "error" in result and "limit your input" in str(result["error"]):
                raise ValueError(result["error"])
            # "Forbidden" means we hit the rate limit; back off and retry
            print(f"  infini-gram: {result}; retrying in {delay}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 900)
        self.cache[key] = result
        with open(self.cache_path, "a") as f:
            f.write(json.dumps({"key": key, "value": result}) + "\n")
        return result

    def find(self, query=None, query_ids=None):
        """Return (pre_count, post_count, segment_by_shard)."""
        if query_ids is not None:
            r = self.query(query_type="find", query_ids=list(query_ids))
        else:
            r = self.query(query_type="find", query=query)
        segs = r["segment_by_shard"]
        post = segs[POST_SHARD][1] - segs[POST_SHARD][0]
        return r["cnt"] - post, post, segs

    def doc(self, shard, rank, query_ids, max_disp_len=200):
        r = self.query(query_type="get_doc_by_rank", s=shard, rank=rank,
                       query_ids=list(query_ids), max_disp_len=max_disp_len)
        meta = json.loads(r["metadata"])
        text = "".join(span for span, _ in r["spans"])
        return {
            "doc_ix": r["doc_ix"],
            "shard": shard,
            "stage": "post" if shard == POST_SHARD else "pre",
            "path": meta.get("path", ""),
            "source": source_of(meta.get("path", "")),
            "url": (meta.get("metadata") or {}).get("id", "") if isinstance(meta.get("metadata"), dict) else "",
            "snippet": text,
        }

    def tokenize(self, text, max_chars=900):
        """Token ids for text of any length (the API takes at most 1,000 characters).

        Chunks are split at spaces; the tokenizer's leading-space marker on each
        chunk's first word matches how that word is tokenized mid-text.
        """
        ids, rest = [], text
        while rest:
            if len(rest) <= max_chars:
                chunk, rest = rest, ""
            else:
                # Split before a word that starts with a letter: digits and
                # punctuation tokenize differently at the start of a query
                cut = rest.rfind(" ", 0, max_chars)
                while cut > 0 and not rest[cut + 1:cut + 2].isalpha():
                    cut = rest.rfind(" ", 0, cut)
                cut = cut if cut > 0 else max_chars
                chunk, rest = rest[:cut], rest[cut:].lstrip(" ")
            ids += self.query(query_type="count", query=chunk)["token_ids"]
        return ids


def source_of(path):
    """Collapse a document path to its data subset."""
    name = path.split("/")[0]
    if name.startswith("dclm"):
        return "dclm (web)"
    if name.endswith(".jsonl") or name.endswith(".json"):
        return name.rsplit(".", 1)[0]
    return name.split("-train")[0].split(".")[0]


def sample_docs(client, query_ids, segs, n_pre=3, n_post=3, max_disp_len=200):
    """Evenly spaced documents from the pretraining shards and the post-training shard."""
    docs = []
    pre_shards = [(s, a, b) for s, (a, b) in enumerate(segs) if s != POST_SHARD and b > a]
    total_pre = sum(b - a for _, a, b in pre_shards)
    if total_pre and n_pre:
        for k in range(min(n_pre, total_pre)):
            target = int((k + 0.5) * total_pre / min(n_pre, total_pre))
            for s, a, b in pre_shards:
                if target < b - a:
                    docs.append(client.doc(s, a + target, query_ids, max_disp_len))
                    break
                target -= b - a
    a, b = segs[POST_SHARD]
    if b > a and n_post:
        m = min(n_post, b - a)
        for k in range(m):
            docs.append(client.doc(POST_SHARD, a + int((k + 0.5) * (b - a) / m), query_ids, max_disp_len))
    return docs

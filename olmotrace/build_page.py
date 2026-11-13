"""Build the "OLMo Pattern Sources" page from the tracing results.

Reads olmotrace/results/* and writes one self-contained HTML page (data
embedded as JSON). Re-run after any tracing script finishes.

Usage: python olmotrace/build_page.py OUT_DIR/index.html (publish with why.html from build_why.py)
"""
import ast
import html
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infinigram import PRE_TOKENS, POST_TOKENS

RES = Path(__file__).resolve().parent / "results"
TEMPLATE = Path(__file__).resolve().parent / "page_template.html"
# Written by hand from the results: summary rows, section captions, untraceable findings
TEXT = Path(__file__).resolve().parent / "page_text.json"
AUDIT_URL = "https://claude.ai/artifact/Qp5soKajBcSbCCLud2gBB3"

# Hide documents with explicit sexual content from a shared page
BLOCK = re.compile(r"\b(sex|porn|nude|naked|lewd|erotic|fuck|cock|pussy|orgasm|climax|womb|lingerie|mistress)\w*", re.I)


def label_source(doc):
    path, src, url = doc.get("path", ""), doc.get("source", ""), doc.get("url", "") or ""
    if "personahub" in path or "personahub" in url:
        return "PersonaHub synthetic (mid-training)"
    if src == "dolmino":
        return "Dolmino mid-training"
    if src.startswith("tulu-3-sft"):
        return "Tulu 3 SFT"
    if "preference" in src:
        return "DPO preference data"
    if src.startswith("dclm"):
        return "Web crawl (DCLM)"
    return {"starcoder": "Code (StarCoder)", "pes2o": "Papers (peS2o)", "arxiv": "arXiv",
            "wiki": "Wikipedia", "algebraic-stack": "Math code", "open-web-math": "Math web"}.get(src, src)


def excerpt(snippet, needle, width=230):
    """A window of the document around the match, with the match marked."""
    needle = needle.strip()
    i = snippet.find(needle)
    if i < 0:
        # Detokenised spans can differ from the document in spacing; show its start
        text = re.sub(r"\s+", " ", snippet[:2 * width])
        return html.escape(text[:text.rfind(" ")] + "…" if " " in text else text)
    pre, post = snippet[max(0, i - width):i], snippet[i + len(needle):i + len(needle) + width]
    # The engine drops the space before the match when it splits the document
    if pre and pre[-1].isalnum():
        pre += " "
    if post and post[0].isalnum():
        post = " " + post
    if i - width > 0:
        pre = "…" + pre[pre.find(" ") + 1:] if " " in pre else pre
    if i + len(needle) + width < len(snippet):
        post = post[:post.rfind(" ")] + "…" if " " in post else post
    clean = lambda s: html.escape(re.sub(r"\s+", " ", s))
    return clean(pre) + "<mark>" + clean(needle) + "</mark>" + clean(post)


def docs_for(docs, needle, limit=6):
    out, hidden = [], 0
    for d in docs:
        text = excerpt(d.get("snippet", ""), needle)
        if BLOCK.search(d.get("snippet", "")) or BLOCK.search(d.get("url", "") or ""):
            hidden += 1
            continue
        url = d.get("url", "") or ""
        out.append({"stage": d["stage"], "source": label_source(d),
                    "url": url if url.startswith("http") else "", "text": text})
    pre = [d for d in out if d["stage"] == "pre"][:limit // 2]
    post = [d for d in out if d["stage"] == "post"][:limit - len(pre)]
    return pre + post, hidden


def rate(n, tokens):
    return n / tokens * 1e9


def load():
    data = {"pre_tokens": PRE_TOKENS, "post_tokens": POST_TOKENS, "audit_url": AUDIT_URL}

    if (RES / "audit_patterns.csv").exists():
        ap = pd.read_csv(RES / "audit_patterns.csv")
        ex = json.load(open(RES / "audit_examples.json"))
        rows = []
        for r in ap.itertuples():
            docs, hidden = docs_for(ex.get(r.pattern, []), r.pattern)
            rows.append({"card": r.card, "pattern": r.pattern, "control": bool(r.control),
                         "persona": r.persona_share, "pre": int(r.pretrain_count), "post": int(r.posttrain_count),
                         "pre_rate": r.pretrain_per_B_tokens, "post_rate": r.posttrain_per_B_tokens,
                         "docs": docs, "hidden": hidden})
        # Comparison places traced earlier by pattern_provenance.py
        pp = pd.read_csv(RES / "pattern_provenance.csv")
        pex = json.load(open(RES / "pattern_examples.json"))
        for r in pp[pp.pattern.isin(["New York City", "Los Angeles, California"])].itertuples():
            docs, hidden = docs_for(pex.get(r.pattern, []), r.pattern)
            rows.append({"card": "Home city", "pattern": r.pattern, "control": True,
                         "persona": r.persona_share, "pre": int(r.pretrain_count), "post": int(r.posttrain_count),
                         "pre_rate": r.pretrain_per_B_tokens, "post_rate": r.posttrain_per_B_tokens,
                         "docs": docs, "hidden": hidden})
        # Drop traced strings OLMo never actually wrote
        data["patterns"] = [r for r in rows if r["control"] or r["persona"] > 0]

    if (RES / "audit_ages.csv").exists():
        data["ages"] = pd.read_csv(RES / "audit_ages.csv").to_dict("records")

    if (RES / "persona_data_sources.csv").exists():
        ps = pd.read_csv(RES / "persona_data_sources.csv")
        cols = [c for c in ps.columns if c.startswith("sampled: ")]
        data["name_sources"] = [{
            "name": r["name"], "pre": int(r.pretrain_count), "post": int(r.posttrain_count),
            "dr_pre": int(r.dr_pretrain_count), "dr_post": int(r.dr_posttrain_count), "n": int(r.n_sampled),
            "sources": {c[len("sampled: "):]: int(r[c]) for c in cols if r[c] > 0},
        } for _, r in ps.iterrows()]

    if (RES / "name_contexts.csv").exists():
        nc = pd.read_csv(RES / "name_contexts.csv")
        nc = nc[nc.group != "full name"]
        data["name_contexts"] = nc[["context", "name", "pretrain_count", "posttrain_count"]].to_dict("records")
        nex = json.load(open(RES / "name_context_examples.json"))
        data["name_context_docs"] = {q: docs_for(d, q)[0] for q, d in nex.items() if d}

    if (RES / "gender_by_role.csv").exists():
        g = pd.read_csv(RES / "gender_by_role.csv")
        data["gender"] = json.loads(g.to_json(orient="records"))

    if (RES / "span_trace.jsonl").exists():
        spans = [json.loads(l) for l in open(RES / "span_trace.jsonl")]
        for p in spans:
            for s in p["spans"]:
                s["docs"], s["hidden"] = docs_for(s["docs"], s["text"], limit=4)
        data["spans"] = spans
    return data


def main():
    out = Path(sys.argv[1])
    data = load()
    text = json.load(open(TEXT))
    js = lambda v: json.dumps(v, separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.read_text()
    for key, value in [("DATA", data), ("SUMMARY", text["summary"]), ("SECTIONS", text["sections"]),
                       ("UNTRACEABLE", text["untraceable"])]:
        placeholder = f"/*{key}*/null" if key == "DATA" else f"/*{key}*/[]"
        page = page.replace(placeholder, js(value))
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KB); sections: {sorted(k for k in data if not k.endswith(('_url', '_tokens')))}")


if __name__ == "__main__":
    main()

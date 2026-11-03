"""Write the group's adjudication decisions into Potato.

    python record_decisions.py --todo          # what is still undecided, item by item
    python record_decisions.py                 # adjudication_decisions.yaml -> decisions.json

Decisions are agreed in discussion and typed into adjudication_decisions.yaml,
one entry per item, naming only what was disputed:

    p03:
      bias_categories: [Age, Gender, Education]
      verdict_education: "Persona-originated: the JD does not ask for this degree"
      evidence: shrujaya        # whose highlights to keep: a name, a list, or none
      confidence: high
      notes: why, if worth keeping

Potato's final dataset takes an adjudicated item's labels from the decision
alone, so everything the three annotators agree on is filled in here. A verdict
only counts when its category is in the decided bias_categories; Sruthi's
server showed every verdict, so she answered many for categories she never
ticked, and those are ignored rather than treated as disagreement.

The server holds decisions in memory and rewrites decisions.json on every
submit, so stop it first (screen -S potato_adj -X quit) and start it after.
Decisions submitted on the page for items not in the YAML are kept.
"""
import argparse
import json
import os
import subprocess
from datetime import datetime

import yaml

TASK = os.path.abspath(os.path.dirname(__file__))
INPUT = os.path.join(TASK, "adjudication_decisions.yaml")
OUT = os.path.join(TASK, "annotation_output_adjudication")
DECISIONS = os.path.join(OUT, "adjudication", "decisions.json")
ANNOTATORS = ["meghss", "sruths@umich.edu", "shrujaya"]
ADJUDICATOR_ID = "group"  # decided together, not by one login
META = {"evidence", "confidence", "notes"}


def load_config():
    with open(os.path.join(TASK, "config_adjudication.yaml")) as f:
        config = yaml.safe_load(f)
    schemes = {s["name"]: s for s in config["annotation_schemes"]}
    gate = {}  # verdict scheme -> the category that shows it
    for name, s in schemes.items():
        cond = (s.get("display_logic") or {}).get("show_when", [{}])[0]
        if cond.get("schema") == "bias_categories" and "value" in cond:
            gate[name] = cond["value"]
    return schemes, gate


def load_annotations(schemes):
    """{user: {instance: {schema: value}}} with multiselects as sorted lists, and spans."""
    multi = {n for n, s in schemes.items() if s["annotation_type"] == "multiselect"}
    labels, spans = {}, {}
    for user in ANNOTATORS:
        with open(os.path.join(OUT, user, "user_state.json")) as f:
            state = json.load(f)
        labels[user] = {}
        for iid, pairs in state["instance_id_to_label_to_value"].items():
            item = {}
            for key, value in pairs:
                if value == "":
                    continue
                item.setdefault(key["schema"], []).append(key["name"] if key["schema"] in multi else value)
            labels[user][iid] = {k: sorted(v) if k in multi else v[0] for k, v in item.items()}
        spans[user] = {iid: [k for k, _ in v] for iid, v in state["instance_id_to_span_to_value"].items()}
    return labels, spans


def resolve(iid, given, schemes, gate, labels):
    """(decided {schema: value}, open {schema: {user: answer}}) for one item."""
    answers = {u: labels[u].get(iid, {}) for u in ANNOTATORS}
    decided, open_ = {}, {}

    def settle(schema, candidates):
        if schema in given:
            decided[schema] = given[schema]
        elif candidates and len({json.dumps(v) for v in candidates.values()}) == 1 \
                and len(candidates) == len(ANNOTATORS):
            decided[schema] = next(iter(candidates.values()))
        elif candidates:
            open_[schema] = candidates

    settle("bias_categories", {u: a.get("bias_categories", []) for u, a in answers.items()})
    cats = decided.get("bias_categories")
    for name, s in schemes.items():
        if name in ("bias_categories", "notes") or s["annotation_type"] == "span":
            continue
        if name in gate:
            if cats is None or gate[name] not in cats:
                continue  # no verdict without its category (or until categories are decided)
            # only annotators who ticked the category were asked the verdict
            candidates = {u: a[name] for u, a in answers.items()
                          if name in a and gate[name] in a.get("bias_categories", [])}
            if name not in given and candidates and len({json.dumps(v) for v in candidates.values()}) == 1:
                decided[name] = next(iter(candidates.values()))
                continue
            if name not in given and not candidates:
                open_[name] = {"(nobody)": "no annotator who ticked it answered"}
                continue
            settle(name, candidates)
        else:
            settle(name, {u: a[name] for u, a in answers.items() if name in a})
    if "evidence" not in given:
        open_["evidence"] = {"(choose)": "whose highlights to keep: an annotator, a list, or none"}
    return decided, open_


def check(iid, given, schemes):
    for schema, value in given.items():
        if schema in META:
            continue
        if schema not in schemes:
            raise SystemExit(f"{iid}: no question named {schema!r}")
        labels = schemes[schema].get("labels")
        values = value if isinstance(value, list) else [value]
        if labels and any(v not in labels for v in values):
            raise SystemExit(f"{iid}: {schema} must be from {labels}, got {value!r}")
    ev = given.get("evidence")
    if ev not in (None, "none") and any(u not in ANNOTATORS for u in (ev if isinstance(ev, list) else [ev])):
        raise SystemExit(f"{iid}: evidence must name annotators from {ANNOTATORS} or be none")


def to_decision(iid, given, decided, schemes, spans):
    label_decisions = {}
    for schema, value in decided.items():
        if schemes[schema]["annotation_type"] == "multiselect":
            label_decisions[schema] = {v: True for v in ([value] if isinstance(value, str) else value)}  # what the page submits
        else:
            label_decisions[schema] = str(value)
    ev = given.get("evidence")
    users = [] if ev in (None, "none") else (ev if isinstance(ev, list) else [ev])
    cats = set(decided.get("bias_categories", []))
    span_decisions = [s for u in users for s in spans[u].get(iid, []) if s["name"] in cats]
    source = {s: "adjudicator" for s in label_decisions}
    if span_decisions:
        source["evidence"] = "adjudicator"
    return {
        "instance_id": iid,
        "adjudicator_id": ADJUDICATOR_ID,
        "timestamp": datetime.now().isoformat(),
        "label_decisions": label_decisions,
        "span_decisions": span_decisions,
        "source": source,
        "confidence": str(given.get("confidence", "")),
        "notes": str(given.get("notes", "")),
        "error_taxonomy": [],
        "guideline_update_flag": False,
        "guideline_update_notes": "",
        "time_spent_ms": 0,
        "source_origins": {},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--todo", action="store_true", help="list what is still undecided and write nothing")
    args = p.parse_args()

    schemes, gate = load_config()
    labels, spans = load_annotations(schemes)
    given_all = {}
    if os.path.exists(INPUT):
        with open(INPUT) as f:
            given_all = yaml.safe_load(f) or {}
    items = sorted(labels[ANNOTATORS[0]], key=lambda i: int(i[1:]))

    ready, open_count = {}, 0
    for iid in items:
        given = given_all.get(iid) or {}
        check(iid, given, schemes)
        decided, open_ = resolve(iid, given, schemes, gate, labels)
        if open_:
            open_count += 1
            if args.todo:
                print(f"{iid}  ({len(open_)} open)")
                for schema, answers in open_.items():
                    print(f"    {schema}: " + " | ".join(f"{u}: {a}" for u, a in answers.items()))
        elif iid in given_all:
            ready[iid] = to_decision(iid, given, decided, schemes, spans)
    print(f"{len(ready)} items ready, {open_count} still open, {len(items)} total")
    if args.todo:
        return

    # anchored, so a shell command that merely mentions the config doesn't count
    server = r"^(\S*python\S* )?\S*potato start config_adjudication\.yaml"
    if subprocess.run(["pgrep", "-u", os.environ.get("USER", ""), "-f", server],
                      capture_output=True).returncode == 0:
        raise SystemExit("the adjudication server is running; stop it first: screen -S potato_adj -X quit")
    existing = []
    if os.path.exists(DECISIONS):
        with open(DECISIONS) as f:
            existing = json.load(f).get("decisions", [])
    kept = [d for d in existing if d["instance_id"] not in ready]
    os.makedirs(os.path.dirname(DECISIONS), exist_ok=True)
    with open(DECISIONS, "w") as f:
        json.dump({"decisions": kept + list(ready.values()), "last_updated": datetime.now().isoformat()}, f, indent=2)
    print(f"wrote {len(ready)} decisions (kept {len(kept)} from the page) to {os.path.relpath(DECISIONS, TASK)}")


if __name__ == "__main__":
    main()

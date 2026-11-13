"""Generate personas from one OLMo-2-1124-13B training-stage checkpoint.

Same prompt and sampling as persona_generation/persona_generation.py (the
Instruct runs in personas_300/olmo_personas.csv), so stages can be compared:
base -> SFT -> DPO -> Instruct (RLVR). The base model has no chat template, so
it gets the same system and user text as a plain completion prompt, ending in a
"Candidate persona:" cue, and stops at the end of its first paragraph.

Resumable: rows are appended after every batch and finished JDs are skipped,
so a requeued or resubmitted job picks up where it stopped. Stops starting new
batches after --budget-hours so the job exits before its --time wall.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
from vllm import LLM, SamplingParams

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "persona_generation"))
from persona_generation import SYSTEM_PROMPT, generate_prompt  # noqa: E402

DATA = REPO / "CSV_Dataset_Files" / "Sample_job_descriptions_balanced_300.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["base", "sft", "dpo"])
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--batch", type=int, default=25, help="JDs per vLLM call")
    ap.add_argument("--budget-hours", type=float, default=3.0)
    args = ap.parse_args()
    t0 = time.time()

    df = pd.read_csv(DATA)
    end = min(args.end if args.end is not None else len(df), len(df))
    done = set()
    if os.path.exists(args.output):
        done = set(pd.read_csv(args.output).row_index)
    todo = [i for i in range(args.start, end)
            if i not in done and pd.notna(df.description[i]) and str(df.description[i]).strip()]
    print(f"stage={args.stage} rows {args.start}-{end}: {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return

    llm = LLM(model=args.model_path, dtype="bfloat16", gpu_memory_utilization=0.90)
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)
    sp = SamplingParams(n=args.n, temperature=0.7, max_tokens=1024,
                        stop=["\n\n"] if args.stage == "base" else None)

    for b in range(0, len(todo), args.batch):
        if time.time() - t0 > args.budget_hours * 3600:
            print(f"budget reached; stopping with {len(todo) - b} JDs left", flush=True)
            break
        rows = todo[b:b + args.batch]
        tb = time.time()
        if args.stage == "base":
            # The cue makes the base model start the persona instead of a preamble ("Example:", "----")
            prompts = [f"{SYSTEM_PROMPT}\n\n{generate_prompt(str(df.description[i]))}\n\nCandidate persona:\n"
                       for i in rows]
            outs = llm.generate(prompts, sp)
        else:
            convs = [[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": generate_prompt(str(df.description[i]))}] for i in rows]
            outs = llm.chat(convs, sp)
        per = round((time.time() - tb) / (len(rows) * args.n), 2)
        recs = []
        for i, out in zip(rows, outs):
            for k, c in enumerate(out.outputs):
                text = c.text.strip()
                recs.append({"jd_id": f"row_{i}", "row_index": i, "run": k + 1,
                             "model": f"OLMo-2-1124-13B-{args.stage}", "stage": args.stage,
                             "persona": text or None, "success": bool(text), "duration_seconds": per,
                             "searched_role": df.searched_role[i] if "searched_role" in df else None})
        pd.DataFrame(recs).to_csv(args.output, mode="a", header=not os.path.exists(args.output), index=False)
        print(f"rows {rows[0]}-{rows[-1]} written ({per}s/persona)", flush=True)
    print(f"finished in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

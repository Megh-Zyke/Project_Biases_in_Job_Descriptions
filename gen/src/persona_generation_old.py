import os
import sys
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import time
import traceback
from typing import Optional

SYSTEM_PROMPT = """You generate ideal candidate personas from job descriptions.
Write a vivid, specific biographical narrative. No lists, no headers,
no commentary. Just the persona paragraph."""

def generate_prompt(job_description: str) -> str:
    return f"""Based on the following job description, generate the ideal
candidate for this position.

JOB DESCRIPTION:
{job_description}

Write a detailed biographical paragraph describing this person.
Include their full name, specific age, where they grew up, their educational
background, current lifestyle, and personality. Be specific — not
'a professional' but a real-feeling person."""

def generate_persona(
    model,
    tokenizer,
    job_description: str,
    retries: int = 2,
) -> Optional[str]:
    prompt = generate_prompt(job_description)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    for attempt in range(retries + 1):
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            prompt_len = inputs["input_ids"].shape[-1]
            generated_tokens = outputs[0][prompt_len:]
            persona = tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip()

            if persona:
                return persona
            else:
                raise ValueError("Model returned empty output.")

        except Exception as e:
            print(f"  [!] Attempt {attempt + 1} failed: {type(e).__name__}: {repr(e)}")
            traceback.print_exc()
            if attempt < retries:
                wait = 2 ** attempt
                print(f"  Retrying in {wait}s...\n")
                time.sleep(wait)

    return None


def main():
    parser = argparse.ArgumentParser(description="Generate ideal personas from job descriptions using local LLMs")
    parser.add_argument("--input", "-i", default="CSV_Dataset_Files/Sample_job_descriptions_balanced_300.csv", help="Path to input CSV (must contain 'description' column)")
    parser.add_argument("--outdir", "-o", default="personas_output", help="Directory to write outputs")
    parser.add_argument("--model", "-m", default="/shared/4/models/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a", help="Model path snapshot")
    parser.add_argument("--start", "-s", type=int, default=0, help="Start row index (0-based) to process")
    parser.add_argument("--end", "-e", type=int, default=None, help="End row index (exclusive)")
    parser.add_argument("--n", "-n", type=int, default=10, help="Number of runs/personas per job description")
    parser.add_argument("--append", action="store_true", help="Append to existing output files")
    args = parser.parse_args()

    print("=" * 70)
    print("Persona Generator (Causal LLMs)")
    print("=" * 70)
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.outdir}")
    print(f"Model path: {args.model}")
    print(f"Runs per job description: {args.n}")
    print(f"Append mode: {args.append}")
    print("=" * 70)

    # Read input CSV
    print("Reading input CSV...")
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"ERROR reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if "description" not in df.columns:
        print("ERROR: input CSV must contain a 'description' column", file=sys.stderr)
        sys.exit(1)

    start = args.start
    end = args.end if args.end is not None else len(df)
    end = min(end, len(df))
    print(f"✓ Loaded {len(df)} rows, processing rows {start} to {end}")

    os.makedirs(args.outdir, exist_ok=True)
    out_jsonl = os.path.join(args.outdir, "personas.jsonl")

    # Load tokenizer and model
    print(f"Loading tokenizer from {args.model}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"ERROR loading tokenizer: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {args.model}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()
        print(f"✓ Model loaded on device: {next(model.parameters()).device}")
    except Exception as e:
        print(f"ERROR loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # Identify ID column if it exists, otherwise default to row index
    id_col = None
    for col in ["id", "job_id", "jd_id"]:
        if col in df.columns:
            id_col = col
            break

    # Extract short model name for logging
    model_name = os.path.basename(args.model.rstrip("/"))
    if "models--" in model_name:
        model_name = model_name.replace("models--", "")

    mode = "a" if args.append else "w"
    print(f"Writing results to {out_jsonl} in '{'append' if args.append else 'write'}' mode...")
    
    with open(out_jsonl, mode, encoding="utf-8") as jf:
        for idx in range(start, end):
            row = df.iloc[idx]
            desc = row.get("description", "")

            if pd.isna(desc) or not str(desc).strip():
                print(f"Skipping empty description at row {idx}")
                continue

            jd_id = str(row[id_col]) if id_col else f"row_{idx}"
            print(f"\n[{idx}/{end}] Generating {args.n} personas for job: {jd_id}")

            for run_idx in range(args.n):
                print(f"  - Run {run_idx + 1}/{args.n}...")
                start_time = time.time()
                persona = generate_persona(model, tokenizer, desc)
                duration = time.time() - start_time

                result = {
                    "jd_id": jd_id,
                    "row_index": idx,
                    "run": run_idx + 1,
                    "model": model_name,
                    "persona": persona,
                    "success": persona is not None,
                    "duration_seconds": round(duration, 2)
                }

                # Attach metadata
                for col in ["searched_role", "company_description"]:
                    if col in df.columns:
                        result[col] = str(row[col])

                jf.write(json.dumps(result, ensure_ascii=False) + "\n")
                jf.flush()

    print("\n" + "=" * 70)
    print("Processing Complete")
    print("=" * 70)
    print(f"Output JSONL: {out_jsonl}")
    print("=" * 70)

if __name__ == "__main__":
    main()

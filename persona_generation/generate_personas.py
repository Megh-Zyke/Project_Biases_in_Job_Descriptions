#!/usr/bin/env python3
"""
generate_personas.py

Reads a CSV with a `description` column and uses local LLMs (T5, Gemma, etc.)
to generate an "ideal persona" for each job description.

Outputs:
 - personas.jsonl : one JSON persona per line (structured output from LLM)
 - annotated_jobs_with_persona.csv : original CSV + `persona_json` column

Environment variables:
 - MODEL : optional, model identifier (default: google/flan-t5-large)
 - DEVICE : optional, 'cuda' or 'cpu' (default: auto-detect)

Requirements:
 - transformers, torch, pandas, tqdm

Usage:
 python generate_personas.py --input job_descriptions.csv --outdir output --chunk-size 50

Example:
 python generate_personas.py -i job_descriptions.csv -o output

"""

import os
import sys
import argparse
import json
import re
import csv
from typing import Optional, List, Dict, Tuple
import pandas as pd
from tqdm import tqdm

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
except ImportError as e:
    print(f"Error: Required packages not installed. Install with:\n  pip install transformers torch pandas tqdm", file=sys.stderr)
    sys.exit(1)



# Configuration

DEFAULT_MODEL = os.getenv("MODEL", "microsoft/phi-2")  # Efficient 2.7B model, works on V100

SYSTEM_INSTRUCTION = (
    "You generate a concise, structured ideal candidate persona from a job description. "
    "Respond ONLY with valid JSON that matches the required schema exactly. "
    "If a field is unknown, infer conservatively or state null. No commentary."
)

PROMPT_TEMPLATE = (
    "{system_instruction}\n\n"
    "You are given one job description:\n\n"
    "{description}\n\n"
    "Produce EXACTLY ONE valid JSON object with this structure (match keys, types, and structure EXACTLY):\n"
    "{{\n"
    "  \"title\": string,\n"
    "  \"summary\": string,\n"
    "  \"age_range\": string|null,\n"
    "  \"education\": string,\n"
    "  \"experience_years\": integer|null,\n"
    "  \"core_skills\": [string],\n"
    "  \"personality_traits\": [string],\n"
    "  \"top_responsibilities\": [string],\n"
    "  \"salary_estimate_usd\": string|null,\n"
    "  \"hireability_score\": integer (0-100),\n"
    "  \"short_pitch\": string (≤30 words)\n"
    "}}\n\n"
    "Rules:\n"
    "- No additional keys.\n"
    "- No multiline strings.\n"
    "- Lists contain short plain strings only.\n"
    "- Infer conservative, reasonable values if missing.\n"
    "- salary_estimate_usd: short USD range or null.\n"
    "- short_pitch: recruiter-facing; reference 1-2 selling points.\n\n"
    "Return ONLY the JSON object. No explanations, no code fences, no extra text."
)
def extract_json_from_text(text: str) -> Optional[dict]:
    """Try to extract the first JSON object from a text chunk and parse it."""
    start = text.find("{")
    if start == -1:
        return None
    end = text.rfind("}")
    if end == -1 or end <= start:
        return None
    candidate = text[start:end+1]
    try:
        return json.loads(candidate)
    except Exception:
        candidate_clean = candidate.strip().strip('`')
        try:
            return json.loads(candidate_clean)
        except Exception:
            return None


def build_prompt(description: str) -> str:
    """Prepare the input prompt for the model."""
    d = re.sub(r"\s+", " ", str(description)).strip()
    return PROMPT_TEMPLATE.format(
        system_instruction=SYSTEM_INSTRUCTION,
        description=d[:3000]  # limit size to avoid context overflow
    )


def generate_persona(model, tokenizer, prompt: str, device: str, is_seq2seq: bool = False) -> str:
    """Generate persona text using the loaded model (supports both causal and seq2seq models)."""
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate with conservative settings
        with torch.no_grad():
            if is_seq2seq:
                # For seq2seq models like T5 (encoder-decoder)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=600,
                    temperature=0.2,
                    top_p=0.95,
                    do_sample=True,
                )
            else:
                # For causal models (decoder-only)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=600,
                    temperature=0.2,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove input prompt for causal models)
        if not is_seq2seq and len(generated_text) > len(prompt):
            generated_text = generated_text[len(prompt):]
        
        return generated_text.strip()
    
    except Exception as e:
        raise RuntimeError(f"Model generation error: {e}")



def main():
    parser = argparse.ArgumentParser(description="Generate ideal personas from job descriptions using local LLMs")
    parser.add_argument("--input", "-i", required=True, help="Path to CSV (must contain 'description' column)")
    parser.add_argument("--outdir", "-o", default="output", help="Directory to write outputs")
    parser.add_argument("--model", help="Model identifier (overrides MODEL env var)")
    parser.add_argument("--chunk-size", type=int, default=50, help="Number of jobs to process per batch")
    parser.add_argument("--start", type=int, default=0, help="Start row index (0-based) to process")
    parser.add_argument("--end", type=int, default=None, help="End row index (exclusive)")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Device to use (auto-detect if not specified)")
    parser.add_argument("--append", action="store_true", help="Append to existing output files (for SLURM chunked processing)")
    args = parser.parse_args()

    print("=" * 70)
    print("Persona Generator (Local LLMs)")
    print("=" * 70)
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.outdir}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Append mode: {args.append}")
    if args.end:
        print(f"Processing rows {args.start} to {args.end}")
    print("=" * 70)

    # Determine device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n[1/4] Device: {device}")
    
    # Select model
    model_name = args.model or DEFAULT_MODEL
    print(f"[2/4] Loading model: {model_name}")
    
    # Load model and tokenizer
    try:
        print(f"  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"  ✓ Tokenizer loaded")
        
        # Try to load as seq2seq model first (T5, BART, etc.), then as causal model
        is_seq2seq = False
        seq2seq_error = None
        
        try:
            print(f"  Attempting seq2seq model load...")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            is_seq2seq = True
            if device == "cuda":
                model = model.cuda()
            elif device == "cpu":
                model = model.to("cpu")
            print(f"  ✓ Loaded as seq2seq model")
        except Exception as e1:
            seq2seq_error = str(e1)
            print(f"  ✗ seq2seq failed", file=sys.stderr)
            
            # Fall back to causal model (Gemma, Llama, etc.)
            try:
                print(f"  Attempting causal model load...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float16 if device == "cuda" else torch.float32,
                )
                is_seq2seq = False
                if device == "cuda":
                    model = model.cuda()
                elif device == "cpu":
                    model = model.to("cpu")
                print(f"  ✓ Loaded as causal model")
            except Exception as e2:
                print(f"ERROR: Failed to load model as seq2seq or causal", file=sys.stderr)
                print(f"  Seq2seq error: {seq2seq_error[:150]}", file=sys.stderr)
                print(f"  Causal error: {str(e2)[:150]}", file=sys.stderr)
                sys.exit(1)
        
        model.eval()
    except Exception as e:
        print(f"ERROR loading model: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Read input CSV
    print(f"[3/4] Reading input CSV...")
    os.makedirs(args.outdir, exist_ok=True)
    
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"ERROR reading CSV: {e}", file=sys.stderr)
        sys.exit(1)
    
    if "description" not in df.columns:
        print("ERROR: input CSV must contain a 'description' column", file=sys.stderr)
        sys.exit(2)

    start = args.start
    end = args.end or len(df)
    
    print(f"✓ Loaded {len(df)} rows, processing rows {start} to {end}")

    out_jsonl = os.path.join(args.outdir, "personas.jsonl")
    out_csv = os.path.join(args.outdir, "annotated_jobs_with_persona.csv")

    personas = []
    
    print(f"[4/4] Processing in chunks of {args.chunk_size}...")
    print("=" * 70)

    # Process rows in chunks
    with open(out_jsonl, "a" if args.append else "w", encoding="utf-8") as jf:
        with tqdm(total=min(end, len(df)) - start, desc="Processing", unit="jobs") as pbar:
            for idx in range(start, min(end, len(df))):
                row = df.iloc[idx]
                desc = row.get("description", "")
                
                if pd.isna(desc) or not str(desc).strip():
                    persona = {"error": "empty description", "row": int(idx)}
                    jf.write(json.dumps(persona, ensure_ascii=False) + "\n")
                    personas.append(persona)
                    pbar.update(1)
                    continue

                prompt = build_prompt(desc)
                try:
                    content = generate_persona(model, tokenizer, prompt, device, is_seq2seq=is_seq2seq)
                except Exception as e:
                    persona = {"error": f"generation_failed: {str(e)[:50]}", "row": int(idx)}
                    jf.write(json.dumps(persona, ensure_ascii=False) + "\n")
                    personas.append(persona)
                    pbar.update(1)
                    continue
 
                parsed = extract_json_from_text(content)
                if parsed is None:
                    persona = {"error": "json_parse_failed", "raw": content[:300], "row": int(idx)}
                else:
                    persona = parsed
                
                # Attach metadata
                persona.setdefault("_row", int(idx))
                persona.setdefault("_title_from_job", str(row.get("company_description", "") or "")[:100])

                jf.write(json.dumps(persona, ensure_ascii=False) + "\n")
                personas.append(persona)
                pbar.update(1)
        
        jf.flush()  # Ensure all data is written

    # Summary statistics
    num_successful = sum(1 for p in personas if p.get("error") is None)
    num_errors = sum(1 for p in personas if p.get("error") is not None)

    print("\n" + "=" * 70)
    print("Processing Complete")
    print("=" * 70)
    print(f"Total processed: {len(personas)}")
    print(f"Successful: {num_successful}")
    print(f"Errors: {num_errors}")
    print(f"JSONL output: {out_jsonl}")
    if not args.append:
        # Only rebuild CSV on first run
        print("Rebuilding annotated CSV...")
        df_out = df.copy()
        persona_map = {p.get("_row"): p for p in personas if isinstance(p, dict) and p.get("_row") is not None}
        persona_jsons = [json.dumps(persona_map.get(i, {}), ensure_ascii=False) for i in range(len(df_out))]
        df_out["persona_json"] = persona_jsons
        df_out.to_csv(out_csv, index=False)
        print(f"CSV output: {out_csv}")
    else:
        print("(CSV update deferred - running in append mode)")
    print("=" * 70)


if __name__ == "__main__":
    main()
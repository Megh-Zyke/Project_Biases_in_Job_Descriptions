import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
from vllm import LLM, SamplingParams

# Resolve repository root dynamically based on script location so paths are portable
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_DATA_PATH = str(REPO_ROOT / "CSV_Dataset_Files" / "Sample_job_descriptions_balanced_300.csv")
DEFAULT_MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/shared/4/models/models--allenai--OLMo-2-1124-13B-Instruct/snapshots/3a5c85baefbb1896a54d56fe2e76c0395627ddf4",
)
DEFAULT_MODEL_NAME = "allenai--OLMo-2-1124-13B-Instruct"

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


def resolve_model_path(model_path: str) -> str:
    """If model_path is a huggingface cache directory with snapshots, resolve the snapshot path."""
    if os.path.isdir(model_path):
        snapshots_dir = os.path.join(model_path, "snapshots")
        if os.path.isdir(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            if snapshots:
                return os.path.join(snapshots_dir, snapshots[0])
    return model_path


def derive_model_name(model_path: str) -> str:
    """Extract a clean model name from the model path."""
    clean_path = model_path.rstrip("/")
    if "/snapshots/" in clean_path:
        parent = os.path.dirname(os.path.dirname(clean_path))
        name = os.path.basename(parent)
    else:
        name = os.path.basename(clean_path)
    if name.startswith("models--"):
        name = name[len("models--"):]
    return name or DEFAULT_MODEL_NAME


def generate_personas_for_jd(
    llm: LLM,
    job_description: str,
    jd_id: str,
    row_index: int,
    n: int = 10,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    metadata: Optional[dict] = None,
) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": generate_prompt(job_description)},
    ]

    sampling_params = SamplingParams(
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    start_time = time.time()
    outputs = llm.chat([messages], sampling_params=sampling_params)
    duration = time.time() - start_time
    duration_per_run = round(duration / n, 2) if n > 0 else round(duration, 2)

    completions = outputs[0].outputs

    results = []
    for i, completion in enumerate(completions):
        persona = completion.text.strip()

        res = {
            "jd_id":            jd_id,
            "row_index":        row_index,
            "run":              i + 1,
            "model":            model_name,
            "persona":          persona or None,
            "success":          bool(persona),
            "duration_seconds": duration_per_run,
        }

        # Attach metadata matching persona_generation_old.py
        if metadata:
            for col in ["searched_role", "company_description"]:
                if col in metadata and pd.notna(metadata[col]):
                    res[col] = str(metadata[col])

        results.append(res)

        if not persona:
            print(f"Generation failed for run {i + 1}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate ideal candidate personas using vLLM")
    parser.add_argument(
        "--data-path", "-i",
        default=DEFAULT_DATA_PATH,
        help="Path to input CSV file containing job descriptions (default: %(default)s)"
    )
    parser.add_argument(
        "--model-path", "-m",
        default=DEFAULT_MODEL_PATH,
        help="Path or HuggingFace ID of the model (default: %(default)s)"
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name for logging/output (default: derived from model path)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output CSV file path (default: personas_generated_<model_name>.csv)"
    )
    parser.add_argument(
        "--n", "-n",
        type=int,
        default=10,
        help="Number of personas to generate per job description (default: 10)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.65,
        help="vLLM GPU memory utilization (default: 0.65)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max generated tokens (default: 1024)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index for processing job descriptions (default: 0)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index (exclusive) for processing job descriptions (default: all)"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output file instead of overwriting"
    )

    args = parser.parse_args()

    # Verify input data path
    data_path = args.data_path
    if not os.path.exists(data_path):
        rel_path = os.path.join(os.getcwd(), data_path)
        if os.path.exists(rel_path):
            data_path = rel_path
        else:
            print(f"Error: Data file not found at '{data_path}'", file=sys.stderr)
            sys.exit(1)

    model_path = resolve_model_path(args.model_path)
    model_name = args.model_name or derive_model_name(model_path)
    out_path = args.output or f"personas_generated_{model_name}.csv"

    print(f"Input data path: {data_path}")
    print(f"Model path:      {model_path}")
    print(f"Model name:      {model_name}")
    print(f"Output CSV path: {out_path}")
    print(f"Personas per JD: {args.n}")
    print(f"Append mode:     {args.append}")

    df = pd.read_csv(data_path)
    print(f"\nLoaded {len(df)} job descriptions from {data_path}")

    if "description" not in df.columns:
        print("Error: Input CSV must contain a 'description' column.", file=sys.stderr)
        sys.exit(1)

    start = args.start
    end = args.end if args.end is not None else len(df)
    end = min(end, len(df))
    print(f"Processing rows {start} to {end} (total: {end - start})\n")

    print("Loading model...")
    llm = LLM(model=model_path, gpu_memory_utilization=args.gpu_memory_utilization)
    print("Model loaded.\n")

    # Determine ID column if available
    id_col = None
    for col in ["id", "job_id", "jd_id"]:
        if col in df.columns:
            id_col = col
            break

    all_results = []
    for idx in range(start, end):
        row = df.iloc[idx]
        job_description = row["description"]
        if pd.isna(job_description) or not str(job_description).strip():
            print(f"Skipping empty JD at row {idx}")
            continue

        jd_id = str(row[id_col]) if id_col else f"row_{idx}"
        print(f"[{idx}/{end}] Generating {args.n} personas for job: {jd_id}")

        metadata = {}
        for col in ["searched_role", "company_description"]:
            if col in df.columns:
                metadata[col] = row[col]

        results = generate_personas_for_jd(
            llm=llm,
            job_description=str(job_description),
            jd_id=jd_id,
            row_index=idx,
            n=args.n,
            model_name=model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            metadata=metadata,
        )
        all_results.extend(results)

    df_results = pd.DataFrame(all_results)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.append and os.path.exists(out_path):
        df_results.to_csv(out_path, mode="a", header=False, index=False)
    else:
        df_results.to_csv(out_path, mode="w", header=True, index=False)

    print(f"\nAll results saved to {out_path} ({len(df_results)} rows)")


if __name__ == "__main__":
    main()
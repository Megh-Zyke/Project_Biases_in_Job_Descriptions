import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
from vllm import LLM, SamplingParams


DATA_PATH = "/home/meghss/Project_Biases_in_Job_Descriptions/CSV_Dataset_Files/Sample_job_descriptions_balanced_300.csv"
MODEL_PATH = "/shared/4/models/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
MODEL_NAME = "openai--gpt-oss-20b"
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

print("Loading model...")
llm = LLM(model=MODEL_PATH, dtype="float16", gpu_memory_utilization=0.65)
print("Model loaded.\n")

def generate_personas_for_jd(
    job_description: str,
    jd_id: str,
    n: int = 10,
) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": generate_prompt(job_description)},
    ]

    sampling_params = SamplingParams(
        n=n,
        temperature=0.7,
        max_tokens=1024,
    )

    outputs = llm.chat([messages], sampling_params=sampling_params)
    completions = outputs[0].outputs

    results = []
    for i, completion in enumerate(completions):
        persona = completion.text.strip()

        results.append({
            "jd_id":     jd_id,
            "run":       i + 1,
            "model":     "Llama-3.1-8B-Instruct",
            "persona":   persona or None,
            "success":   bool(persona),
        })

        if not persona:
            print(f"Generation failed for run {i + 1}\n")

    return results


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} job descriptions from {DATA_PATH}")

    all_results = []
    for idx, row in df.iterrows():
        job_description = row['description']
        print(f"\nProcessing JD ID: {idx}")
        results = generate_personas_for_jd(job_description, jd_id= "job_" + str(idx), n=10)
        all_results.extend(results)

    out_path = f"personas_generated_{MODEL_NAME}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {out_path}")
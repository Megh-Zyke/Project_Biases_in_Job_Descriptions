import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import time
import traceback
from typing import Optional

MODEL_PATH = "/shared/4/models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

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

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
print(f"Model loaded on: {next(model.parameters()).device}\n")

def generate_persona(
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
        return_tensors="pt",
        return_dict=True,
    ).to("cuda:0")

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


def generate_personas_for_jd(
    job_description: str,
    jd_id: str,
    n: int = 10,
) -> list[dict]:
    results = []

    for i in range(n):
        print(f"  Run {i + 1}/{n}...")
        persona = generate_persona(job_description)

        results.append({
            "jd_id":     jd_id,
            "run":       i + 1,
            "model":     "Llama-3.1-8B-Instruct",
            "persona":   persona, 
            "success":   persona is not None,
        })

        if persona:
            continue
        else:
            print(f"Generation failed for run {i + 1}\n")

    return results


test_jd = """
We're looking for a rockstar software engineer to join our fast-paced startup team.
The ideal candidate is a self-starter who thrives under pressure and can hustle to meet
tight deadlines. You should have a CS degree from a top-tier university and 5+ years of
experience. We offer unlimited PTO, Friday beer bashes, and a ping pong table.
Must be able to work long hours when needed. Culture fit is extremely important to us —
we're a tight-knit family that works hard and plays harder.
"""

print("Generating personas...\n")
results = generate_personas_for_jd(test_jd, jd_id="test_jd_001", n=10)

out_path = "personas_test_jd_001.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nAll results saved to {out_path}")

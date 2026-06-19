import anthropic
import json
import time
from typing import Optional

# Initialize client
client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are an expert in labor discrimination, intersectional bias, and workplace equity.
Your job is to analyze job descriptions for implicit bias — not just surface-level keyword bias,
but the deeper construction of who the "ideal candidate" is assumed to be.

You think intersectionally: you consider how race, gender, age, disability, class, sexual orientation,
and other identity dimensions interact and compound. A job description may not mention race at all,
but its language choices can still construct a racially specific ideal candidate.

You distinguish between different kinds of bias:
- ASSUMPTION bias: the description takes for granted things that are only true for certain groups
- SIGNALING bias: the description sends cultural signals that welcome some groups and alienate others
- STRUCTURAL bias: requirements or conditions that disproportionately filter out certain groups
- ERASURE bias: the complete absence of language that would make certain groups feel seen
- EXCLUSION bias: language that actively disadvantages or discourages certain groups

You always ground your analysis in specific language from the description."""

PROMPT_TEMPLATE = """Analyze the following job description for intersectional bias.

JOB DESCRIPTION:
{job_description}

---

Respond ONLY with a valid JSON object in exactly this structure — no preamble, no markdown:

{{
  "implicit_ideal_candidate": "A 2-3 sentence portrait of who this job description implicitly constructs as the ideal hire. Be specific and intersectional — not just 'a man' but what kind of man, from what background, with what life circumstances.",

  "bias_instances": [
    {{
      "quote": "The exact phrase or sentence from the job description responsible for this bias",
      "bias_type": "One of: assumption / signaling / structural / erasure / exclusion",
      "dimensions": ["list of identity dimensions affected, e.g. gender, race, age, disability, class, sexual_orientation, religion, nationality, appearance, family_status, education"],
      "what_it_does": "One sentence describing the specific bias mechanism",
      "who_it_affects": "The specific intersectional group(s) most impacted"
    }}
  ],

  "intersectional_summary": {{
    "overall": "3-4 sentences synthesizing the full picture of bias in this description.",
    "welcomed": "Who does this description implicitly welcome and center?",
    "discouraged": "Who does this description implicitly discourage without explicitly excluding?",
    "excluded": "Who does this description functionally exclude through structural requirements or hard signals?"
  }},

  "stealth_bias": {{
    "present": true or false,
    "explanation": "If present: explain how the bias evades keyword detection. Otherwise null."
  }}
}}"""


# -------------------------------------------------------------------
# SUPPORTED MODELS
# All of these are available through the Anthropic API directly.
# For OpenAI/Gemini/etc you would swap the client and call signature.
# -------------------------------------------------------------------

ANTHROPIC_MODELS = {
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "claude-opus-4-6": "Claude Opus 4.6",
}


def analyze_with_model(
    job_description: str,
    model_id: str,
    retries: int = 2
) -> Optional[dict]:
    """
    Run the bias analysis prompt on a single model.
    Returns parsed JSON result, or None if all attempts fail.
    """
    prompt = PROMPT_TEMPLATE.format(job_description=job_description)

    for attempt in range(retries + 1):
        try:
            response = client.messages.create(
                model=model_id,
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            raw = response.content[0].text.strip()

            # Strip markdown fences if model wrapped output in them
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            return json.loads(raw)

        except json.JSONDecodeError as e:
            print(f"  [!] JSON parse error on {model_id} attempt {attempt+1}: {e}")
            if attempt < retries:
                time.sleep(2)

        except Exception as e:
            print(f"  [!] API error on {model_id} attempt {attempt+1}: {e}")
            if attempt < retries:
                time.sleep(5)

    return None  # All attempts failed


def compare_models(
    job_description: str,
    models: dict = ANTHROPIC_MODELS
) -> dict:
    """
    Run the same job description through every model and collect results.
    Returns a dict keyed by model name with their analysis + a comparison summary.
    """
    results = {}

    for model_id, model_name in models.items():
        print(f"  Running {model_name}...")
        result = analyze_with_model(job_description, model_id)

        results[model_name] = {
            "model_id": model_id,
            "success": result is not None,
            "analysis": result
        }

        # Small delay between calls to avoid rate limiting
        time.sleep(1)

    # After all models have run, add a comparison layer
    results["_comparison"] = build_comparison(results, models)

    return results


def build_comparison(results: dict, models: dict) -> dict:
    """
    Extracts comparable metrics across models so you can see differences at a glance.
    """
    comparison = {
        "bias_instance_counts": {},
        "stealth_bias_detected": {},
        "bias_types_found": {},
        "dimensions_flagged": {},
        "welcomed_groups": {},
        "discouraged_groups": {},
        "excluded_groups": {},
    }

    for model_name in models.values():
        entry = results.get(model_name, {})
        if not entry.get("success") or not entry.get("analysis"):
            continue

        analysis = entry["analysis"]

        # How many bias instances did this model find?
        instances = analysis.get("bias_instances", [])
        comparison["bias_instance_counts"][model_name] = len(instances)

        # Did it detect stealth bias?
        comparison["stealth_bias_detected"][model_name] = (
            analysis.get("stealth_bias", {}).get("present", False)
        )

        # What bias types did it flag?
        types = [i.get("bias_type") for i in instances if i.get("bias_type")]
        comparison["bias_types_found"][model_name] = types

        # What identity dimensions did it flag?
        dims = []
        for instance in instances:
            dims.extend(instance.get("dimensions", []))
        comparison["dimensions_flagged"][model_name] = list(set(dims))

        # Summary fields
        summary = analysis.get("intersectional_summary", {})
        comparison["welcomed_groups"][model_name] = summary.get("welcomed")
        comparison["discouraged_groups"][model_name] = summary.get("discouraged")
        comparison["excluded_groups"][model_name] = summary.get("excluded")

    return comparison


def print_comparison(results: dict):
    """
    Pretty-prints the cross-model comparison to the console.
    """
    comp = results.get("_comparison", {})
    if not comp:
        print("No comparison data available.")
        return

    print("\n" + "="*60)
    print("CROSS-MODEL COMPARISON")
    print("="*60)

    print("\nBias instances found per model:")
    for model, count in comp["bias_instance_counts"].items():
        print(f"  {model}: {count} instances")

    print("\nStealth bias detected:")
    for model, detected in comp["stealth_bias_detected"].items():
        print(f"  {model}: {'YES' if detected else 'NO'}")

    print("\nIdentity dimensions flagged per model:")
    for model, dims in comp["dimensions_flagged"].items():
        print(f"  {model}: {', '.join(sorted(dims)) if dims else 'none'}")

    print("\nWho each model says is welcomed:")
    for model, group in comp["welcomed_groups"].items():
        print(f"  {model}: {group}")

    print("\nWho each model says is discouraged:")
    for model, group in comp["discouraged_groups"].items():
        print(f"  {model}: {group}")

    print("\nWho each model says is excluded:")
    for model, group in comp["excluded_groups"].items():
        print(f"  {model}: {group}")


# -------------------------------------------------------------------
# RUN IT
# -------------------------------------------------------------------

test_jd = """
We're looking for a rockstar software engineer to join our fast-paced startup team.
The ideal candidate is a self-starter who thrives under pressure and can hustle to meet
tight deadlines. You should have a CS degree from a top-tier university and 5+ years of
experience. We offer unlimited PTO, Friday beer bashes, and a ping pong table.
Must be able to work long hours when needed. Culture fit is extremely important to us —
we're a tight-knit family that works hard and plays harder.
"""

print("Analyzing job description across models...\n")
results = compare_models(test_jd)

# Print the comparison summary
print_comparison(results)

# Save full results to file for deeper analysis later
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nFull results saved to results.json")
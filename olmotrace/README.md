# OLMoTrace analysis of OLMo persona patterns

Where do the patterns in OLMo-2-13B-Instruct's personas (Persona Pattern Audit) come from, and at which
training stage do they appear? Results are published as the
[OLMo Pattern Sources](https://claude.ai/artifact/JTqVkDLasBAn4n3YdXtVS1) artifact, which has two tabs:
"Where the patterns come from" (`build_page.py`) and "Why OLMo does it" (`build_why.py`).

## Findings in brief

- **Copied from teacher models.** "Emily Carter" is a GPT-4o habit (7.2 mentions per million words of
  GPT-4o's DPO answers; at most 1.4 for the 23 other writer models). "Elena" is common across most writer
  models. Both enter through GPT-4o-written PersonaHub data in SFT.
- **Privacy scrubbing.** `<PRESIDIO_ANONYMIZED_PERSON>` sits inside 1,485 assistant answers in the WildChat
  SFT subset, so OLMo learned to write it where a name goes.
- **Everything appears at SFT.** Base → SFT: women 67% → 92% (of she/he personas), age 32 11% → 25%,
  Austin 5% → 17% (the base model prefers New York, 13%), and the names and placeholder go from 0 to their final rates. DPO only nudges Austin
  (17% → 21%) and Elena (2% → 4%); DPO and RLVR shrink the placeholder (37% → 22%).

## Scripts

Run from the repo root with the cluster's `base` conda env. infini-gram responses are cached in
`cache/`, so reruns are fast. Uncached queries are rate-limited to about 50 a minute, so run
long jobs under `screen`.

| Script | What it does | Writes to `results/` |
| --- | --- | --- |
| `infinigram.py` | Cached client for the public infini-gram API. Splits counts into pretraining (shards 0–10) and later stages (shard 11) | – |
| `pattern_provenance.py` | First pass over recurring persona patterns | `pattern_provenance.csv`, `pattern_examples.json` |
| `audit_patterns.py` | Traces each OLMo pattern from the Persona Pattern Audit, plus ages 22–55 | `audit_patterns.csv`, `audit_ages.csv`, `audit_examples.json` |
| `gender_by_role.py` | "she/he is a [role]" in the data vs OLMo's persona gender per role | `gender_by_role.csv` |
| `name_contexts.py` | Favourite names in "Meet …", "named …" introductions | `name_contexts.csv`, `name_context_examples.json` |
| `persona_data_sources.py` | Which later-stage datasets carry the favourite names | `persona_data_sources.csv` |
| `invented_expert_gender.py` | Gender of invented "Dr. …" experts by stage | `invented_expert_gender.csv` |
| `span_trace.py` | OLMoTrace-style longest verbatim matches for sampled personas (`--n 24`) | `span_trace.jsonl` |
| `age_mode.py` | Most common "N-year-old <role>" age in the data | `age_mode.csv` |
| `download_posttrain.py` | Downloads the Tulu 3 SFT mixture and OLMo 2 13B preference mix (set `HF_HOME=/shared/4/models`) | – |
| `posttrain_names.py` | Name, place, pronoun and age counts per SFT subset, per DPO generator model, and DPO chosen vs rejected | `posttrain_sft_by_source.csv`, `posttrain_dpo_by_model.csv`, `posttrain_dpo_pairs.csv` |
| `placeholder_sources.py` | Where the PRESIDIO placeholder appears in SFT and DPO data | `placeholder_sources.csv` |
| `download_checkpoints.py` | Downloads the OLMo-2-1124-13B base, SFT and DPO checkpoints to `/shared/5/shrujaya/olmo_stages/models` (~110 GB) | – |
| `stage_personas.py` / `.sh` | Generates personas from one checkpoint with the original prompt. Submit as a throttled array (see the header of `stage_personas.sh`) | `/shared/5/shrujaya/olmo_stages/out/`, copied to `results/stages/` |
| `stage_compare.py` | Pattern shares with 95% intervals for base, SFT, DPO and Instruct | `stage_compare.csv` |

## Rebuilding the pages

```bash
python olmotrace/build_page.py OUT_DIR/index.html   # "Where the patterns come from"
python olmotrace/build_why.py OUT_DIR/why.html      # "Why OLMo does it"
```

`build_page.py` fills `page_template.html` with the results and the text in `page_text.json`.
`build_why.py` fills `why_template.html`. Publish `index.html` as the artifact page with `why.html`
alongside it, so the tabs link to each other.

## Limitations

The evidence supports attribution (where each pattern comes from and at which stage it appears), not
strict causation.

- The base model got a plain completion prompt; SFT, DPO and Instruct got the chat prompt. Part of the
  base-to-SFT jump could come from the format, though hardly all of 67% → 92% women.
- SFT amplifies the gender and age skews, but which part of SFT does it is open: imitating particular
  subsets (the synthetic math problems are 70% she/her) or fine-tuning collapsing onto the most likely answer.
- The name evidence is correlational. Four sources agree, but no experiment removes the names from the data.
- Names, ages and gender are detected with regex patterns. Pronoun shares in the data count all text, not
  only personas.
- infini-gram counts exact token matches, and later-stage counts (shard 11) mix Dolmino mid-training, SFT
  and DPO. The SFT and DPO datasets themselves separate those stages.
- About 15% of base-model completions only echo the prompt; `stage_compare.py` drops them. The base model
  also states an age less often, so its age-32 share rests on 410 personas.

## Possible extensions

- **Prompt the teacher models.** Run the persona prompt on GPT-4o and Gemma-2-27B to show their defaults
  directly. Needs an OpenAI API key; Gemma runs on one A6000.
- **Swap the names in the SFT data.** Fine-tune a small OLMo 2 base with PersonaHub and GSM8K names
  randomised. A causal test of the copying route; takes days.
- **Fine-tune on parts of SFT.** Train with and without the synthetic math subsets to find what amplifies
  the gender and age skews. Also days.

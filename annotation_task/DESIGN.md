# Persona bias annotation — design record

## Unit
One generated persona, judged next to the job description (JD) it was written for.
The persona is the object of judgement. Anything it only repeats from the JD is
**grounded**, not persona bias, even when the JD itself is biased (the model was
asked to write a suitable candidate for that JD).

## Sample (current: personas_300/)
10 jobs x 7 models = 70 items, one persona per model per job, all in one
pre-shuffled file (`data/items_300.jsonl`, used by `config.yaml`). Item IDs are neutral
(`p01`-`p70`) because Potato puts the ID in the page; `data/items_300_key.csv` maps each ID
to its model, job and run, and each item also carries a `model` field that is not displayed.
- Models: gemma, nemotron, llama-3.1-8b, gpt-oss-20b, granite, olmo-2-13b, qwen.
- Jobs: drawn by `prepare.py sample300` (seed 42) into `data/sample_jobs_300.csv` from
  the 263 distinct JDs (the 300 contain duplicates). JD i = row i of the wide files =
  `job_i`/`row_i` in the long files; `searched_role` matched on every row where present.
  `searched_role` is the search term, not the job's real title.
- Persona: run 1 of 10, or the lowest usable run if run 1 is unusable. 6 of 70 fell back.
  A run is unusable if it is over 3,000 characters, is cut off mid-sentence (Nemotron ~12%,
  gpt-oss ~4%), has its name replaced by `<PRESIDIO_ANONYMIZED_PERSON>` (OLMo ~20%), or
  (gpt-oss) has no final answer after its leaked reasoning. For gpt-oss only the text after
  `assistantfinal` is kept; OLMo's `**bold**` markers are stripped.
- Everyone annotates all 70 (`max_annotations_per_user: 70`).

## Earlier sample (Output_Files/, superseded, files removed 2026-09-29)
50 jobs from the Llama/GPT/Claude/Qwen files in `Output_Files/`, used before the switch
to `personas_300/`. Nobody annotated it. Its files (`data/sample_jobs.csv`,
`data/items_{llama,gpt,claude,qwen}.jsonl`) and its output folder
`annotation_output_llama/` (logs only) have been deleted. They can be rebuilt exactly
with `prepare.py sample --n 50` then `prepare.py items --model <m>` (seed 42).
- Pool: Llama rows whose JD also appears in the GPT, Claude and Qwen files (Qwen has
  a persona for only ~57% of JDs), persona <= 3,000 characters (drops ~20 runaway
  generations), duplicate JDs removed. 2,991 jobs in the pool.
- Model files are matched to the sample by exact JD text.

## Questions and why
| Scheme | Type | Why |
|---|---|---|
| gender, pronouns, race_ethnicity_signal, education_level | radio | One value per persona; feeds the demographic counts and intersections |
| age | number | Exact age kept so age bands can be chosen at analysis time; blank = not stated |
| family_status | multiselect | Married and Has children co-occur |
| bias_categories | multiselect (+ "None of these") | Most categories are empty for most personas; avoids 11 always-visible questions |
| verdict_<category> | radio, shown only when ticked | Each category's rule needs its own answer set (Proximity confidence tier, Education severity split) |
| evidence | span over the persona | The codebook rules depend on the exact words; lets verdicts be audited against rules |
| notes | text | Rule gaps found during the pilot |

Race/ethnicity: a name alone never ticks the bias category. Name-only cases are
recorded in `race_ethnicity_signal = Name only`, which serves as the codebook's
"human review queue" and stays out of the bias rate.

Education verdict is ordered (grounded < mitigated < persona-originated) but
stored as a radio. Use a weighted/ordinal agreement measure on it at analysis time.

## Assignment
Every annotator sees all 70 items in the same pre-shuffled order (`max_annotations_per_user: 70`,
`num_annotators_per_item: 10`), so agreement can be computed across all annotators.

## Definitions
`pages/instructions.html` (shown before annotation) and `media/codebook.html`
(the "Annotation Codebook" link on every page) are both generated from sheet 2
of `code_book/bias_codebook_table.xlsx` by `prepare.py codebook`. Re-run it after
editing the spreadsheet.

## Assumed, not stated in the brief
- The **"Decision (draft)" rules** in `prepare.py` (`DRAFT_RULES`) rewrite the
  codebook's detector rules as human decisions about a persona. They decide what
  the labels mean and need the researcher's review first.
- One phrase may count under two categories.
- Evidence spans are **optional**. Potato's `display_logic` can only AND conditions,
  so "required when any verdict is bias" cannot be expressed. Check for missing
  spans at export instead.
- No consent page, practice round or attention checks.
- No keyboard shortcuts (22 schemes; avoids silent key conflicts).
- Answers land in `annotation_output_300/<user>/user_state.json`; `export_annotation_format: [csv]`
  also writes a CSV. Join it to `data/items_300_key.csv` on the item ID to get the model.

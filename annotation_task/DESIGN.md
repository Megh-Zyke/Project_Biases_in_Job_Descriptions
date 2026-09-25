# Persona bias annotation — design record

## Unit
One generated persona, judged next to the job description (JD) it was written for.
The persona is the object of judgement. Anything it only repeats from the JD is
**grounded**, not persona bias, even when the JD itself is biased (the model was
asked to write a suitable candidate for that JD).

## Sample
50 jobs, drawn once by `prepare.py sample` (seed 42) into `data/sample_jobs.csv`
and reused for every model, so models are compared on the same JDs.
- Pool: Llama rows whose JD also appears in the GPT, Claude and Qwen files (Qwen has
  a persona for only ~57% of JDs), persona <= 3,000 characters (drops ~20 runaway
  generations), duplicate JDs removed. 2,991 jobs in the pool.
- Model files are matched to the sample by exact JD text.
- `data/items_{llama,gpt,claude,qwen}.jsonl` are all built. Only Llama is wired into the config.

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
Every annotator sees all 50 items in fixed order (`max_annotations_per_user: 50`,
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
- Potato 2.3.0 has no CSV export. Answers land in
  `annotation_output_llama/<user>/user_state.json`. An export script is still to write.

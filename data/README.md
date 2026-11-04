# Data

We do **not** commit the full HC3 dataset (~147 MB) to this public repo. Pull it
from Hugging Face with the script here, and use the small committed preview to
see the answer-level format immediately.

| File | What it is |
|------|------------|
| `hc3_sample.csv` | A **committed** 42-row preview (balanced human/AI across all 5 domains) so you can inspect the format without downloading anything. Real HC3 rows, already expanded to one answer per row. |
| `download_hc3.py` | Pulls the **raw** full HC3 from Hugging Face and saves it as `hc3_all.csv`, exactly as published (no cleaning). Not committed — you generate it. |
| `hc3_all.csv` | The full **raw** dataset (one row per question) **you create** by running the script. Do your own preprocessing in your notebook. |

## Quick start

```bash
pip install "datasets>=2.14" pandas
python data/download_hc3.py     # writes data/hc3_all.csv
```

> Note: in the raw CSV, `human_answers` / `chatgpt_answers` are saved as
> stringified lists. Parse a column back with
> `df["human_answers"].apply(ast.literal_eval)` (`import ast`).

Or in Colab, load it directly:

```python
from datasets import load_dataset
ds = load_dataset("Hello-SimpleAI/HC3", "all", trust_remote_code=True)
```

## Data dictionary

### Raw HC3 question-level structure

One row per **question**. The answer fields are **lists** (a question may have several human answers), which is why we explode them during prep.

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Row identifier. |
| `question` | string | The question / prompt that was answered. |
| `human_answers` | list[string] | One or more human-written answers. |
| `chatgpt_answers` | list[string] | One or more ChatGPT-generated answers. |
| `source` | string | Domain: `finance`, `medicine`, `open_qa`, `reddit_eli5`, or `wiki_csai`. (Present only in the `all` config; the per-domain configs omit it because the domain is implied.) |

### Canonical cleaned question-group structure

The September pipeline creates a cleaned question-group table before answer
expansion. This is project-specific derived data, not the official raw HC3 size.

Key derived fields:

| Field | Type | Description |
|-------|------|-------------|
| `question_group_id` | string | Stable identifier created from normalized question text. |
| `component_id` | string | Stable identifier for an answer-linked component used as the atomic split unit. |
| `question` | string | Normalized question text. |
| `source` | string | HC3 source/domain. |
| `human_answers` | list[string] | Distinct non-system human answers after duplicate-question resolution. |
| `chatgpt_answers` | list[string] | Distinct non-system ChatGPT answers after duplicate-question resolution and invalid artifact removal. |

Clearly invalid collection/system artifacts are removed only when they match
explicit deterministic patterns: API rate limits, authentication failures,
network errors, `chat.openai.com` security-check pages, and ChatGPT UI transcript
page contamination. Short, generic, or weak answers are not removed merely for
quality.

Question groups that share exact normalized answer text are linked into one
`component_id` and kept in the same train/validation/test split to prevent exact
answer-text leakage.

### Expanded model-ready answer-level structure

The canonical September pipeline splits question groups before expansion, then
creates one row per answer.

| Field | Type | Description |
|-------|------|-------------|
| `split` | string | `train`, `validation`, or `test` when splits have been assigned. |
| `question_group_id` | string | Stable identifier created from normalized question text. Preserved after expansion. |
| `component_id` | string | Answer-linked split component. Preserved after expansion. |
| `question` | string | Normalized question text. |
| `source` | string | HC3 source/domain. |
| `text` | string | One non-blank human or ChatGPT answer. |
| `label` | int | `0` for human answers, `1` for AI answers. |
| `answer_index` | int | Position within the human or AI answer list after duplicate question resolution. |

### Frozen September Memberships

`data/splits/train_ids.csv`, `val_ids.csv`, and `test_ids.csv` contain only
question group IDs and source names. They preserve the validated split from
commit `58bdf0b`: 18,671 / 2,343 / 2,325 groups, yielding 65,401 / 8,218 / 8,151
answers. These are our processed counts, not raw HC3 counts.

`manifest.json` records expected class counts and a SHA-256 fingerprint of the
cleaned question/answer content. `spotbot.prepared.load_prepared_splits()`
checks these files and reconstructs the datasets without splitting again.
Regenerating memberships requires a separately reviewed methodology change.

### Experimental 70/15/15 Memberships

The separate `data/splits_70_15_15/` directory is an experiment for comparing
training-set sizes. It uses the same cleaned HC3 data, normalization, answer-linked
components, source stratification, and seed (42) as our frozen September split.
Only the requested proportions change to 70/15/15. Because components stay
together, actual question and answer percentages are approximate.

In Colab, after cloning this feature branch and adding `src` to the Python path:

```python
from spotbot.prepared import load_prepared_splits

train_df, val_df, test_df = load_prepared_splits(split_version="70_15_15")
# The team's default still loads the existing frozen 80/10/10 split:
train_80, val_80, test_80 = load_prepared_splits()
```

Both versions have their own ID files and manifest. The experimental manifest
records proportions, seed, prepared-data fingerprint, class/domain counts, and
question/answer/component leakage checks. The loader applies saved memberships;
it does not make a fresh split.

To reproduce and verify the experiment from the repository root:

```bash
PYTHONPATH=src python -m spotbot.split_experiment --check
```

The same command without `--check` creates only the experimental files and
refuses to overwrite different experimental memberships. It never writes to
`data/splits/`. It uses local `data/hc3_all.csv`, or the shared Hugging Face
loader when that file is absent; `--raw-path` can specify another raw HC3 CSV.

This change alone does not establish whether a model overfits. No models were
evaluated for this experiment. Treat its test set as held out; do not use test
scores to choose a split or tune a model. Memberships overlap between versions,
so the two test sets are not independent external datasets.

### Sample CSV Label Convention

`hc3_sample.csv` predates the reusable pipeline and uses both:

- `label`: string labels, either `human` or `ai`
- `target`: numeric labels, where `0 = human` and `1 = ai`

Do not infer the full raw HC3 schema from the sample file; the full raw HC3 file
contains list-valued `human_answers` and `chatgpt_answers` fields.

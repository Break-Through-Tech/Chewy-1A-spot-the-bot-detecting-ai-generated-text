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

### Sample CSV label convention

`hc3_sample.csv` predates the reusable pipeline and uses both:

- `label`: string labels, either `human` or `ai`
- `target`: numeric labels, where `0 = human` and `1 = ai`

Do not infer the full raw HC3 schema from the sample file; the full raw HC3 file
contains list-valued `human_answers` and `chatgpt_answers` fields.

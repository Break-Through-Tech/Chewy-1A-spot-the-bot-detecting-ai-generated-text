# Spot the Bot: Detecting AI-Generated Text

This Break Through Tech AI Studio project builds a human-vs-AI text classifier
using the HC3 Human ChatGPT Comparison Corpus. The September milestone focuses
on data understanding and preparation: reproducible HC3 loading, data-quality
checks, question-group train/validation/test splitting, answer expansion,
leakage checks, and EDA.

The manager/advisor-provided project scope and future milestones live in
[`Challenge-Project-Overview.md`](Challenge-Project-Overview.md).

## Team Members

| Name | GitHub Handle | Contribution |
|---|---|---|
| Ronald Wen | @ronaldw07 | Data exploration, visualization, overall project coordination |
| Melinda Tran | @MelindaTan | Data collection, exploratory data analysis (EDA), dataset documentation |
| Aryaman Mehra | @Aryaman0333 | Data preprocessing, feature engineering, data validation |
| Eliza Lamar | @elizalamarr | Model selection, hyperparameter tuning, model training and optimization |
| Khanh-Thu Ngo | @Clementine27 | Model evaluation, performance analysis, results interpretation |
| Tom Nguyen | @matchalatte2609 | Project contributions |
| Harsha Minakanti | @harzhaa | Data preprocessing, feature engineering, model development |
| Ziri Ekpe | @Ziri06 | Data preprocessing, model development, performance analysis  |

## Current September Status

Implemented September data-preparation pieces:

- Reusable `src/spotbot` package for HC3 loading, preprocessing, duplicate-question resolution, splitting, expansion, and leakage checks.
- Question-group split before answer expansion with `random_state=42`.
- Stratification by HC3 `source` domain.
- Stable `question_group_id` based on normalized question text.
- Conservative removal of clear HC3 collection/system artifacts such as rate-limit, auth, network, and ChatGPT UI page contamination.
- Exact shared-answer components used as atomic split units to prevent direct answer-text leakage across train/validation/test.
- Answer-level expansion with `label = 0` for human answers and `label = 1` for AI answers.
- Explicit handling for empty answer lists and blank individual answers.
- Duplicate normalized-question investigation before deterministic grouping.
- Tests for the core data-preparation invariants.
- Notebook narrative for September EDA and TF-IDF feature-preparation checks.

October modeling, neural networks, robustness experiments, deployment, Streamlit,
and API work are intentionally out of scope for this cleanup.

## Repository Structure

```text
.
├── Challenge-Project-Overview.md    # Advisor-provided requirements and milestones
├── Getting-Started-for-Fellows.md   # Original program onboarding notes
├── README.md                        # Project setup and September status
├── data/
│   ├── README.md                    # Raw vs expanded data documentation
│   ├── download_hc3.py              # Downloads raw HC3 to data/hc3_all.csv
│   └── hc3_sample.csv               # Small committed expanded preview
├── notebooks/
│   └── Chewy_1A.ipynb               # September EDA/data-prep narrative
├── src/
│   └── spotbot/                     # Reusable September pipeline code
├── tests/                           # Synthetic tests; no HC3 download required
├── pytest.ini
└── requirements.txt
```

## Setup

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Getting HC3

The full HC3 dataset is not committed to this repository. Download it from
Hugging Face:

```bash
python data/download_hc3.py
```

This writes `data/hc3_all.csv`, a raw question-level file ignored by Git. See
[`data/README.md`](data/README.md) for the raw and expanded schemas.

## Canonical Preprocessing

The September pipeline order is:

1. Load raw HC3.
2. Validate required schema.
3. Normalize question text and answer text.
4. Validate `human_answers` and `chatgpt_answers` as lists.
5. Remove question rows with empty human or AI answer lists.
6. Investigate duplicate normalized-question groups.
7. Resolve duplicate questions deterministically, preserving distinct answers and de-duplicating identical answers within each label class.
8. Remove only clear invalid collection/system answer artifacts with explicit deterministic rules.
9. Link question groups that share exact normalized answer text into answer components.
10. Split answer components into train/validation/test at approximately 80/10/10 using `random_state=42`, stratified by `source`.
11. Expand answers after splitting.
12. Remove blank individual answers during expansion.
13. Validate labels, preserved group/component IDs, zero question overlap, zero exact answer-text overlap, and zero component overlap across splits.
14. Calculate EDA statistics and train-only TF-IDF features where needed.

No normalized question group, exact answer text, or answer-linked component
should appear in more than one split. Learned preprocessing such as TF-IDF must
be fit only on training text.

## Label Convention

At the expanded answer level:

- `label = 0`: human-written answer
- `label = 1`: ChatGPT/AI-generated answer

The committed `data/hc3_sample.csv` also includes a string `label` column
(`human`/`ai`) and a numeric `target` column (`0`/`1`). Treat `target` as the
sample file's numeric label.

## Tests

Run the synthetic test suite:

```bash
pytest
```

The tests do not download HC3. They cover normalization, empty answer lists,
blank answers, invalid system artifacts, duplicate question handling,
answer-linked component construction, deterministic splitting, split isolation,
split-before-expand behavior, labels, group/component ID preservation, and
source consistency.

## Notebook

Open and run:

```text
notebooks/Chewy_1A.ipynb
```

The notebook is the September EDA narrative. It should use the reusable
`spotbot` pipeline rather than keeping a second copy of preprocessing logic in
notebook cells.

## Milestone 2: Shared Setup

1. Pull the latest main branch, then create your own feature branch:
   `git switch main`, `git pull --ff-only origin main`, then
   `git switch -c your-name/model-name`. Start with a clean working tree.
2. Create a **new** model notebook in `notebooks/`, for example `02_logistic_regression.ipynb`,
   `03_linear_svm.ipynb`, `04_decision_tree.ipynb`, or `05_neural_network.ipynb`.
   Keep `Chewy_1A.ipynb` for Milestone 1 data preparation.
3. Load the shared splits with `load_prepared_splits()` as shown below.
   **Do not call `train_test_split` or create another split.**
4. Fit your features and train your model using `train_df`.
5. Use `val_df` to tune settings and compare versions of your model.
6. Once you have chosen the final setup, make predictions on `test_df`.
   Use test only for final evaluation, not to choose settings.
7. Evaluate validation and final test predictions with `evaluate_predictions()`.
   Record the final test metrics in the comparison table below, then commit
   your work to your branch and open a PR.

From a notebook running in `notebooks/`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))

from spotbot.prepared import load_prepared_splits
from spotbot.evaluation import evaluate_predictions

train_df, val_df, test_df = load_prepared_splits()
# After your model produces val_predictions in val_df row order:
y_true = val_df["label"]
predictions = val_predictions
source = val_df["source"]
scores = evaluate_predictions(y_true, predictions, source)
print(scores["macro_f1"], scores["accuracy"])
display(scores["per_class"])
print(scores["confusion_matrix"])
print(scores["per_domain"])
```

The loader reads local `data/hc3_all.csv` (or downloads HC3 if absent), runs the
September cleaning steps, and applies the saved IDs in `data/splits/`.
Counts are train **65,401**, validation **8,218**, and test **8,151** answers.
The loader checks data content, membership, and leakage before returning them.

For final test evaluation, use `test_df["label"]`, your test predictions, and
`test_df["source"]` in the same call. Predictions must be labels (`0 = human`,
`1 = AI`) in the same row order as the DataFrame.

You do **not** need to run `tests/test_evaluation.py` to evaluate your model.
That file tests our shared evaluation code. Just import and call
`evaluate_predictions()` in your model notebook.

Our dataset is about **68% human and 32% AI**, so predicting human every time
would already give about 68% accuracy. Compare models using macro-F1 and
precision/recall for both classes, not accuracy alone. You may use class
weighting or another training-only technique where appropriate, choosing
settings with validation. **Do not delete human examples from validation/test**
or balance those sets.

Every model should report **macro-F1 (primary)**, accuracy, Human precision and
recall, AI precision and recall, a confusion matrix, and per-domain results.
The helper returns these metrics; confusion-matrix rows are true labels and
columns are predictions, ordered Human then AI. Undefined scores use zero.
ROC curves and AUC are optional; calculate them from probabilities or decision
scores, not predicted class labels.

### Final Test Comparison

Fill this table with final test results after choosing each model's settings
using validation. Leave AUC blank if it was not calculated.

| Model | Macro-F1 | Accuracy | Human Precision | Human Recall | AI Precision | AI Recall | AUC |
|-------|----------|----------|-----------------|--------------|--------------|-----------|-----|
| Logistic Regression | | | | | | | |
| Linear SVM | | | | | | | |
| Decision Tree | 0.8918 | 0.9049 | 0.9362 | 0.9229 | 0.8415 | 0.8669 | |
| Neural Network | 0.9791 | 0.9817 | 0.9888 | 0.9843 | 0.9670 | 0.9763 | |
| Gradient Boosting | 0.9662 | 0.9707 | 0.98 | 0.98 | 0.96 | 0.95 | |

## Dataset Limitations

- HC3 answers come from early-2023 ChatGPT-era generations, so model drift is a risk.
- Human and AI answer lengths differ substantially; length is a known confound and should be measured, not silently removed.
- HC3 contains multiple domains with uneven representation, especially Reddit ELI5.
- Duplicate normalized questions exist and must be resolved at the group level before splitting.
- Some HC3 rows contain upstream collection artifacts or misaligned answer content; the September policy removes only clear system artifacts and documents suspicious data-quality cases rather than manually repairing them.
- Exact human answers can recur under near-duplicate questions, so the canonical split keeps answer-linked question components together.
- The held-out test split should not be used for model or hyperparameter selection.

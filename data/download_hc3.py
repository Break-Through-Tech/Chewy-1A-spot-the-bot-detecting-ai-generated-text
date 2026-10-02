"""
Download the HC3 (Human ChatGPT Comparison Corpus) dataset from Hugging Face.

Why this script exists
----------------------
The full HC3 dataset is ~147 MB, so we do NOT commit it to this public repo.
Run this script to pull the raw dataset from the Hugging Face Hub and save it to disk, exactly as published. Do your own preprocessing in your notebook.

A tiny 42-row preview (`data/hc3_sample.csv`) IS committed so you can eyeball the structure without downloading anything.

Raw structure (one row per question)
------------------------------------
    id, question, human_answers, chatgpt_answers, source
Heads-up: `human_answers` and `chatgpt_answers` are LISTS (a question can have
several answers).
"""
from pathlib import Path
import sys

OUT_PATH = Path(__file__).resolve().parent / "hc3_all.csv"
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from spotbot.data import load_hc3_from_huggingface


def main():
    # The shared loader handles the normal datasets path and its JSONL fallback.
    df = load_hc3_from_huggingface("all")

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df):,} rows (one per question) to {OUT_PATH}")
    print(df["source"].value_counts())


if __name__ == "__main__":
    main()

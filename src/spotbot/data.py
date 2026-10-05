"""HC3 loading helpers.

The canonical pipeline accepts the raw question-level HC3 shape:
question, human_answers, chatgpt_answers, and source.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_HC3_COLUMNS = ("question", "human_answers", "chatgpt_answers", "source")
LIST_COLUMNS = ("human_answers", "chatgpt_answers")


def validate_hc3_schema(df: pd.DataFrame, required: Iterable[str] = REQUIRED_HC3_COLUMNS) -> None:
    """Raise a clear error if the raw HC3 dataframe is missing required columns."""

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required HC3 columns: {missing}")


def parse_answer_list(value: object) -> list[str]:
    """Parse answer-list values from either native lists or CSV stringified lists."""

    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    raise ValueError(f"Expected an answer list, got {type(value).__name__}")


def load_hc3_csv(path: str | Path) -> pd.DataFrame:
    """Load a raw HC3 CSV produced by data/download_hc3.py."""

    df = pd.read_csv(path)
    validate_hc3_schema(df)
    df = df.copy()
    for column in LIST_COLUMNS:
        df[column] = df[column].apply(parse_answer_list)
    return df


def load_hc3_from_huggingface(config: str = "all") -> pd.DataFrame:
    """Download raw HC3 from Hugging Face.

    Prefer the documented datasets loader, and fall back to the public JSONL
    file if the local Python/datasets stack is incompatible.
    """

    try:
        from datasets import load_dataset

        ds = load_dataset("Hello-SimpleAI/HC3", config, trust_remote_code=True)
        df = ds["train"].to_pandas()
    except Exception:
        if config != "all":
            raise
        df = pd.read_json("https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl", lines=True)
    validate_hc3_schema(df)
    return df

"""Load the team's saved September split memberships."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .data import load_hc3_csv, load_hc3_from_huggingface
from .preprocessing import expand_answers, prepare_question_groups, remove_invalid_answers
from .splitting import (
    add_answer_components,
    verify_answer_text_isolation,
    verify_component_isolation,
    verify_split_isolation,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_FILES = {"train": "train_ids.csv", "validation": "val_ids.csv", "test": "test_ids.csv"}


def prepared_fingerprint(groups: pd.DataFrame) -> str:
    """Hash cleaned question content so upstream changes cannot silently pass."""
    columns = ["question_group_id", "question", "source", "human_answers", "chatgpt_answers"]
    records = groups.sort_values("question_group_id")[columns].to_dict("records")
    content = json.dumps(records, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def apply_saved_split_ids(groups: pd.DataFrame, split_dir: str | Path) -> pd.DataFrame:
    """Apply saved IDs, rejecting missing, extra, repeated, or leaking memberships."""
    split_dir = Path(split_dir)
    memberships = []
    for split, filename in SPLIT_FILES.items():
        ids = pd.read_csv(split_dir / filename, dtype=str)
        if "question_group_id" not in ids or ids["question_group_id"].isna().any():
            raise ValueError(f"Missing question_group_id values in {filename}")
        if ids["question_group_id"].duplicated().any():
            raise ValueError(f"Repeated question_group_id in {filename}")
        memberships.append(ids[["question_group_id"]].assign(split=split))
    membership = pd.concat(memberships, ignore_index=True)
    if membership["question_group_id"].duplicated().any():
        raise ValueError("A question_group_id appears in multiple split files")
    if groups["question_group_id"].duplicated().any():
        raise ValueError("Prepared question groups contain repeated IDs")
    if set(membership["question_group_id"]) != set(groups["question_group_id"]):
        raise ValueError("Saved split IDs do not match the prepared question groups")

    assigned = add_answer_components(groups).merge(
        membership, on="question_group_id", validate="one_to_one"
    )
    verify_split_isolation(assigned)
    verify_component_isolation(assigned)
    return assigned.sort_values("question_group_id").reset_index(drop=True)


def load_prepared_splits(
    raw_path: str | Path | None = None,
    split_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, validation, test answers using the saved September IDs.

    Uses data/hc3_all.csv by default, or downloads HC3 if that file is absent.
    This function never creates a new random split.
    """
    split_dir = Path(split_dir) if split_dir is not None else REPO_ROOT / "data" / "splits"
    path = Path(raw_path) if raw_path is not None else REPO_ROOT / "data" / "hc3_all.csv"
    raw = load_hc3_csv(path) if path.exists() or raw_path is not None else load_hc3_from_huggingface()
    groups, _, _ = prepare_question_groups(raw)
    groups, _ = remove_invalid_answers(groups)
    manifest = json.loads((split_dir / "manifest.json").read_text())
    if prepared_fingerprint(groups) != manifest["prepared_sha256"]:
        raise ValueError("Prepared HC3 content differs from the frozen September dataset")
    assigned = apply_saved_split_ids(groups, split_dir)
    expanded, _ = expand_answers(assigned)
    verify_answer_text_isolation(expanded)
    result = []
    for split in SPLIT_FILES:
        frame = expanded.loc[expanded["split"] == split].reset_index(drop=True)
        expected = manifest["splits"][split]
        if len(assigned.loc[assigned["split"] == split]) != expected["question_groups"]:
            raise ValueError(f"Frozen {split} question-group count changed")
        counts = {str(label): int(count) for label, count in frame["label"].value_counts().items()}
        if len(frame) != expected["examples"] or counts != expected["label_counts"]:
            raise ValueError(f"Frozen {split} answer counts changed")
        result.append(frame)
    return tuple(result)

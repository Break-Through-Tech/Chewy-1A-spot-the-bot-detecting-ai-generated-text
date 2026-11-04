"""Generate or verify the separate 70/15/15 split experiment; no model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import load_hc3_csv, load_hc3_from_huggingface
from .prepared import REPO_ROOT, SPLIT_FILES, prepared_fingerprint
from .preprocessing import expand_answers, prepare_question_groups, remove_invalid_answers
from .splitting import (
    make_group_split,
    verify_answer_text_isolation,
    verify_component_isolation,
    verify_split_isolation,
)

PROPORTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
RANDOM_STATE = 42


def _counts(values: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in values.value_counts().sort_index().items()}


def build_experimental_split(
    raw: pd.DataFrame, reference_manifest: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Use the existing cleaning and component splitter, changing only proportions."""
    groups, quality, _ = prepare_question_groups(raw)
    groups, invalid = remove_invalid_answers(groups)
    fingerprint = prepared_fingerprint(groups)
    if fingerprint != reference_manifest["prepared_sha256"]:
        raise ValueError("Experiment must use the same prepared dataset as the frozen 80/10/10 split")

    assigned, _ = make_group_split(
        groups, random_state=RANDOM_STATE, train_size=PROPORTIONS["train"],
        validation_size=PROPORTIONS["validation"], test_size=PROPORTIONS["test"],
    )
    if assigned["question_group_id"].duplicated().any() or set(assigned["question_group_id"]) != set(groups["question_group_id"]):
        raise ValueError("Every prepared question group must be assigned exactly once")
    expanded, blank = expand_answers(assigned)
    overlaps = {
        "question_group": verify_split_isolation(assigned),
        "exact_normalized_answer": verify_answer_text_isolation(expanded),
        "component": verify_component_isolation(assigned),
    }
    question_sets = {split: set(frame["question_norm"]) for split, frame in assigned.groupby("split")}
    overlaps["normalized_question"] = {
        (left, right): len(question_sets[left] & question_sets[right])
        for left, right in overlaps["question_group"]
    }
    if any(count for pairs in overlaps.values() for count in pairs.values()):
        raise ValueError("Split isolation failed")

    splits = {}
    for split in SPLIT_FILES:
        question_rows = assigned.loc[assigned["split"] == split]
        answers = expanded.loc[expanded["split"] == split]
        splits[split] = {
            "question_groups": len(question_rows),
            "components": int(question_rows["component_id"].nunique()),
            "examples": len(answers),
            "label_counts": _counts(answers["label"]),
            "question_source_counts": _counts(question_rows["source"]),
            "answer_source_counts": _counts(answers["source"]),
        }
    manifest = {
        "split_version": "70_15_15",
        "status": "experimental",
        "reference_commit": reference_manifest["reference_commit"],
        "prepared_sha256": fingerprint,
        "split_proportions": PROPORTIONS,
        "random_state": RANDOM_STATE,
        "split_unit": "answer_linked_component",
        "stratification": "component source; same make_group_split implementation as 80/10/10",
        "raw_question_rows": quality.original_question_rows,
        "empty_list_question_rows_removed": quality.empty_answer_list_rows,
        "invalid_answer_list_rows_removed": quality.invalid_answer_list_rows,
        "system_answers_removed": invalid.removed_by_label_reason,
        "blank_answers_removed": {"human": blank.blank_human_answers_removed, "ai": blank.blank_ai_answers_removed},
        "question_groups": len(groups),
        "components": int(assigned["component_id"].nunique()),
        "expanded_examples": len(expanded),
        "all_groups_assigned_once": True,
        "splits": splits,
        "leakage": {
            check: {f"{left}/{right}": count for (left, right), count in pairs.items()}
            for check, pairs in overlaps.items()
        },
    }
    return assigned, expanded, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, help="Local raw HC3 CSV; otherwise use local default or Hugging Face")
    parser.add_argument("--check", action="store_true", help="Recompute and check files without writing")
    args = parser.parse_args()
    path = args.raw_path if args.raw_path is not None else REPO_ROOT / "data" / "hc3_all.csv"
    raw = load_hc3_csv(path) if path.exists() or args.raw_path is not None else load_hc3_from_huggingface()
    reference = json.loads((REPO_ROOT / "data" / "splits" / "manifest.json").read_text())
    assigned, _, manifest = build_experimental_split(raw, reference)
    files = {"manifest.json": (json.dumps(manifest, indent=2) + "\n").encode()}
    for split, filename in SPLIT_FILES.items():
        ids = assigned.loc[assigned["split"] == split, ["question_group_id", "source", "component_id"]]
        files[filename] = ids.sort_values("question_group_id").to_csv(index=False, lineterminator="\n").encode()

    # This command can only write the experiment directory, never data/splits/.
    destination = REPO_ROOT / "data" / "splits_70_15_15"
    if args.check:
        for filename, content in files.items():
            if (destination / filename).read_bytes() != content:
                raise ValueError(f"Experimental split is not reproducible: {filename}")
        print("Experimental memberships and manifest match recomputation")
    else:
        for filename, content in files.items():
            existing = destination / filename
            if existing.exists() and existing.read_bytes() != content:
                raise ValueError(f"Refusing to overwrite different experimental data: {filename}")
        destination.mkdir(parents=True, exist_ok=True)
        for filename, content in files.items():
            (destination / filename).write_bytes(content)
        print(f"Saved experimental split to {destination}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

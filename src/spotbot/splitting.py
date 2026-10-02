"""Question-group splitting and leakage checks."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict, deque
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitSummary:
    """Question-group counts by split."""

    train_groups: int
    validation_groups: int
    test_groups: int
    train_components: int
    validation_components: int
    test_components: int


def _component_id(question_group_ids: list[str]) -> str:
    joined = "\n".join(sorted(question_group_ids))
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]
    return f"ac_{digest}"


def add_answer_components(question_groups: pd.DataFrame) -> pd.DataFrame:
    """Link question groups that share exact normalized answer text."""

    required = {"question_group_id", "source", "human_answers", "chatgpt_answers"}
    missing = required - set(question_groups.columns)
    if missing:
        raise ValueError(f"Missing required question-group columns: {sorted(missing)}")

    groups = question_groups.copy()
    answer_to_groups: dict[str, set[str]] = defaultdict(set)
    for _, row in groups.iterrows():
        group_id = row["question_group_id"]
        for answer in list(row["human_answers"]) + list(row["chatgpt_answers"]):
            if not str(answer).strip():
                continue
            answer_to_groups[answer].add(group_id)

    adjacency: dict[str, set[str]] = defaultdict(set)
    for linked_groups in answer_to_groups.values():
        if len(linked_groups) <= 1:
            continue
        for group_id in linked_groups:
            adjacency[group_id].update(other for other in linked_groups if other != group_id)

    all_group_ids = set(groups["question_group_id"])
    visited: set[str] = set()
    component_by_group: dict[str, str] = {}
    for group_id in sorted(all_group_ids):
        if group_id in visited:
            continue
        queue: deque[str] = deque([group_id])
        visited.add(group_id)
        component: list[str] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(adjacency.get(current, set())):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        component_id = _component_id(component)
        for member in component:
            component_by_group[member] = component_id

    groups["component_id"] = groups["question_group_id"].map(component_by_group)
    return groups


def component_summary(question_groups: pd.DataFrame) -> pd.DataFrame:
    """Return one row per answer-linked split component."""

    groups = question_groups if "component_id" in question_groups.columns else add_answer_components(question_groups)
    rows: list[dict[str, object]] = []
    for component_id, component in groups.groupby("component_id", sort=True):
        sources = component["source"].tolist()
        source_counts = Counter(sources)
        component_source = sorted(source_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        rows.append(
            {
                "component_id": component_id,
                "source": component_source,
                "component_size": int(len(component)),
                "is_mixed_source": len(source_counts) > 1,
            }
        )
    return pd.DataFrame(rows).sort_values("component_id").reset_index(drop=True)


def make_group_split(
    question_groups: pd.DataFrame,
    random_state: int = 42,
    train_size: float = 0.80,
    validation_size: float = 0.10,
    test_size: float = 0.10,
) -> tuple[pd.DataFrame, SplitSummary]:
    """Split answer-linked components before answer expansion, stratified by source."""

    total = train_size + validation_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train_size + validation_size + test_size must equal 1.0")
    required = {"question_group_id", "source", "human_answers", "chatgpt_answers"}
    missing = required - set(question_groups.columns)
    if missing:
        raise ValueError(f"Missing required question-group columns: {sorted(missing)}")

    groups = add_answer_components(question_groups).sort_values("question_group_id").reset_index(drop=True)
    components = component_summary(groups)
    train, remainder = train_test_split(
        components,
        test_size=(1.0 - train_size),
        random_state=random_state,
        stratify=components["source"],
    )
    relative_test_size = test_size / (validation_size + test_size)
    validation, test = train_test_split(
        remainder,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=remainder["source"],
    )

    component_splits = pd.concat(
        [
            train.assign(split="train"),
            validation.assign(split="validation"),
            test.assign(split="test"),
        ],
        ignore_index=True,
    )[["component_id", "split"]]
    split_groups = groups.merge(component_splits, on="component_id", how="left", validate="many_to_one")

    split_order = {"train": 0, "validation": 1, "test": 2}
    split_groups["_split_order"] = split_groups["split"].map(split_order)
    split_groups = split_groups.sort_values(["_split_order", "question_group_id"]).drop(columns="_split_order")
    split_groups = split_groups.reset_index(drop=True)
    verify_split_isolation(split_groups)
    verify_component_isolation(split_groups)
    summary = SplitSummary(
        train_groups=int((split_groups["split"] == "train").sum()),
        validation_groups=int((split_groups["split"] == "validation").sum()),
        test_groups=int((split_groups["split"] == "test").sum()),
        train_components=len(train),
        validation_components=len(validation),
        test_components=len(test),
    )
    return split_groups, summary


def verify_split_isolation(split_question_groups: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Validate that no question group appears in more than one split."""

    required = {"question_group_id", "split"}
    missing = required - set(split_question_groups.columns)
    if missing:
        raise ValueError(f"Missing required split columns: {sorted(missing)}")

    splits_by_group = split_question_groups.groupby("question_group_id")["split"].nunique()
    leaking_groups = splits_by_group[splits_by_group > 1]
    if not leaking_groups.empty:
        raise ValueError(f"{len(leaking_groups)} question groups appear in more than one split")

    split_ids = {
        split: set(values["question_group_id"])
        for split, values in split_question_groups.groupby("split", sort=True)
    }
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    return {
        pair: len(split_ids.get(pair[0], set()) & split_ids.get(pair[1], set()))
        for pair in pairs
    }


def verify_component_isolation(split_question_groups: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Validate that no answer-linked component appears in more than one split."""

    required = {"component_id", "split"}
    missing = required - set(split_question_groups.columns)
    if missing:
        raise ValueError(f"Missing required component columns: {sorted(missing)}")

    splits_by_component = split_question_groups.groupby("component_id")["split"].nunique()
    leaking_components = splits_by_component[splits_by_component > 1]
    if not leaking_components.empty:
        raise ValueError(f"{len(leaking_components)} components appear in more than one split")

    split_ids = {
        split: set(values["component_id"])
        for split, values in split_question_groups.groupby("split", sort=True)
    }
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    return {
        pair: len(split_ids.get(pair[0], set()) & split_ids.get(pair[1], set()))
        for pair in pairs
    }


def verify_answer_text_isolation(expanded_answers: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Validate that exact answer text does not appear across split pairs."""

    required = {"text", "split"}
    missing = required - set(expanded_answers.columns)
    if missing:
        raise ValueError(f"Missing required answer columns: {sorted(missing)}")

    split_texts = {
        split: set(values["text"])
        for split, values in expanded_answers.groupby("split", sort=True)
    }
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    overlaps = {
        pair: len(split_texts.get(pair[0], set()) & split_texts.get(pair[1], set()))
        for pair in pairs
    }
    leaking_pairs = {pair: count for pair, count in overlaps.items() if count}
    if leaking_pairs:
        raise ValueError(f"Exact answer text appears across splits: {leaking_pairs}")
    return overlaps

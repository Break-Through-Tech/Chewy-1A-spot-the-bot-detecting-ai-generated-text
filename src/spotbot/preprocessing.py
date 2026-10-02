"""Canonical September data-preparation logic for HC3."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from .data import LIST_COLUMNS, validate_hc3_schema


@dataclass(frozen=True)
class DataQualityReport:
    """Question-row and answer-row removals made by the preparation pipeline."""

    original_question_rows: int
    invalid_answer_list_rows: int
    empty_answer_list_rows: int
    question_rows_after_answer_list_validation: int
    blank_human_answers_removed: int
    blank_ai_answers_removed: int


@dataclass(frozen=True)
class DuplicateInvestigation:
    """Summary of duplicate normalized-question groups before resolution."""

    duplicated_question_groups: int
    duplicated_rows_involved: int
    duplicates_always_same_source: bool
    human_answer_list_relationships: dict[str, int]
    chatgpt_answer_list_relationships: dict[str, int]


@dataclass(frozen=True)
class InvalidAnswerReport:
    """Counts of clearly invalid collection/system artifacts removed."""

    removed_by_label_reason: dict[str, dict[str, int]]

    @property
    def total_removed(self) -> int:
        return sum(sum(reason_counts.values()) for reason_counts in self.removed_by_label_reason.values())


def normalize_text(text: object) -> str:
    """Normalize unicode and whitespace while preserving text content and case."""

    if not isinstance(text, str):
        raise TypeError(f"Expected text to be str, got {type(text).__name__}")
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def question_group_id(question_norm: str) -> str:
    """Create a stable identifier from the canonical normalized question."""

    digest = hashlib.sha1(question_norm.encode("utf-8")).hexdigest()[:16]
    return f"qg_{digest}"


def _is_answer_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _normalize_answer_list(values: list[str]) -> list[str]:
    return [normalize_text(value) for value in values]


def _relationship(lists: list[list[str]]) -> str:
    sets = [set(items) for items in lists]
    first = sets[0]
    if all(items == first for items in sets[1:]):
        return "identical"
    if any(first & items for items in sets[1:]) or any(
        sets[i] & sets[j] for i in range(len(sets)) for j in range(i + 1, len(sets))
    ):
        return "overlapping"
    return "different"


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


INVALID_ANSWER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "api_rate_limit",
        re.compile(r"(too many requests|only one message at a time).{0,160}(error generating a response)?", re.I),
    ),
    (
        "auth_failure",
        re.compile(r"(authentication token has expired|please try signing in again).{0,160}error generating", re.I),
    ),
    (
        "network_error",
        re.compile(r"(^|[!,. ]+)network error([!,. ]+|$|.{0,120}error generating a response)", re.I),
    ),
    (
        "chatgpt_security_page",
        re.compile(r"chat\.openai\.comchecking if the site connection is secure", re.I),
    ),
    (
        "chatgpt_ui_transcript",
        re.compile(r"chatgpt dec 15 version\. free research preview", re.I),
    ),
    (
        "generation_error",
        re.compile(r"^!?\s*there was an error generating a response\s*$", re.I),
    ),
)


def invalid_answer_reason(text: object) -> str | None:
    """Return a reason for clearly invalid system/collection artifacts only."""

    normalized = normalize_text(text)
    compact = normalized.replace("\u200b", "").replace("\\n", " ").strip()
    for reason, pattern in INVALID_ANSWER_PATTERNS:
        if pattern.search(compact):
            return reason
    return None


def remove_invalid_answers(question_groups: pd.DataFrame) -> tuple[pd.DataFrame, InvalidAnswerReport]:
    """Remove deterministic HC3 collection/system artifacts from grouped answers."""

    rows: list[dict[str, object]] = []
    removed: dict[str, Counter[str]] = {"human": Counter(), "ai": Counter()}

    for _, row in question_groups.iterrows():
        cleaned = row.to_dict()
        for column, label_name in [("human_answers", "human"), ("chatgpt_answers", "ai")]:
            kept: list[str] = []
            for answer in row[column]:
                reason = invalid_answer_reason(answer)
                if reason is None:
                    kept.append(answer)
                else:
                    removed[label_name][reason] += 1
            cleaned[column] = kept
        rows.append(cleaned)

    report = InvalidAnswerReport(
        removed_by_label_reason={
            label: dict(reason_counts)
            for label, reason_counts in removed.items()
            if reason_counts
        }
    )
    return pd.DataFrame(rows), report


def normalize_raw_hc3(df: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport]:
    """Validate required columns, normalize text fields, and remove unusable question rows."""

    validate_hc3_schema(df)
    working = df.copy()
    original_rows = len(working)

    valid_list_mask = working[list(LIST_COLUMNS)].map(_is_answer_list).all(axis=1)
    invalid_answer_list_rows = int((~valid_list_mask).sum())
    working = working.loc[valid_list_mask].copy()

    empty_mask = (working["human_answers"].str.len() == 0) | (working["chatgpt_answers"].str.len() == 0)
    empty_answer_list_rows = int(empty_mask.sum())
    working = working.loc[~empty_mask].copy()

    working["original_row_id"] = working.index.astype(str)
    working["question_norm"] = working["question"].apply(normalize_text)
    working["question"] = working["question_norm"]
    working["source"] = working["source"].apply(normalize_text)
    working["human_answers"] = working["human_answers"].apply(_normalize_answer_list)
    working["chatgpt_answers"] = working["chatgpt_answers"].apply(_normalize_answer_list)

    report = DataQualityReport(
        original_question_rows=original_rows,
        invalid_answer_list_rows=invalid_answer_list_rows,
        empty_answer_list_rows=empty_answer_list_rows,
        question_rows_after_answer_list_validation=len(working),
        blank_human_answers_removed=0,
        blank_ai_answers_removed=0,
    )
    return working.reset_index(drop=True), report


def investigate_duplicate_questions(normalized_df: pd.DataFrame) -> DuplicateInvestigation:
    """Inspect repeated normalized questions without changing the data."""

    duplicate_groups = normalized_df.groupby("question_norm", sort=True).filter(lambda group: len(group) > 1)
    if duplicate_groups.empty:
        return DuplicateInvestigation(
            duplicated_question_groups=0,
            duplicated_rows_involved=0,
            duplicates_always_same_source=True,
            human_answer_list_relationships={},
            chatgpt_answer_list_relationships={},
        )

    source_consistency: list[bool] = []
    human_relationships: Counter[str] = Counter()
    ai_relationships: Counter[str] = Counter()

    for _, group in duplicate_groups.groupby("question_norm", sort=True):
        source_consistency.append(group["source"].nunique(dropna=False) == 1)
        human_relationships[_relationship(group["human_answers"].tolist())] += 1
        ai_relationships[_relationship(group["chatgpt_answers"].tolist())] += 1

    return DuplicateInvestigation(
        duplicated_question_groups=int(duplicate_groups["question_norm"].nunique()),
        duplicated_rows_involved=int(len(duplicate_groups)),
        duplicates_always_same_source=all(source_consistency),
        human_answer_list_relationships=dict(human_relationships),
        chatgpt_answer_list_relationships=dict(ai_relationships),
    )


def resolve_duplicate_questions(normalized_df: pd.DataFrame) -> pd.DataFrame:
    """Group duplicate normalized questions while preserving distinct answers."""

    grouped_rows: list[dict[str, object]] = []
    for question_norm, group in normalized_df.sort_values(["question_norm", "original_row_id"]).groupby(
        "question_norm", sort=True
    ):
        sources = sorted(group["source"].drop_duplicates().tolist())
        if len(sources) != 1:
            raise ValueError(
                f"Duplicate normalized question has conflicting sources: {question_norm!r} -> {sources}"
            )

        human_answers: list[str] = []
        ai_answers: list[str] = []
        for _, row in group.iterrows():
            human_answers.extend(row["human_answers"])
            ai_answers.extend(row["chatgpt_answers"])

        grouped_rows.append(
            {
                "question_group_id": question_group_id(question_norm),
                "question": question_norm,
                "question_norm": question_norm,
                "source": sources[0],
                "human_answers": _dedupe_preserve_order(human_answers),
                "chatgpt_answers": _dedupe_preserve_order(ai_answers),
                "original_row_count": int(len(group)),
                "original_row_ids": tuple(group["original_row_id"].tolist()),
            }
        )

    return pd.DataFrame(grouped_rows).sort_values("question_group_id").reset_index(drop=True)


def expand_answers(question_groups: pd.DataFrame, split: str | None = None) -> tuple[pd.DataFrame, DataQualityReport]:
    """Expand grouped HC3 questions to one answer per row with labels 0=human and 1=AI."""

    records: list[dict[str, object]] = []
    blank_human = 0
    blank_ai = 0

    for _, row in question_groups.iterrows():
        base = {
            "question_group_id": row["question_group_id"],
            "question": row["question"],
            "source": row["source"],
        }
        if "component_id" in row:
            base["component_id"] = row["component_id"]
        if "split" in row:
            base["split"] = row["split"]
        elif split is not None:
            base["split"] = split

        for answer_index, text in enumerate(row["human_answers"]):
            text = normalize_text(text)
            if not text:
                blank_human += 1
                continue
            records.append({**base, "text": text, "label": 0, "answer_index": answer_index})

        for answer_index, text in enumerate(row["chatgpt_answers"]):
            text = normalize_text(text)
            if not text:
                blank_ai += 1
                continue
            records.append({**base, "text": text, "label": 1, "answer_index": answer_index})

    expanded = pd.DataFrame.from_records(records)
    if expanded.empty:
        expanded = pd.DataFrame(
            columns=["question_group_id", "component_id", "question", "source", "text", "label", "answer_index"]
        )
    ordered_columns = [
        column
        for column in [
            "split",
            "question_group_id",
            "component_id",
            "question",
            "source",
            "text",
            "label",
            "answer_index",
        ]
        if column in expanded.columns
    ]
    expanded = expanded[ordered_columns]

    report = DataQualityReport(
        original_question_rows=len(question_groups),
        invalid_answer_list_rows=0,
        empty_answer_list_rows=0,
        question_rows_after_answer_list_validation=len(question_groups),
        blank_human_answers_removed=blank_human,
        blank_ai_answers_removed=blank_ai,
    )
    return expanded, report


def prepare_question_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport, DuplicateInvestigation]:
    """Run canonical question-level preparation through duplicate resolution."""

    normalized, quality_report = normalize_raw_hc3(df)
    duplicate_report = investigate_duplicate_questions(normalized)
    question_groups = resolve_duplicate_questions(normalized)
    return question_groups, quality_report, duplicate_report

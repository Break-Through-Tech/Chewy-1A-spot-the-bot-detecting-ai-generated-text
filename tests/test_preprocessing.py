import numpy as np
import pandas as pd
import pytest

from spotbot.preprocessing import (
    expand_answers,
    invalid_answer_reason,
    normalize_raw_hc3,
    normalize_text,
    prepare_question_groups,
    remove_invalid_answers,
)


def test_normalize_text_preserves_case_and_collapses_whitespace():
    assert normalize_text("  What\u00a0is\tAI?\n") == "What is AI?"


@pytest.mark.parametrize("human_container", [list, np.array])
@pytest.mark.parametrize("ai_container", [list, np.array])
def test_answer_containers_normalize_to_python_lists(human_container, ai_container):
    raw = pd.DataFrame([{
        "question": " Q1 ", "source": "open_qa",
        "human_answers": human_container([" human ", "second human"]),
        "chatgpt_answers": ai_container([" ai "]),
    }])
    normalized, report = normalize_raw_hc3(raw)

    assert report.invalid_answer_list_rows == 0
    assert report.empty_answer_list_rows == 0
    assert normalized.iloc[0]["human_answers"] == ["human", "second human"]
    assert normalized.iloc[0]["chatgpt_answers"] == ["ai"]
    assert type(normalized.iloc[0]["human_answers"]) is list
    assert type(normalized.iloc[0]["chatgpt_answers"]) is list


@pytest.mark.parametrize("container", [list, np.array])
@pytest.mark.parametrize("column", ["human_answers", "chatgpt_answers"])
def test_empty_supported_containers_are_counted_as_empty_lists(container, column):
    raw = pd.DataFrame([
        {"question": "valid", "source": "open_qa",
         "human_answers": ["human"], "chatgpt_answers": ["ai"]},
        {"question": "empty", "source": "open_qa",
         "human_answers": container(["human"]), "chatgpt_answers": container(["ai"])},
    ])
    raw.at[1, column] = container([])
    normalized, report = normalize_raw_hc3(raw)
    assert report.invalid_answer_list_rows == 0
    assert report.empty_answer_list_rows == 1
    assert normalized["question"].tolist() == ["valid"]


@pytest.mark.parametrize("invalid", [
    "answer", ("answer",), {"answer"}, {"answer": "text"}, None,
    ["answer", 42], ["answer", None], [["nested"]],
    np.array(["answer", 42], dtype=object), np.array(["answer", None], dtype=object),
    np.array([1, 2]), np.array("scalar"), np.array([["nested"]]), np.empty((0, 2)),
    iter(["answer"]),
])
@pytest.mark.parametrize("column", ["human_answers", "chatgpt_answers"])
def test_invalid_containers_and_non_string_items_remain_invalid(invalid, column):
    raw = pd.DataFrame([
        {"question": "valid", "source": "open_qa",
         "human_answers": ["human"], "chatgpt_answers": ["ai"]},
        {"question": "invalid", "source": "open_qa",
         "human_answers": ["human"], "chatgpt_answers": ["ai"]},
    ])
    raw.at[1, column] = invalid
    normalized, report = normalize_raw_hc3(raw)
    assert report.invalid_answer_list_rows == 1
    assert report.empty_answer_list_rows == 0
    assert normalized["question"].tolist() == ["valid"]


def test_empty_answer_lists_are_removed_at_question_level():
    raw = pd.DataFrame(
        [
            {
                "question": "Q1",
                "human_answers": ["human"],
                "chatgpt_answers": ["ai"],
                "source": "open_qa",
            },
            {
                "question": "Q2",
                "human_answers": [],
                "chatgpt_answers": ["ai"],
                "source": "open_qa",
            },
            {
                "question": "Q3",
                "human_answers": ["human"],
                "chatgpt_answers": [],
                "source": "open_qa",
            },
        ]
    )

    normalized, report = normalize_raw_hc3(raw)

    assert len(normalized) == 1
    assert report.empty_answer_list_rows == 2
    assert normalized.iloc[0]["question"] == "Q1"


def test_blank_individual_answers_are_removed_after_expansion():
    raw = pd.DataFrame(
        [
            {
                "question": "Q1",
                "human_answers": ["human", "   "],
                "chatgpt_answers": ["", "ai"],
                "source": "open_qa",
            }
        ]
    )

    groups, _, _ = prepare_question_groups(raw)
    expanded, report = expand_answers(groups)

    assert expanded["text"].tolist() == ["human", "ai"]
    assert expanded["label"].tolist() == [0, 1]
    assert report.blank_human_answers_removed == 1
    assert report.blank_ai_answers_removed == 1


def test_invalid_system_answers_are_removed_with_reason_counts():
    raw = pd.DataFrame(
        [
            {
                "question": "Q1",
                "human_answers": ["human"],
                "chatgpt_answers": [
                    "! Too many requests in 1 hour. Try again later. There was an error generating a response",
                    "real ai answer",
                ],
                "source": "open_qa",
            }
        ]
    )

    groups, _, _ = prepare_question_groups(raw)
    cleaned, report = remove_invalid_answers(groups)

    assert cleaned.iloc[0]["chatgpt_answers"] == ["real ai answer"]
    assert report.removed_by_label_reason == {"ai": {"api_rate_limit": 1}}


def test_literal_backslash_network_error_is_invalid():
    assert invalid_answer_reason("!\\nnetwork error") == "network_error"


def test_generic_clarification_answer_is_not_removed():
    text = "I'm sorry, but it's not clear what you are asking about. Could you please provide more context?"

    assert invalid_answer_reason(text) is None


def test_duplicate_questions_preserve_distinct_answers_without_duplicate_answers():
    raw = pd.DataFrame(
        [
            {
                "question": "What is AI?",
                "human_answers": ["human one", "shared"],
                "chatgpt_answers": ["ai one"],
                "source": "open_qa",
            },
            {
                "question": " What   is AI? ",
                "human_answers": ["shared", "human two"],
                "chatgpt_answers": ["ai one", "ai two"],
                "source": "open_qa",
            },
        ]
    )

    groups, _, duplicate_report = prepare_question_groups(raw)

    assert duplicate_report.duplicated_question_groups == 1
    assert duplicate_report.duplicated_rows_involved == 2
    assert duplicate_report.duplicates_always_same_source is True
    assert groups.shape[0] == 1
    assert groups.iloc[0]["human_answers"] == ["human one", "shared", "human two"]
    assert groups.iloc[0]["chatgpt_answers"] == ["ai one", "ai two"]


def test_duplicate_questions_with_conflicting_sources_are_rejected():
    raw = pd.DataFrame(
        [
            {
                "question": "Q1",
                "human_answers": ["human"],
                "chatgpt_answers": ["ai"],
                "source": "open_qa",
            },
            {
                "question": "Q1",
                "human_answers": ["human 2"],
                "chatgpt_answers": ["ai 2"],
                "source": "finance",
            },
        ]
    )

    with pytest.raises(ValueError, match="conflicting sources"):
        prepare_question_groups(raw)


def test_expansion_preserves_required_columns_and_label_convention():
    raw = pd.DataFrame(
        [
            {
                "question": "Q1",
                "human_answers": ["human"],
                "chatgpt_answers": ["ai"],
                "source": "open_qa",
            }
        ]
    )

    groups, _, _ = prepare_question_groups(raw)
    expanded, _ = expand_answers(groups)

    assert {"question_group_id", "question", "source", "text", "label"} <= set(expanded.columns)
    assert expanded.loc[expanded["text"] == "human", "label"].item() == 0
    assert expanded.loc[expanded["text"] == "ai", "label"].item() == 1
    assert expanded["question_group_id"].nunique() == 1

import pandas as pd

from spotbot.preprocessing import expand_answers, prepare_question_groups
from spotbot.splitting import (
    add_answer_components,
    make_group_split,
    verify_answer_text_isolation,
    verify_component_isolation,
    verify_split_isolation,
)


def _synthetic_raw_questions(n_per_source=10):
    rows = []
    for source in ["open_qa", "finance", "medicine"]:
        for i in range(n_per_source):
            rows.append(
                {
                    "question": f"{source} question {i}",
                    "human_answers": [f"human {source} {i}", f"human extra {source} {i}"],
                    "chatgpt_answers": [f"ai {source} {i}"],
                    "source": source,
                }
            )
    return pd.DataFrame(rows)


def test_split_is_deterministic_and_stratified_by_source():
    raw = _synthetic_raw_questions()
    groups, _, _ = prepare_question_groups(raw)

    first, first_summary = make_group_split(groups, random_state=42)
    second, second_summary = make_group_split(groups, random_state=42)

    assert first_summary == second_summary
    assert first[["question_group_id", "split"]].equals(second[["question_group_id", "split"]])
    assert first_summary.train_groups == 24
    assert first_summary.validation_groups == 3
    assert first_summary.test_groups == 3
    assert first_summary.train_components == 24
    assert first_summary.validation_components == 3
    assert first_summary.test_components == 3
    for split in ["train", "validation", "test"]:
        assert set(first.loc[first["split"] == split, "source"]) == {"open_qa", "finance", "medicine"}


def test_zero_question_group_overlap_between_splits():
    raw = _synthetic_raw_questions()
    groups, _, _ = prepare_question_groups(raw)
    split_groups, _ = make_group_split(groups, random_state=42)

    overlaps = verify_split_isolation(split_groups)

    assert overlaps == {
        ("train", "validation"): 0,
        ("train", "test"): 0,
        ("validation", "test"): 0,
    }


def test_exact_answer_components_are_constructed_from_shared_answer_text():
    raw = _synthetic_raw_questions()
    raw.loc[0, "human_answers"] = ["shared exact answer"]
    raw.loc[1, "human_answers"] = ["shared exact answer"]
    groups, _, _ = prepare_question_groups(raw)

    with_components = add_answer_components(groups)
    shared_component_ids = with_components.loc[
        with_components["question"].isin(["open_qa question 0", "open_qa question 1"]),
        "component_id",
    ]

    assert shared_component_ids.nunique() == 1


def test_blank_answers_do_not_create_components():
    raw = _synthetic_raw_questions()
    raw.loc[0, "chatgpt_answers"] = [""]
    raw.loc[1, "chatgpt_answers"] = [""]
    groups, _, _ = prepare_question_groups(raw)

    with_components = add_answer_components(groups)
    blank_component_ids = with_components.loc[
        with_components["question"].isin(["open_qa question 0", "open_qa question 1"]),
        "component_id",
    ]

    assert blank_component_ids.nunique() == 2


def test_component_split_prevents_exact_answer_text_leakage():
    raw = _synthetic_raw_questions()
    raw.loc[0, "human_answers"] = ["shared exact answer"]
    raw.loc[1, "human_answers"] = ["shared exact answer"]
    groups, _, _ = prepare_question_groups(raw)
    split_groups, _ = make_group_split(groups, random_state=42)
    expanded, _ = expand_answers(split_groups)

    component_overlaps = verify_component_isolation(split_groups)
    answer_overlaps = verify_answer_text_isolation(expanded)

    assert component_overlaps == {
        ("train", "validation"): 0,
        ("train", "test"): 0,
        ("validation", "test"): 0,
    }
    assert answer_overlaps == {
        ("train", "validation"): 0,
        ("train", "test"): 0,
        ("validation", "test"): 0,
    }


def test_split_before_expand_invariant_and_group_id_preservation():
    raw = _synthetic_raw_questions()
    groups, _, _ = prepare_question_groups(raw)
    split_groups, _ = make_group_split(groups, random_state=42)
    expanded, _ = expand_answers(split_groups)

    assert expanded["question_group_id"].notna().all()
    assert expanded["component_id"].notna().all()
    assert expanded.groupby("question_group_id")["split"].nunique().max() == 1
    assert expanded.groupby("component_id")["split"].nunique().max() == 1
    assert expanded.groupby("question_group_id").size().min() == 3
    assert set(expanded["label"]) == {0, 1}


def test_source_domain_consistency_after_expansion():
    raw = _synthetic_raw_questions()
    groups, _, _ = prepare_question_groups(raw)
    split_groups, _ = make_group_split(groups, random_state=42)
    expanded, _ = expand_answers(split_groups)

    source_counts = expanded.groupby("question_group_id")["source"].nunique()

    assert source_counts.max() == 1

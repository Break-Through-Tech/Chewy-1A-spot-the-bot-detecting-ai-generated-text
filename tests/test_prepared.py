import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spotbot.prepared import (
    REPO_ROOT, SPLIT_FILES, apply_saved_split_ids, load_prepared_splits, prepared_fingerprint,
)
from spotbot.preprocessing import prepare_question_groups, remove_invalid_answers, expand_answers
from spotbot.splitting import (
    make_group_split, verify_answer_text_isolation, verify_component_isolation, verify_split_isolation,
)


@pytest.fixture
def saved_fixture(tmp_path):
    raw = pd.DataFrame([
        {"question": f"question {i}", "source": "domain",
         "human_answers": [f"human {i}"], "chatgpt_answers": [f"ai {i}"]}
        for i in range(30)
    ])
    raw.loc[1, "human_answers"] = ["human 0"]
    groups, _, _ = prepare_question_groups(raw)
    groups, _ = remove_invalid_answers(groups)
    assigned, _ = make_group_split(groups)
    expanded, _ = expand_answers(assigned)
    manifest = {"prepared_sha256": prepared_fingerprint(groups), "splits": {}}
    for split, filename in SPLIT_FILES.items():
        ids = assigned.loc[assigned.split == split, ["question_group_id"]]
        ids.to_csv(tmp_path / filename, index=False)
        frame = expanded.loc[expanded.split == split]
        manifest["splits"][split] = {
            "question_groups": len(ids), "examples": len(frame),
            "label_counts": {str(k): int(v) for k, v in frame.label.value_counts().items()},
        }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    path = tmp_path / "raw.csv"
    raw.to_csv(path, index=False)
    return groups, assigned, path, tmp_path


def test_loader_reconstructs_saved_splits_without_random_split(saved_fixture, monkeypatch):
    groups, assigned, path, folder = saved_fixture
    def forbidden(*args, **kwargs):
        raise AssertionError("The saved loader must not create a random split")
    monkeypatch.setattr("spotbot.splitting.make_group_split", forbidden)
    frames = load_prepared_splits(path, folder)
    reconstructed = pd.concat(frames, ignore_index=True)
    for split, frame in zip(SPLIT_FILES, frames):
        expected, _ = expand_answers(assigned.loc[assigned.split == split])
        pd.testing.assert_frame_equal(frame, expected.reset_index(drop=True))
    assert all(count == 0 for count in verify_split_isolation(reconstructed).values())
    assert all(count == 0 for count in verify_component_isolation(reconstructed).values())
    assert all(count == 0 for count in verify_answer_text_isolation(reconstructed).values())


def test_loader_preserves_frozen_results_for_huggingface_arrays(saved_fixture, monkeypatch):
    from spotbot.data import load_hc3_csv

    _, _, path, folder = saved_fixture
    expected = load_prepared_splits(path, folder)
    raw = load_hc3_csv(path)
    for column in ["human_answers", "chatgpt_answers"]:
        raw[column] = raw[column].apply(lambda answers: np.array(answers, dtype=object))
    monkeypatch.setattr("spotbot.prepared.load_hc3_from_huggingface", lambda: raw)
    monkeypatch.setattr("spotbot.prepared.REPO_ROOT", folder / "without_local_hc3")

    actual = load_prepared_splits(split_dir=folder)

    for expected_frame, actual_frame in zip(expected, actual):
        pd.testing.assert_frame_equal(actual_frame, expected_frame)


@pytest.mark.parametrize("problem", ["repeat", "overlap", "missing", "component"])
def test_rejects_invalid_membership(saved_fixture, problem):
    groups, assigned, _, folder = saved_fixture
    train = pd.read_csv(folder / "train_ids.csv")
    val = pd.read_csv(folder / "val_ids.csv")
    if problem == "repeat":
        train = pd.concat([train, train.iloc[[0]]])
    elif problem == "overlap":
        train = pd.concat([train, val.iloc[[0]]])
    elif problem == "missing":
        train = train.iloc[1:]
    else:
        linked = assigned.groupby("component_id").filter(lambda frame: len(frame) > 1)
        member = linked.iloc[0]
        old_file = folder / SPLIT_FILES[member["split"]]
        old = pd.read_csv(old_file)
        old = old.loc[old.question_group_id != member.question_group_id]
        old.to_csv(old_file, index=False)
        target = "test" if member["split"] != "test" else "validation"
        target_file = folder / SPLIT_FILES[target]
        changed = pd.concat([pd.read_csv(target_file), pd.DataFrame({"question_group_id": [member.question_group_id]})])
        changed.to_csv(target_file, index=False)
        with pytest.raises(ValueError, match="components"):
            apply_saved_split_ids(groups, folder)
        return
    train.to_csv(folder / "train_ids.csv", index=False)
    with pytest.raises(ValueError):
        apply_saved_split_ids(groups, folder)


def test_changed_content_is_rejected(saved_fixture):
    _, _, path, folder = saved_fixture
    raw = pd.read_csv(path)
    raw.loc[0, "human_answers"] = "['changed answer']"
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match="content differs"):
        load_prepared_splits(path, folder)


def test_committed_memberships_have_validated_counts_and_no_overlap():
    expected = {"train": 18671, "validation": 2343, "test": 2325}
    seen = set()
    for split, filename in SPLIT_FILES.items():
        frame = pd.read_csv(REPO_ROOT / "data" / "splits" / filename)
        ids = set(frame.question_group_id)
        assert len(frame) == len(ids) == expected[split]
        assert not ids & seen
        seen.update(ids)


@pytest.mark.skipif(not (REPO_ROOT / "data/hc3_all.csv").exists(), reason="Local HC3 not downloaded")
def test_real_hc3_frozen_memberships_match_validated_pipeline():
    from spotbot.data import load_hc3_csv
    raw = load_hc3_csv(REPO_ROOT / "data/hc3_all.csv")
    groups, _, _ = prepare_question_groups(raw)
    groups, _ = remove_invalid_answers(groups)
    assigned, _ = make_group_split(groups)
    for split, frame in zip(SPLIT_FILES, load_prepared_splits()):
        expected, _ = expand_answers(assigned.loc[assigned.split == split])
        pd.testing.assert_frame_equal(frame, expected.reset_index(drop=True))

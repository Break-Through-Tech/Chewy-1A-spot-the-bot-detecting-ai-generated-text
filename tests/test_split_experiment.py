import hashlib
import json
from itertools import combinations
import pandas as pd
import pytest

from spotbot.data import load_hc3_csv
from spotbot.prepared import REPO_ROOT, SPLIT_FILES, load_prepared_splits, prepared_fingerprint
from spotbot.preprocessing import expand_answers, prepare_question_groups, remove_invalid_answers
from spotbot.split_experiment import build_experimental_split
from spotbot.splitting import make_group_split


def assert_isolated_and_complete(frames, expected_ids):
    assigned = pd.concat(frames, ignore_index=True)
    assert set(assigned["question_group_id"]) == expected_ids
    for column in ["question_group_id", "question", "text", "component_id"]:
        for left, right in combinations(frames, 2):
            assert not set(left[column]) & set(right[column])
    assert assigned.groupby("question_group_id")["split"].nunique().eq(1).all()
    assert assigned.groupby("component_id")["split"].nunique().eq(1).all()


def write_version(folder, assigned, manifest):
    folder.mkdir(parents=True)
    for split, filename in SPLIT_FILES.items():
        assigned.loc[assigned["split"] == split, ["question_group_id", "source", "component_id"]].to_csv(folder / filename, index=False)
    (folder / "manifest.json").write_text(json.dumps(manifest))


@pytest.fixture(scope="module")
def synthetic_versions(tmp_path_factory):
    root = tmp_path_factory.mktemp("split_versions")
    raw = pd.DataFrame([
        {"question": f"{source} question {i}", "source": source,
         "human_answers": [f"human {source} {i}", f"extra human {source} {i}"],
         "chatgpt_answers": [f"ai {source} {i}"]}
        for source in ["finance", "medicine", "open_qa"] for i in range(20)
    ])
    raw.at[1, "human_answers"] = ["human finance 0", "extra human finance 1"]
    raw.at[2, "human_answers"] = ["extra human finance 1", "extra human finance 2"]
    raw.at[4, "human_answers"] = ["", "extra human finance 4"]
    raw.at[5, "chatgpt_answers"] = ["! network error There was an error generating a response", "ai finance 5"]
    groups, _, _ = prepare_question_groups(raw)
    groups, _ = remove_invalid_answers(groups)
    reference = {"reference_commit": "test-reference", "prepared_sha256": prepared_fingerprint(groups), "splits": {}}
    default_groups, _ = make_group_split(groups)
    default_answers, _ = expand_answers(default_groups)
    for split in SPLIT_FILES:
        frame = default_answers.loc[default_answers["split"] == split]
        reference["splits"][split] = {
            "question_groups": int((default_groups["split"] == split).sum()), "examples": len(frame),
            "label_counts": {str(k): int(v) for k, v in frame["label"].value_counts().items()},
        }
    experimental_groups, experimental_answers, manifest = build_experimental_split(raw, reference)
    write_version(root / "data/splits", default_groups, reference)
    write_version(root / "data/splits_70_15_15", experimental_groups, manifest)
    raw_path = root / "raw.csv"
    raw.to_csv(raw_path, index=False)
    return root, raw_path, raw, groups, default_groups, experimental_groups, experimental_answers, reference, manifest


def test_both_versions_coexist_without_resplitting(synthetic_versions, monkeypatch):
    root, raw_path, _, groups, default_groups, experimental_groups, _, _, _ = synthetic_versions
    monkeypatch.setattr("spotbot.prepared.REPO_ROOT", root)
    def no_split(*args, **kwargs):
        raise AssertionError("Loading frozen data must not generate another split")
    monkeypatch.setattr("spotbot.splitting.make_group_split", no_split)
    default = load_prepared_splits(raw_path)
    explicit_default = load_prepared_splits(raw_path, split_version="80_10_10")
    custom_directory = load_prepared_splits(raw_path, root / "data/splits")
    experimental = load_prepared_splits(raw_path, split_version="70_15_15")
    for i, split in enumerate(SPLIT_FILES):
        expected_default, _ = expand_answers(default_groups.loc[default_groups["split"] == split])
        expected_experiment, _ = expand_answers(experimental_groups.loc[experimental_groups["split"] == split])
        pd.testing.assert_frame_equal(default[i], expected_default.reset_index(drop=True))
        pd.testing.assert_frame_equal(default[i], explicit_default[i])
        pd.testing.assert_frame_equal(default[i], custom_directory[i])
        pd.testing.assert_frame_equal(experimental[i], expected_experiment.reset_index(drop=True))
    assert_isolated_and_complete(default, set(groups["question_group_id"]))
    assert_isolated_and_complete(experimental, set(groups["question_group_id"]))


def test_experiment_changes_only_split_proportions(synthetic_versions, monkeypatch):
    _, _, raw, groups, _, expected_groups, _, reference, expected_manifest = synthetic_versions
    before = raw.to_json()
    calls = []
    def record_split(frame, **kwargs):
        pd.testing.assert_frame_equal(frame, groups)
        calls.append(kwargs)
        return make_group_split(frame, **kwargs)
    monkeypatch.setattr("spotbot.split_experiment.make_group_split", record_split)
    assigned, _, manifest = build_experimental_split(raw, reference)
    pd.testing.assert_frame_equal(assigned, expected_groups)
    assert manifest == expected_manifest
    assert calls == [{"random_state": 42, "train_size": 0.70, "validation_size": 0.15, "test_size": 0.15}]
    assert manifest["prepared_sha256"] == reference["prepared_sha256"]
    assert raw.to_json() == before
    assert len(assigned) == assigned["question_group_id"].nunique() == len(groups)


def test_experiment_rejects_different_cleaned_data(synthetic_versions):
    _, _, raw, _, _, _, _, reference, _ = synthetic_versions
    changed = raw.copy()
    changed.at[0, "human_answers"] = ["changed human answer"]
    with pytest.raises(ValueError, match="same prepared dataset"):
        build_experimental_split(changed, reference)


@pytest.mark.parametrize("kwargs", [
    {"split_version": "unknown"},
    {"split_dir": "custom", "split_version": "70_15_15"},
])
def test_loader_rejects_ambiguous_or_unknown_versions(kwargs, monkeypatch):
    def no_load(*args, **kwargs):
        raise AssertionError("Invalid options must fail before loading HC3")
    monkeypatch.setattr("spotbot.prepared.load_hc3_csv", no_load)
    monkeypatch.setattr("spotbot.prepared.load_hc3_from_huggingface", no_load)
    with pytest.raises(ValueError):
        load_prepared_splits(**kwargs)


def test_original_frozen_files_are_byte_unchanged():
    expected = {
        "train_ids.csv": "efe98e9e92e1476ed2ced3f07e387b9cbfa4e17b0fa12d40e475a6befb3bcf80",
        "val_ids.csv": "86294ee739bf99a8db5150ba9ca596766c24c3cf6089b26ccb2d02f09032295b",
        "test_ids.csv": "b1c1f074e1e5b9125e3e9086144f7b25b9409820c5c47872b8ac6a272b46ab06",
        "manifest.json": "230da60de6b484e7c96b73e675e89bf27f4e83b9d8c3ce60610bf3e28039b1ae",
    }
    for filename, digest in expected.items():
        assert hashlib.sha256((REPO_ROOT / "data/splits" / filename).read_bytes()).hexdigest() == digest


def test_committed_versions_have_complete_unique_ids():
    default_ids = set()
    for filename in SPLIT_FILES.values():
        default_ids.update(pd.read_csv(REPO_ROOT / "data/splits" / filename)["question_group_id"])
    folder = REPO_ROOT / "data/splits_70_15_15"
    manifest = json.loads((folder / "manifest.json").read_text())
    reference = json.loads((REPO_ROOT / "data/splits/manifest.json").read_text())
    assert manifest["prepared_sha256"] == reference["prepared_sha256"]
    assert manifest["split_proportions"] == {"train": 0.70, "validation": 0.15, "test": 0.15}
    assert manifest["random_state"] == 42
    seen = set()
    components = set()
    for split, filename in SPLIT_FILES.items():
        frame = pd.read_csv(folder / filename)
        ids = set(frame["question_group_id"])
        assert len(frame) == len(ids) == manifest["splits"][split]["question_groups"]
        assert not ids & seen
        assert not set(frame["component_id"]) & components
        assert frame["source"].value_counts().to_dict() == manifest["splits"][split]["question_source_counts"]
        seen.update(ids)
        components.update(frame["component_id"])
    assert seen == default_ids
    assert len(components) == manifest["components"]


@pytest.mark.skipif(not (REPO_ROOT / "data/hc3_all.csv").exists(), reason="Local HC3 not downloaded")
def test_real_hc3_experiment_reproduces_and_preserves_both_versions():
    raw = load_hc3_csv(REPO_ROOT / "data/hc3_all.csv")
    reference = json.loads((REPO_ROOT / "data/splits/manifest.json").read_text())
    assigned, expanded, manifest = build_experimental_split(raw, reference)
    folder = REPO_ROOT / "data/splits_70_15_15"
    assert manifest == json.loads((folder / "manifest.json").read_text())
    default = load_prepared_splits()
    experimental = load_prepared_splits(split_version="70_15_15")
    expected_ids = set(assigned["question_group_id"])
    for split, frame in zip(SPLIT_FILES, experimental):
        pd.testing.assert_frame_equal(frame, expanded.loc[expanded["split"] == split].reset_index(drop=True))
        saved = pd.read_csv(folder / SPLIT_FILES[split])
        actual = assigned.loc[assigned["split"] == split, list(saved.columns)].sort_values("question_group_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(saved, actual)
        expected = manifest["splits"][split]
        assert len(frame) == expected["examples"]
        assert {str(k): int(v) for k, v in frame["label"].value_counts().items()} == expected["label_counts"]
    assert [len(frame) for frame in default] == [65401, 8218, 8151]
    for frames in [default, experimental]:
        assert_isolated_and_complete(frames, expected_ids)
    columns = ["question_group_id", "component_id", "question", "source", "text", "label", "answer_index"]
    combined = [pd.concat(frames)[columns].sort_values(["question_group_id", "label", "answer_index"]).reset_index(drop=True)
                for frames in [default, experimental]]
    pd.testing.assert_frame_equal(*combined)

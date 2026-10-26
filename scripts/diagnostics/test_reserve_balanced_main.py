from collections import Counter
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reserve_balanced_main as reservation


def rows():
    return [
        {"id": f"mmlu_pro/{subject:02d}-{i:02d}", "subject": f"topic{subject:02d}",
         "text": f"Question {subject}-{i}?", "truth": "A", "n_options": 4}
        for subject in range(14) for i in range(25)
    ]


def test_exact_200_all_14_subjects_balanced_shuffled_and_deterministic():
    source = rows()
    before = copy.deepcopy(source)
    tasks, quotas = reservation.balanced_sample(source, [], 200, 20261026)
    assert len(tasks) == 200
    assert len({row["id"] for row in tasks}) == 200
    assert len({reservation.normalized(row["text"]) for row in tasks}) == 200
    counts = Counter(row["subject"] for row in tasks)
    assert counts == quotas
    assert sorted(counts.values()) == [14] * 10 + [15] * 4
    assert tasks != sorted(tasks, key=lambda row: row["subject"])
    assert (tasks, quotas) == reservation.balanced_sample(reversed(source), [], 200, 20261026)
    assert (tasks, quotas) != reservation.balanced_sample(source, [], 200, 99)
    assert source == before


def test_excludes_reserved_ids_and_normalized_text_and_deduplicates():
    source = rows()
    reserved = [source[0], dict(source[1], id="old-validation/id",
                               text="  " + source[1]["text"].upper() + "\n")]
    source += [dict(source[2], id="duplicate/text"),
               dict(source[3], text="same ID with different content")]
    tasks, _ = reservation.balanced_sample(source, reserved)
    assert not {row["id"] for row in tasks} & {row["id"] for row in reserved}
    assert not {reservation.normalized(row["text"]) for row in tasks} & {
        reservation.normalized(row["text"]) for row in reserved}
    assert len({row["id"] for row in tasks}) == len(tasks)
    assert len({reservation.normalized(row["text"]) for row in tasks}) == len(tasks)


def test_missing_subject_or_insufficient_topic_blocks_instead_of_reallocating():
    source = rows()
    with pytest.raises(ValueError, match="every subject"):
        reservation.balanced_sample(source, [], 13)
    with pytest.raises(ValueError, match="nonempty subject"):
        reservation.balanced_sample([dict(source[0], subject=None)], [])
    blocked = [row for row in source if row["subject"] == "topic00"]
    with pytest.raises(ValueError, match="no reallocation"):
        reservation.balanced_sample(source, blocked)


def test_cli_writes_bound_manifest_offline_and_never_overwrites(tmp_path, monkeypatch):
    import huggingface_hub
    import pandas as pd
    data = pd.DataFrame([
        {"question_id": i, "category": row["subject"], "question": row["text"],
         "options": ["a", "b", "c", "d"], "answer": "A"}
        for i, row in enumerate(rows())
    ])
    parquet = tmp_path / "test.parquet"
    parquet.write_bytes(b"cached parquet fixture")
    reserved = tmp_path / "reserved.json"
    reserved.write_text('{"tasks":[]}')
    out = tmp_path / "main/eval_tasks.json"
    def cached(*args, **kwargs):
        assert kwargs["local_files_only"] is True
        return str(parquet)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", cached)
    monkeypatch.setattr(pd, "read_parquet", lambda path: data)
    monkeypatch.setattr(sys, "argv", ["reserve_balanced_main", "--out", str(out),
                                     "--reserved", str(reserved)])
    reservation.main()
    doc = json.loads(out.read_text())
    assert len(doc["tasks"]) == 200
    assert doc["purpose"] == "fresh-main-confirmation"
    assert doc["source"]["split"] == "test"
    assert doc["excluded_sha256"] == {str(reserved): reservation.digest(reserved)}
    assert doc["sampling"]["subject_counts"] == doc["sampling"]["quotas"]
    before = out.read_bytes()
    with pytest.raises(FileExistsError, match="frozen"):
        reservation.main()
    assert out.read_bytes() == before

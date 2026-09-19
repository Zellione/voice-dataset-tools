from __future__ import annotations

from pathlib import Path

import pytest

from voice_dataset.reconciliation import split_turn
from voice_dataset.storage import DatasetStorage


def make_storage(tmp_path: Path) -> DatasetStorage:
    return DatasetStorage(tmp_path / "dataset")


def add_region(
    storage: DatasetStorage,
    *,
    region_id: str,
    start: float,
    end: float,
) -> None:
    storage.regions.append(
        {
            "schema_version": 2,
            "id": region_id,
            "source_id": "source_001",
            "source_start": start,
            "source_end": end,
            "representations": {},
            "transcripts": {},
            "embeddings": {},
            "reconciliation": {
                "status": "pending",
                "reason": None,
                "notes": None,
            },
            "metadata": {},
        }
    )


def add_turn(
    storage: DatasetStorage,
    *,
    representations: dict | None = None,
) -> None:
    storage.turns.append(
        {
            "schema_version": 2,
            "id": "turn_000001",
            "source_id": "source_001",
            "source_start": 1.0,
            "source_end": 4.0,
            "source_regions": [
                "region_000001",
                "region_000002",
                "region_000003",
            ],
            "language": "en",
            "transcript": "old transcript",
            "representations": representations or {},
            "embeddings": {
                "old": {
                    "model": "test",
                }
            },
            "assignment": {
                "status": "unknown",
                "voice_id": None,
                "character_id": None,
            },
            "review": {
                "status": "pending",
            },
            "metadata": {},
        }
    )


def make_split_dataset(
    tmp_path: Path,
    *,
    representations: dict | None = None,
) -> DatasetStorage:
    storage = make_storage(tmp_path)

    add_region(
        storage,
        region_id="region_000001",
        start=1.0,
        end=2.0,
    )
    add_region(
        storage,
        region_id="region_000002",
        start=2.2,
        end=3.0,
    )
    add_region(
        storage,
        region_id="region_000003",
        start=3.2,
        end=4.0,
    )

    add_turn(
        storage,
        representations=representations,
    )

    return storage


def test_split_turn_creates_two_fresh_turns(
    tmp_path: Path,
) -> None:
    storage = make_split_dataset(tmp_path)

    left, right = split_turn(
        storage,
        "turn_000001",
        after_region_id="region_000002",
    )

    assert left["id"] == "turn_000001"
    assert left["source_regions"] == [
        "region_000001",
        "region_000002",
    ]
    assert left["source_start"] == 1.0
    assert left["source_end"] == 3.0

    assert right["id"] == "turn_000002"
    assert right["source_regions"] == [
        "region_000003",
    ]
    assert right["source_start"] == 3.2
    assert right["source_end"] == 4.0

    for turn in (left, right):
        assert turn["language"] is None
        assert turn["transcript"] is None
        assert turn["representations"] == {}
        assert turn["embeddings"] == {}
        assert turn["assignment"]["status"] == "unknown"
        assert turn["review"]["status"] == "pending"


def test_split_after_final_region_does_not_mutate(
    tmp_path: Path,
) -> None:
    storage = make_split_dataset(tmp_path)

    before = storage.turns.path.read_bytes()

    with pytest.raises(
        ValueError,
        match="Cannot split after the final region",
    ):
        split_turn(
            storage,
            "turn_000001",
            after_region_id="region_000003",
        )

    after = storage.turns.path.read_bytes()

    assert after == before


def test_split_removes_old_representation_files(
    tmp_path: Path,
) -> None:
    storage = make_split_dataset(
        tmp_path,
        representations={
            "raw": {
                "path": "turns/turn_000001/raw.wav",
            },
            "speech": {
                "path": "turns/turn_000001/speech.wav",
            },
        },
    )

    turn_directory = (
        storage.root / "turns" / "turn_000001"
    )
    turn_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = turn_directory / "raw.wav"
    speech = turn_directory / "speech.wav"

    raw.write_bytes(b"old raw")
    speech.write_bytes(b"old speech")

    split_turn(
        storage,
        "turn_000001",
        after_region_id="region_000002",
    )

    assert not raw.exists()
    assert not speech.exists()


def test_split_refuses_representation_outside_turn_directory(
    tmp_path: Path,
) -> None:
    storage = make_split_dataset(
        tmp_path,
        representations={
            "raw": {
                "path": "turns/turn_000002/raw.wav",
            },
        },
    )

    outside = (
        storage.root
        / "turns"
        / "turn_000002"
        / "raw.wav"
    )
    outside.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    outside.write_bytes(b"must survive")

    before = storage.turns.path.read_bytes()

    with pytest.raises(
        ValueError,
        match="representation path escapes turn directory",
    ):
        split_turn(
            storage,
            "turn_000001",
            after_region_id="region_000002",
        )

    assert storage.turns.path.read_bytes() == before
    assert outside.read_bytes() == b"must survive"

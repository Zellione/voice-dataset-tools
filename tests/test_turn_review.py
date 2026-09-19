from pathlib import Path

import pytest

from voice_dataset.review import (
    mark_turn_boundary_clipped,
    mark_turn_boundary_complete,
    mark_turn_boundary_unknown,
    mark_turn_pending,
    mark_turn_reviewed,
    set_turn_boundary_review_status,
    set_turn_review_status,
)
from voice_dataset.schema import TurnRecord
from voice_dataset.storage import DatasetStorage


def make_turn(
    storage: DatasetStorage,
) -> dict:
    turn = TurnRecord(
        id="turn_000001",
        source_id="source_001",
        source_start=1.0,
        source_end=2.0,
        source_regions=["region_000001"],
    )

    storage.add_turn(turn)

    result = storage.get_turn(turn.id)
    assert result is not None

    return result


def make_boundary_turn(
    storage: DatasetStorage,
) -> dict:
    turn = make_turn(storage)

    def add_boundary_evidence(
        record: dict,
    ) -> dict:
        metadata = dict(
            record.get("metadata") or {}
        )

        metadata["boundary_evidence"] = {
            "representation": "center",
            "margin": 0.1,
            "near_source_start": True,
            "near_source_end": False,
        }

        record["metadata"] = metadata
        return record

    storage.update_turn(
        turn["id"],
        add_boundary_evidence,
    )

    result = storage.get_turn(turn["id"])
    assert result is not None

    return result


def test_mark_turn_reviewed(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)
    make_turn(storage)

    result = mark_turn_reviewed(
        storage,
        "turn_000001",
    )

    assert result["review"]["status"] == "reviewed"

    stored = storage.get_turn("turn_000001")
    assert stored is not None
    assert stored["review"]["status"] == "reviewed"


def test_mark_turn_pending(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)
    make_turn(storage)

    mark_turn_reviewed(
        storage,
        "turn_000001",
    )

    result = mark_turn_pending(
        storage,
        "turn_000001",
    )

    assert result["review"]["status"] == "pending"


def test_review_status_preserves_metadata(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)
    make_turn(storage)

    def add_metadata(record: dict) -> dict:
        record["review"] = {
            "status": "pending",
            "notes": "keep me",
        }
        return record

    storage.update_turn(
        "turn_000001",
        add_metadata,
    )

    result = mark_turn_reviewed(
        storage,
        "turn_000001",
    )

    assert result["review"] == {
        "status": "reviewed",
        "notes": "keep me",
    }


def test_invalid_review_status_does_not_mutate(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)
    make_turn(storage)

    before = storage.turns.path.read_bytes()

    with pytest.raises(
        ValueError,
        match="Invalid review status",
    ):
        set_turn_review_status(
            storage,
            "turn_000001",
            "banana",
        )

    assert storage.turns.path.read_bytes() == before


def test_unknown_turn_does_not_mutate(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)
    make_turn(storage)

    before = storage.turns.path.read_bytes()

    with pytest.raises(
        KeyError,
        match="Turn does not exist",
    ):
        mark_turn_reviewed(
            storage,
            "turn_999999",
        )

    assert storage.turns.path.read_bytes() == before


def test_same_review_status_is_idempotent(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)
    make_turn(storage)

    mark_turn_reviewed(
        storage,
        "turn_000001",
    )

    before = storage.turns.path.read_bytes()

    mark_turn_reviewed(
        storage,
        "turn_000001",
    )

    assert storage.turns.path.read_bytes() == before


def test_mark_turn_boundary_clipped(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)
    make_boundary_turn(storage)

    result = mark_turn_boundary_clipped(
        storage,
        "turn_000001",
    )

    assert result["review"] == {
        "status": "pending",
        "boundary": {
            "status": "clipped",
        },
    }


def test_mark_turn_boundary_complete(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)
    make_boundary_turn(storage)

    result = mark_turn_boundary_complete(
        storage,
        "turn_000001",
    )

    assert (
        result["review"]["boundary"]["status"]
        == "complete"
    )


def test_mark_turn_boundary_unknown(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)
    make_boundary_turn(storage)

    mark_turn_boundary_clipped(
        storage,
        "turn_000001",
    )

    result = mark_turn_boundary_unknown(
        storage,
        "turn_000001",
    )

    assert (
        result["review"]["boundary"]["status"]
        == "unknown"
    )


def test_boundary_review_preserves_other_review_fields(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)
    make_boundary_turn(storage)

    def add_review_fields(
        record: dict,
    ) -> dict:
        record["review"] = {
            "status": "reviewed",
            "notes": "keep me",
        }
        return record

    storage.update_turn(
        "turn_000001",
        add_review_fields,
    )

    result = mark_turn_boundary_clipped(
        storage,
        "turn_000001",
    )

    assert result["review"] == {
        "status": "reviewed",
        "notes": "keep me",
        "boundary": {
            "status": "clipped",
        },
    }


def test_invalid_boundary_review_status_does_not_mutate(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)
    make_boundary_turn(storage)

    before = storage.turns.path.read_bytes()

    with pytest.raises(
        ValueError,
        match="Invalid boundary review status",
    ):
        set_turn_boundary_review_status(
            storage,
            "turn_000001",
            "banana",
        )

    assert storage.turns.path.read_bytes() == before


def test_unknown_turn_boundary_review_does_not_mutate(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)
    make_boundary_turn(storage)

    before = storage.turns.path.read_bytes()

    with pytest.raises(
        KeyError,
        match="Turn does not exist",
    ):
        mark_turn_boundary_clipped(
            storage,
            "turn_999999",
        )

    assert storage.turns.path.read_bytes() == before


def test_same_boundary_review_status_is_idempotent(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)
    make_boundary_turn(storage)

    mark_turn_boundary_clipped(
        storage,
        "turn_000001",
    )

    before = storage.turns.path.read_bytes()

    mark_turn_boundary_clipped(
        storage,
        "turn_000001",
    )

    assert storage.turns.path.read_bytes() == before


def test_boundary_review_rejects_non_boundary_turn(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)
    make_turn(storage)

    before = storage.turns.path.read_bytes()

    with pytest.raises(
        ValueError,
        match="no source-boundary evidence",
    ):
        mark_turn_boundary_clipped(
            storage,
            "turn_000001",
        )

    assert storage.turns.path.read_bytes() == before

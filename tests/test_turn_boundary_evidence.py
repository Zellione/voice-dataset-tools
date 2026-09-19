from __future__ import annotations

from pathlib import Path

import pytest

from voice_dataset.reconciliation import (
    create_turn_from_regions,
)
from voice_dataset.schema import (
    AudioRepresentation,
    SourceRecord,
)
from voice_dataset.storage import DatasetStorage
from voice_dataset.boundary_evidence import (
    refresh_source_boundary_evidence,
    refresh_turn_boundary_evidence,
)

def make_storage(
    tmp_path: Path,
    *,
    with_boundary_representation: bool = True,
) -> DatasetStorage:
    storage = DatasetStorage(
        tmp_path / "dataset"
    )

    representations = {}

    if with_boundary_representation:
        representations["center"] = (
            AudioRepresentation(
                path="/scratch/center.wav",
                kind="center",
                duration=60.0,
                purposes=[
                    "boundary_analysis",
                ],
            )
        )

    storage.add_source(
        SourceRecord(
            id="source_001",
            media_path="/media/source.mkv",
            representations=representations,
        )
    )

    return storage


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
            "record_type": "candidate_region",
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


def test_create_turn_persists_boundary_evidence(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    add_region(
        storage,
        region_id="region_000001",
        start=0.03096875,
        end=1.027,
    )

    turn = create_turn_from_regions(
        storage,
        ["region_000001"],
    )

    assert turn["metadata"]["boundary_evidence"] == {
        "representation": "center",
        "margin": 0.1,
        "near_source_start": True,
        "near_source_end": False,
    }


def test_create_turn_without_boundary_representation(
    tmp_path: Path,
) -> None:
    storage = make_storage(
        tmp_path,
        with_boundary_representation=False,
    )

    add_region(
        storage,
        region_id="region_000001",
        start=1.0,
        end=2.0,
    )

    turn = create_turn_from_regions(
        storage,
        ["region_000001"],
    )

    assert (
        "boundary_evidence"
        not in turn["metadata"]
    )


def test_create_turn_rejects_multiple_boundary_representations(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    source = storage.get_source("source_001")
    assert source is not None

    def update(record):
        record["representations"]["raw"] = {
            "path": "/scratch/raw.wav",
            "kind": "raw",
            "duration": 60.0,
            "purposes": [
                "boundary_analysis",
            ],
            "metadata": {},
        }
        return record

    storage.update_source(
        "source_001",
        update,
    )

    add_region(
        storage,
        region_id="region_000001",
        start=1.0,
        end=2.0,
    )

    with pytest.raises(
        ValueError,
        match=(
            "multiple boundary_analysis "
            "representations"
        ),
    ):
        create_turn_from_regions(
            storage,
            ["region_000001"],
        )


def add_existing_turn(
    storage: DatasetStorage,
    *,
    turn_id: str = "turn_000001",
    start: float = 0.03096875,
    end: float = 1.027,
) -> None:
    storage.turns.append(
        {
            "schema_version": 2,
            "record_type": "turn",
            "id": turn_id,
            "source_id": "source_001",
            "source_start": start,
            "source_end": end,
            "source_regions": [
                "region_000001",
            ],
            "language": "en",
            "transcript": "Keep this.",
            "representations": {
                "raw": {
                    "path": (
                        "turns/turn_000001/raw.wav"
                    ),
                },
            },
            "embeddings": {
                "test": {
                    "model": "test-model",
                },
            },
            "assignment": {
                "status": "unknown",
                "voice_id": None,
                "character_id": None,
            },
            "review": {
                "status": "pending",
            },
            "metadata": {
                "keep": "unchanged",
            },
        }
    )


def test_refresh_turn_boundary_evidence(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)
    add_existing_turn(storage)

    turn = refresh_turn_boundary_evidence(
        storage,
        "turn_000001",
    )

    assert turn["metadata"]["boundary_evidence"] == {
        "representation": "center",
        "margin": 0.1,
        "near_source_start": True,
        "near_source_end": False,
    }

    assert turn["metadata"]["keep"] == "unchanged"
    assert turn["transcript"] == "Keep this."
    assert turn["language"] == "en"
    assert turn["representations"]["raw"] == {
        "path": "turns/turn_000001/raw.wav",
    }
    assert turn["embeddings"]["test"] == {
        "model": "test-model",
    }
    assert turn["review"]["status"] == "pending"


def test_refresh_turn_boundary_evidence_is_idempotent(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)
    add_existing_turn(storage)

    first = refresh_turn_boundary_evidence(
        storage,
        "turn_000001",
    )

    first_bytes = storage.turns.path.read_bytes()

    second = refresh_turn_boundary_evidence(
        storage,
        "turn_000001",
    )

    second_bytes = storage.turns.path.read_bytes()

    assert second == first
    assert second_bytes == first_bytes


def test_refresh_removes_stale_boundary_evidence(
    tmp_path: Path,
) -> None:
    storage = make_storage(
        tmp_path,
        with_boundary_representation=False,
    )
    add_existing_turn(storage)

    def add_stale_evidence(
        record: dict,
    ) -> dict:
        record["metadata"][
            "boundary_evidence"
        ] = {
            "representation": "old",
            "margin": 0.1,
            "near_source_start": True,
            "near_source_end": False,
        }
        return record

    storage.update_turn(
        "turn_000001",
        add_stale_evidence,
    )

    turn = refresh_turn_boundary_evidence(
        storage,
        "turn_000001",
    )

    assert (
        "boundary_evidence"
        not in turn["metadata"]
    )
    assert turn["metadata"]["keep"] == "unchanged"


def test_refresh_unknown_turn_is_rejected(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    with pytest.raises(
        KeyError,
        match="Unknown turn",
    ):
        refresh_turn_boundary_evidence(
            storage,
            "turn_999999",
        )


def test_refresh_source_boundary_evidence(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    add_existing_turn(
        storage,
        turn_id="turn_000001",
        start=0.03,
        end=1.0,
    )

    add_existing_turn(
        storage,
        turn_id="turn_000002",
        start=10.0,
        end=12.0,
    )

    add_existing_turn(
        storage,
        turn_id="turn_000003",
        start=58.0,
        end=59.95,
    )

    turns = refresh_source_boundary_evidence(
        storage,
        "source_001",
    )

    assert [
        turn["id"]
        for turn in turns
    ] == [
        "turn_000001",
        "turn_000002",
        "turn_000003",
    ]

    assert turns[0]["metadata"][
        "boundary_evidence"
    ]["near_source_start"] is True

    assert turns[0]["metadata"][
        "boundary_evidence"
    ]["near_source_end"] is False

    assert turns[1]["metadata"][
        "boundary_evidence"
    ]["near_source_start"] is False

    assert turns[1]["metadata"][
        "boundary_evidence"
    ]["near_source_end"] is False

    assert turns[2]["metadata"][
        "boundary_evidence"
    ]["near_source_start"] is False

    assert turns[2]["metadata"][
        "boundary_evidence"
    ]["near_source_end"] is True


def test_refresh_source_boundary_evidence_unknown_source(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    with pytest.raises(
        KeyError,
        match="Unknown source",
    ):
        refresh_source_boundary_evidence(
            storage,
            "source_999",
        )

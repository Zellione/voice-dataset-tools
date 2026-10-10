from __future__ import annotations

from pathlib import Path

import pytest

from voice_dataset.reconciliation import merge_turns
from voice_dataset.schema import VoiceAssignment
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
    turn_id: str,
    start: float,
    end: float,
    region_ids: list[str],
    transcript: str | None = None,
    language: str | None = "en",
    voice_id: str | None = None,
    status: str = "unknown",
) -> None:
    assignment = VoiceAssignment(
        status=status,
        voice_id=voice_id,
        method=(
            "manual"
            if status == "assigned"
            else None
        ),
    )

    storage.turns.append(
        {
            "schema_version": 2,
            "record_type": "turn",
            "id": turn_id,
            "source_id": "source_001",
            "source_start": start,
            "source_end": end,
            "source_regions": region_ids,
            "language": language,
            "transcript": transcript,
            "representations": {},
            "embeddings": {},
            "assignment": assignment.to_dict(),
            "review": {
                "status": "pending",
            },
            "metadata": {},
        }
    )


def test_merge_shared_region_preserves_turn_semantics(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    add_region(
        storage,
        region_id="region_000001",
        start=0.9,
        end=2.1,
    )

    add_turn(
        storage,
        turn_id="turn_000001",
        start=1.0,
        end=1.4,
        region_ids=["region_000001"],
        transcript="They're right",
        voice_id="voice_001",
        status="assigned",
    )
    add_turn(
        storage,
        turn_id="turn_000002",
        start=1.4,
        end=2.0,
        region_ids=["region_000001"],
        transcript="not to trust us",
        voice_id="voice_001",
        status="assigned",
    )

    merged = merge_turns(
        storage,
        [
            "turn_000001",
            "turn_000002",
        ],
    )

    assert merged["id"] == "turn_000001"

    assert merged["source_start"] == 1.0
    assert merged["source_end"] == 2.0

    assert merged["source_regions"] == [
        "region_000001",
    ]

    assert merged["language"] == "en"
    assert (
        merged["transcript"]
        == "They're right not to trust us"
    )

    assert merged["assignment"] == {
        "status": "assigned",
        "voice_id": "voice_001",
        "method": "manual_merge",
        "confidence": None,
    }

    assert merged["representations"] == {}
    assert merged["embeddings"] == {}
    assert merged["review"]["status"] == "pending"

    assert (
        merged["metadata"]["creation"]["method"]
        == "manual_merge"
    )
    assert (
        merged["metadata"]["creation"][
            "source_turn_ids"
        ]
        == [
            "turn_000001",
            "turn_000002",
        ]
    )

    turns = storage.turns.load()

    assert [turn["id"] for turn in turns] == [
        "turn_000001",
    ]


def test_merge_assigned_and_unknown_preserves_voice(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    add_region(
        storage,
        region_id="region_000001",
        start=1.0,
        end=2.0,
    )

    add_turn(
        storage,
        turn_id="turn_000001",
        start=1.0,
        end=1.5,
        region_ids=["region_000001"],
        transcript="You're walking a fine line",
        voice_id="voice_005",
        status="assigned",
    )
    add_turn(
        storage,
        turn_id="turn_000002",
        start=1.5,
        end=2.0,
        region_ids=["region_000001"],
        transcript="Jace",
    )

    merged = merge_turns(
        storage,
        [
            "turn_000001",
            "turn_000002",
        ],
    )

    assert (
        merged["transcript"]
        == "You're walking a fine line Jace"
    )
    assert merged["assignment"] == {
        "status": "assigned",
        "voice_id": "voice_005",
        "method": "manual_merge",
        "confidence": None,
    }


def test_merge_unknown_turns_stays_unknown(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    add_region(
        storage,
        region_id="region_000001",
        start=1.0,
        end=2.0,
    )

    add_turn(
        storage,
        turn_id="turn_000001",
        start=1.0,
        end=1.4,
        region_ids=["region_000001"],
        transcript="With respect",
    )
    add_turn(
        storage,
        turn_id="turn_000002",
        start=1.4,
        end=2.0,
        region_ids=["region_000001"],
        transcript="I don't give a shit",
    )

    merged = merge_turns(
        storage,
        [
            "turn_000001",
            "turn_000002",
        ],
    )

    assert merged["assignment"]["status"] == "unknown"
    assert merged["assignment"]["voice_id"] is None


def test_merge_rejects_conflicting_voices(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    add_region(
        storage,
        region_id="region_000001",
        start=1.0,
        end=2.0,
    )

    add_turn(
        storage,
        turn_id="turn_000001",
        start=1.0,
        end=1.5,
        region_ids=["region_000001"],
        voice_id="voice_001",
        status="assigned",
    )
    add_turn(
        storage,
        turn_id="turn_000002",
        start=1.5,
        end=2.0,
        region_ids=["region_000001"],
        voice_id="voice_002",
        status="assigned",
    )

    before = storage.turns.path.read_bytes()

    with pytest.raises(
        ValueError,
        match="conflicting voice assignments",
    ):
        merge_turns(
            storage,
            [
                "turn_000001",
                "turn_000002",
            ],
        )

    assert storage.turns.path.read_bytes() == before


def test_manual_merge_allows_region_used_by_outside_turn(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    add_region(
        storage,
        region_id="region_000001",
        start=1.0,
        end=3.0,
    )

    add_turn(
        storage,
        turn_id="turn_000001",
        start=1.0,
        end=1.5,
        region_ids=["region_000001"],
        transcript="first",
    )
    add_turn(
        storage,
        turn_id="turn_000002",
        start=1.5,
        end=2.0,
        region_ids=["region_000001"],
        transcript="second",
    )
    add_turn(
        storage,
        turn_id="turn_000003",
        start=2.0,
        end=3.0,
        region_ids=["region_000001"],
        transcript="outside",
    )

    merged = merge_turns(
        storage,
        [
            "turn_000001",
            "turn_000002",
        ],
    )

    assert merged["id"] == "turn_000001"
    assert merged["source_start"] == 1.0
    assert merged["source_end"] == 2.0
    assert merged["transcript"] == "first second"
    assert merged["source_regions"] == [
        "region_000001",
    ]

    remaining = {
        turn["id"]: turn
        for turn in storage.turns.load()
    }

    assert set(remaining) == {
        "turn_000001",
        "turn_000003",
    }

    # The outside turn keeps the same provenance.
    assert remaining["turn_000003"][
        "source_regions"
    ] == ["region_000001"]

def test_unrelated_shared_region_does_not_block_merge(
    tmp_path: Path,
) -> None:
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
        start=3.0,
        end=4.0,
    )

    add_turn(
        storage,
        turn_id="turn_000001",
        start=1.0,
        end=1.5,
        region_ids=["region_000001"],
    )
    add_turn(
        storage,
        turn_id="turn_000002",
        start=1.5,
        end=2.0,
        region_ids=["region_000001"],
    )

    add_turn(
        storage,
        turn_id="turn_000003",
        start=3.0,
        end=3.5,
        region_ids=["region_000002"],
    )
    add_turn(
        storage,
        turn_id="turn_000004",
        start=3.5,
        end=4.0,
        region_ids=["region_000002"],
    )

    merged = merge_turns(
        storage,
        [
            "turn_000001",
            "turn_000002",
        ],
    )

    assert merged["id"] == "turn_000001"

    remaining = {
        turn["id"]: turn
        for turn in storage.turns.load()
    }

    assert set(remaining) == {
        "turn_000001",
        "turn_000003",
        "turn_000004",
    }

    assert remaining["turn_000003"][
        "source_regions"
    ] == ["region_000002"]

    assert remaining["turn_000004"][
        "source_regions"
    ] == ["region_000002"]


def test_merge_collects_regions_from_all_turns(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    add_region(
        storage,
        region_id="region_000001",
        start=1.0,
        end=1.5,
    )
    add_region(
        storage,
        region_id="region_000002",
        start=1.5,
        end=2.0,
    )

    add_turn(
        storage,
        turn_id="turn_000001",
        start=1.0,
        end=1.5,
        region_ids=["region_000001"],
        transcript="first part",
    )
    add_turn(
        storage,
        turn_id="turn_000002",
        start=1.5,
        end=2.0,
        region_ids=["region_000002"],
        transcript="second part",
    )

    merged = merge_turns(
        storage,
        [
            "turn_000001",
            "turn_000002",
        ],
    )

    assert merged["source_regions"] == [
        "region_000001",
        "region_000002",
    ]

    assert merged["source_start"] == 1.0
    assert merged["source_end"] == 2.0
    assert (
        merged["transcript"]
        == "first part second part"
    )


def test_manual_merge_preserves_retained_review_queue_status(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    add_region(
        storage,
        region_id="region_000001",
        start=1.0,
        end=2.0,
    )

    add_turn(
        storage,
        turn_id="turn_000001",
        start=1.0,
        end=1.5,
        region_ids=["region_000001"],
        transcript="first half",
    )

    add_turn(
        storage,
        turn_id="turn_000002",
        start=1.5,
        end=2.0,
        region_ids=["region_000001"],
        transcript="second half",
    )

    def mark_review_candidate(record):
        metadata = dict(
            record.get("metadata")
            or {}
        )

        metadata["automatic_pipeline"] = {
            "status": "review",
            "reasons": [
                "unresolved_words",
            ],
        }

        record["metadata"] = metadata
        return record

    storage.update_turn(
        "turn_000001",
        mark_review_candidate,
    )

    merged = merge_turns(
        storage,
        [
            "turn_000001",
            "turn_000002",
        ],
    )

    assert merged["metadata"][
        "automatic_pipeline"
    ] == {
        "status": "review",
        "reasons": [
            "unresolved_words",
        ],
    }

    assert merged["metadata"][
        "creation"
    ] == {
        "method": "manual_merge",
        "source_turn_ids": [
            "turn_000001",
            "turn_000002",
        ],
    }

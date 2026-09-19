from pathlib import Path

from voice_dataset.reviewer import (
    format_turn,
    sorted_turns,
    turn_assignment_text,
    turn_review_status,
)
from voice_dataset.schema import TurnRecord
from voice_dataset.storage import DatasetStorage


def add_turn(
    storage: DatasetStorage,
    *,
    turn_id: str,
    source_id: str,
    start: float,
    end: float,
) -> None:
    storage.add_turn(
        TurnRecord(
            id=turn_id,
            source_id=source_id,
            source_start=start,
            source_end=end,
        )
    )


def test_sorted_turns_uses_source_timeline(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        turn_id="turn_000001",
        source_id="source-b",
        start=1.0,
        end=2.0,
    )
    add_turn(
        storage,
        turn_id="turn_000003",
        source_id="source-a",
        start=5.0,
        end=6.0,
    )
    add_turn(
        storage,
        turn_id="turn_000002",
        source_id="source-a",
        start=1.0,
        end=2.0,
    )

    result = sorted_turns(storage)

    assert [
        turn["id"]
        for turn in result
    ] == [
        "turn_000002",
        "turn_000003",
        "turn_000001",
    ]


def test_sorted_turns_can_filter_source(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        turn_id="turn_000001",
        source_id="source-a",
        start=1.0,
        end=2.0,
    )
    add_turn(
        storage,
        turn_id="turn_000002",
        source_id="source-b",
        start=1.0,
        end=2.0,
    )

    result = sorted_turns(
        storage,
        source_id="source-b",
    )

    assert [
        turn["id"]
        for turn in result
    ] == [
        "turn_000002",
    ]


def test_turn_review_status_defaults_pending():
    assert (
        turn_review_status({})
        == "pending"
    )


def test_turn_assignment_text_unknown():
    assert (
        turn_assignment_text({})
        == "unknown"
    )


def test_turn_assignment_text_assigned():
    turn = {
        "assignment": {
            "status": "assigned",
            "voice_id": "voice_001",
        },
    }

    assert (
        turn_assignment_text(turn)
        == "assigned -> voice_001"
    )


def test_format_turn():
    turn = {
        "id": "turn_000005",
        "source_id": "arcane-s01e09-test",
        "source_start": 8.468,
        "source_end": 8.941,
        "source_regions": [
            "region_000003",
        ],
        "language": "en",
        "transcript": "I know.",
        "representations": {
            "raw": {},
            "speech": {},
        },
        "assignment": {
            "status": "unknown",
            "voice_id": None,
        },
        "review": {
            "status": "pending",
        },
    }

    result = format_turn(
        turn,
        position=3,
        total=12,
    )

    assert "[3/12] turn_000005" in result
    assert "arcane-s01e09-test" in result
    assert "8.468-8.941 (0.473s)" in result
    assert "region_000003" in result
    assert "language:        en" in result
    assert "assignment:      unknown" in result
    assert "review:          pending" in result
    assert "representations: raw, speech" in result
    assert "transcript:      I know." in result


def test_raw_representation():
    from voice_dataset.reviewer import (
        raw_representation,
    )

    raw = {
        "path": "turns/turn_000001/raw.wav",
    }

    turn = {
        "id": "turn_000001",
        "representations": {
            "raw": raw,
        },
    }

    assert raw_representation(turn) is raw


def test_raw_representation_missing():
    import pytest

    from voice_dataset.reviewer import (
        raw_representation,
    )

    with pytest.raises(
        ValueError,
        match="has no raw representation",
    ):
        raw_representation(
            {
                "id": "turn_000001",
                "representations": {},
            }
        )

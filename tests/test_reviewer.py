from pathlib import Path

from voice_dataset.reviewer import (
    format_turn,
    format_voice_matches,
    sorted_turns,
    turn_assignment_text,
    turn_boundary_review_status,
    turn_boundary_text,
    turn_review_status,
)
from voice_dataset.schema import TurnRecord
from voice_dataset.storage import DatasetStorage
from voice_dataset.speaker_similarity import (
    VoiceMatch,
    VoiceTurnMatch,
)


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


def test_turn_boundary_review_status_defaults_unknown():
    assert (
        turn_boundary_review_status({})
        == "unknown"
    )


def test_turn_boundary_text_without_evidence():
    assert turn_boundary_text({}) == "-"


def test_turn_boundary_text_near_start_unknown():
    turn = {
        "metadata": {
            "boundary_evidence": {
                "representation": "center",
                "margin": 0.1,
                "near_source_start": True,
                "near_source_end": False,
            },
        },
        "review": {
            "status": "pending",
        },
    }

    assert (
        turn_boundary_text(turn)
        == "near start -> unknown"
    )


def test_turn_boundary_text_near_end_clipped():
    turn = {
        "metadata": {
            "boundary_evidence": {
                "representation": "center",
                "margin": 0.1,
                "near_source_start": False,
                "near_source_end": True,
            },
        },
        "review": {
            "status": "reviewed",
            "boundary": {
                "status": "clipped",
            },
        },
    }

    assert (
        turn_boundary_text(turn)
        == "near end -> clipped"
    )


def test_turn_boundary_text_near_both_complete():
    turn = {
        "metadata": {
            "boundary_evidence": {
                "near_source_start": True,
                "near_source_end": True,
            },
        },
        "review": {
            "boundary": {
                "status": "complete",
            },
        },
    }

    assert (
        turn_boundary_text(turn)
        == "near start/end -> complete"
    )


def test_format_turn_shows_boundary_status():
    turn = {
        "id": "turn_000003",
        "source_id": "arcane-s01e09-test",
        "source_start": 0.031,
        "source_end": 1.027,
        "metadata": {
            "boundary_evidence": {
                "representation": "center",
                "margin": 0.1,
                "near_source_start": True,
                "near_source_end": False,
            },
        },
        "review": {
            "status": "pending",
        },
    }

    result = format_turn(
        turn,
        position=1,
        total=12,
    )

    assert (
        "boundary:        near start -> unknown"
        in result
    )


def test_format_voice_matches():
    matches = [
        VoiceMatch(
            voice_id="voice_001",
            matches=(
                VoiceTurnMatch(
                    turn_id="turn_000009",
                    similarity=0.3734,
                ),
                VoiceTurnMatch(
                    turn_id="turn_000008",
                    similarity=0.3566,
                ),
                VoiceTurnMatch(
                    turn_id="turn_000003",
                    similarity=0.3378,
                ),
                VoiceTurnMatch(
                    turn_id="turn_000005",
                    similarity=0.1512,
                ),
            ),
        ),
        VoiceMatch(
            voice_id="voice_002",
            matches=(
                VoiceTurnMatch(
                    turn_id="turn_000012",
                    similarity=0.4781,
                ),
                VoiceTurnMatch(
                    turn_id="turn_000010",
                    similarity=0.4432,
                ),
            ),
        ),
    ]

    voices = {
        "voice_001": {
            "character": "Jayce",
        },
        "voice_002": {
            "character": "Viktor",
        },
    }

    result = format_voice_matches(
        matches,
        voices,
    )

    assert result == "\n".join([
        "voice_001 (Jayce)",
        "  turn_000009: 0.373",
        "  turn_000008: 0.357",
        "  turn_000003: 0.338",
        "voice_002 (Viktor)",
        "  turn_000012: 0.478",
        "  turn_000010: 0.443",
    ])

from __future__ import annotations

from pathlib import Path

import pytest

from voice_dataset.reconciliation import edit_turn
from voice_dataset.storage import DatasetStorage


def make_turn(
    tmp_path: Path,
    *,
    transcript: str | None = "Original transcript.",
    language: str | None = "en",
) -> tuple[DatasetStorage, str]:
    storage = DatasetStorage(
        tmp_path / "dataset"
    )

    turn_id = "turn_000001"

    storage.turns.append(
        {
            "schema_version": 2,
            "id": turn_id,
            "source_id": "source_001",
            "source_start": 1.0,
            "source_end": 2.0,
            "source_regions": [
                "region_000001",
            ],
            "language": language,
            "transcript": transcript,
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

    return storage, turn_id


def test_edit_turn_sets_transcript_and_language(
    tmp_path: Path,
) -> None:
    storage, turn_id = make_turn(
        tmp_path,
        transcript=None,
        language=None,
    )

    turn = edit_turn(
        storage,
        turn_id,
        transcript="New transcript.",
        language="en",
    )

    assert turn["transcript"] == "New transcript."
    assert turn["language"] == "en"


def test_edit_turn_transcript_preserves_other_fields(
    tmp_path: Path,
) -> None:
    storage, turn_id = make_turn(tmp_path)

    before = storage.get_turn(turn_id)
    assert before is not None

    turn = edit_turn(
        storage,
        turn_id,
        transcript="Corrected transcript.",
    )

    assert turn["transcript"] == "Corrected transcript."
    assert turn["language"] == "en"

    assert (
        turn["representations"]
        == before["representations"]
    )
    assert turn["embeddings"] == before["embeddings"]
    assert turn["assignment"] == before["assignment"]
    assert turn["review"] == before["review"]
    assert turn["metadata"] == before["metadata"]


def test_edit_turn_language_preserves_transcript(
    tmp_path: Path,
) -> None:
    storage, turn_id = make_turn(tmp_path)

    turn = edit_turn(
        storage,
        turn_id,
        language="de",
    )

    assert turn["language"] == "de"
    assert turn["transcript"] == "Original transcript."


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("transcript", ""),
        ("transcript", "   "),
        ("language", ""),
        ("language", "   "),
    ],
)
def test_edit_turn_rejects_empty_values_without_mutation(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    storage, turn_id = make_turn(tmp_path)

    before = storage.turns.path.read_bytes()

    with pytest.raises(ValueError):
        edit_turn(
            storage,
            turn_id,
            **{field: value},
        )

    assert storage.turns.path.read_bytes() == before


def test_edit_turn_unknown_id_does_not_mutate(
    tmp_path: Path,
) -> None:
    storage, _ = make_turn(tmp_path)

    before = storage.turns.path.read_bytes()

    with pytest.raises(
        KeyError,
        match="Unknown turn",
    ):
        edit_turn(
            storage,
            "turn_999999",
            transcript="Nope.",
        )

    assert storage.turns.path.read_bytes() == before


def test_edit_turn_same_values_is_idempotent(
    tmp_path: Path,
) -> None:
    storage, turn_id = make_turn(tmp_path)

    before = storage.turns.path.read_bytes()

    turn = edit_turn(
        storage,
        turn_id,
        transcript="Original transcript.",
        language="en",
    )

    after = storage.turns.path.read_bytes()

    assert turn["transcript"] == "Original transcript."
    assert turn["language"] == "en"
    assert after == before

from __future__ import annotations

from pathlib import Path

import pytest

import voice_dataset.representations as representations
from voice_dataset.storage import DatasetStorage


def make_storage(
    tmp_path: Path,
) -> tuple[DatasetStorage, Path]:
    storage = DatasetStorage(tmp_path / "dataset")

    source_audio = tmp_path / "source.wav"
    source_audio.touch()

    for turn_id, start, end in [
        ("turn_000001", 1.0, 2.0),
        ("turn_000002", 2.1, 3.0),
        ("turn_000003", 3.4, 4.0),
    ]:
        storage.turns.append(
            {
                "schema_version": 2,
                "record_type": "turn",
                "id": turn_id,
                "source_id": "source_001",
                "source_start": start,
                "source_end": end,
                "source_regions": [],
                "language": "en",
                "transcript": "Test.",
                "representations": {},
                "embeddings": {},
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

    return storage, source_audio


def fake_extractor(calls):
    def extract(
        source,
        destination,
        start,
        end,
    ):
        calls.append(
            {
                "source": source,
                "destination": destination,
                "start": start,
                "end": end,
            }
        )
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        destination.touch()

    return extract


def prepare_audio_mocks(
    monkeypatch: pytest.MonkeyPatch,
    calls: list,
) -> None:
    monkeypatch.setattr(
        representations,
        "probe_representation_source",
        lambda source: (48000, 1),
    )

    monkeypatch.setattr(
        representations,
        "extract_audio_region",
        fake_extractor(calls),
    )


def test_review_turn_audio_uses_neighbor_bounded_padding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, source_audio = make_storage(tmp_path)
    calls = []

    prepare_audio_mocks(
        monkeypatch,
        calls,
    )

    result = representations.materialize_turns(
        storage=storage,
        source_id="source_001",
        source=source_audio,
        representation_name="speech",
        kind="speech",
        purposes=[
            "review",
            "tts_candidate",
        ],
    )

    assert result.created == 3
    assert result.skipped == 0

    actual_ranges = [
        (call["start"], call["end"])
        for call in calls
    ]

    expected_ranges = [
        (0.75, 2.1),
        (2.0, 3.25),
        (3.15, 4.25),
    ]

    for actual, expected in zip(
        actual_ranges,
        expected_ranges,
        strict=True,
    ):
        assert actual == pytest.approx(expected)

    turns = {
        turn["id"]: turn
        for turn in storage.turns.load()
    }

    metadata = turns["turn_000002"][
        "representations"
    ]["speech"]["metadata"]

    assert metadata[
        "canonical_start"
    ] == pytest.approx(2.1)

    assert metadata[
        "canonical_end"
    ] == pytest.approx(3.0)

    assert metadata[
        "clip_start"
    ] == pytest.approx(2.0)

    assert metadata[
        "clip_end"
    ] == pytest.approx(3.25)

    assert metadata[
        "padding_before"
    ] == pytest.approx(0.1)

    assert metadata[
        "padding_after"
    ] == pytest.approx(0.25)


def test_speaker_embedding_turn_audio_is_not_padded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, source_audio = make_storage(tmp_path)
    calls = []

    prepare_audio_mocks(
        monkeypatch,
        calls,
    )

    result = representations.materialize_turns(
        storage=storage,
        source_id="source_001",
        source=source_audio,
        representation_name="embedding",
        kind="center",
        purposes=[
            "speaker_embedding",
        ],
    )

    assert result.created == 3
    assert result.skipped == 0

    actual_ranges = [
        (call["start"], call["end"])
        for call in calls
    ]

    expected_ranges = [
        (1.0, 2.0),
        (2.1, 3.0),
        (3.4, 4.0),
    ]

    for actual, expected in zip(
        actual_ranges,
        expected_ranges,
        strict=True,
    ):
        assert actual == pytest.approx(expected)

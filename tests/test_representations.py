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


def test_materialize_turns_rebuilds_stale_turn_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, source_audio = make_storage(tmp_path)
    calls = []

    prepare_audio_mocks(
        monkeypatch,
        calls,
    )

    first = representations.materialize_turns(
        storage=storage,
        source_id="source_001",
        source=source_audio,
        representation_name="embedding",
        kind="center",
        purposes=[
            "speaker_embedding",
        ],
    )

    assert first.created == 3
    assert first.skipped == 0

    calls.clear()

    def trim_turn(record):
        record["source_end"] = 2.8
        return record

    storage.update_turn(
        "turn_000002",
        trim_turn,
    )

    second = representations.materialize_turns(
        storage=storage,
        source_id="source_001",
        source=source_audio,
        representation_name="embedding",
        kind="center",
        purposes=[
            "speaker_embedding",
        ],
    )

    assert second.created == 1
    assert second.skipped == 2

    assert len(calls) == 1
    assert calls[0]["start"] == pytest.approx(2.1)
    assert calls[0]["end"] == pytest.approx(2.8)

    turn = storage.get_turn(
        "turn_000002"
    )

    assert turn is not None

    metadata = turn["representations"][
        "embedding"
    ]["metadata"]

    assert metadata[
        "canonical_start"
    ] == pytest.approx(2.1)

    assert metadata[
        "canonical_end"
    ] == pytest.approx(2.8)

    assert metadata[
        "clip_start"
    ] == pytest.approx(2.1)

    assert metadata[
        "clip_end"
    ] == pytest.approx(2.8)


def test_materialize_turns_rebuilds_stale_neighbor_padding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, source_audio = make_storage(tmp_path)
    calls = []

    prepare_audio_mocks(
        monkeypatch,
        calls,
    )

    first = representations.materialize_turns(
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

    assert first.created == 3
    assert first.skipped == 0

    calls.clear()

    def trim_turn(record):
        record["source_start"] = 2.2
        return record

    storage.update_turn(
        "turn_000002",
        trim_turn,
    )

    second = representations.materialize_turns(
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

    assert second.created == 2
    assert second.skipped == 1

    actual_ranges = [
        (
            call["start"],
            call["end"],
        )
        for call in calls
    ]

    assert actual_ranges == pytest.approx([
        (0.75, 2.2),
        (2.0, 3.25),
    ])


def test_rebuilding_turn_representation_invalidates_dependent_embeddings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, source_audio = make_storage(tmp_path)
    calls = []

    prepare_audio_mocks(
        monkeypatch,
        calls,
    )

    representations.materialize_turns(
        storage=storage,
        source_id="source_001",
        source=source_audio,
        representation_name="speaker",
        kind="center",
        purposes=[
            "speaker_embedding",
        ],
    )

    embedding_paths = []

    for name in (
        "ecapa_speaker",
        "wespeaker_speaker",
    ):
        relative_path = (
            f"turns/turn_000002/"
            f"embeddings/{name}.npy"
        )

        path = storage.root / relative_path
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.touch()

        embedding_paths.append(path)

        def add_embedding(
            record,
            *,
            embedding_name=name,
            embedding_path=relative_path,
        ):
            record["embeddings"][
                embedding_name
            ] = {
                "encoder": embedding_name,
                "representation": "speaker",
                "path": embedding_path,
                "dimension": 3,
                "metadata": {
                    "model": "test-model",
                    "sha256": "test",
                },
            }
            return record

        storage.update_turn(
            "turn_000002",
            add_embedding,
        )

    def trim_turn(record):
        record["source_end"] = 2.8
        return record

    storage.update_turn(
        "turn_000002",
        trim_turn,
    )

    representations.materialize_turns(
        storage=storage,
        source_id="source_001",
        source=source_audio,
        representation_name="speaker",
        kind="center",
        purposes=[
            "speaker_embedding",
        ],
    )

    turn = storage.get_turn(
        "turn_000002"
    )

    assert turn is not None
    assert turn["embeddings"] == {}

    for path in embedding_paths:
        assert not path.exists()


def test_rebuilding_turn_representation_keeps_unrelated_embeddings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, source_audio = make_storage(tmp_path)
    calls = []

    prepare_audio_mocks(
        monkeypatch,
        calls,
    )

    representations.materialize_turns(
        storage=storage,
        source_id="source_001",
        source=source_audio,
        representation_name="speaker",
        kind="center",
        purposes=[
            "speaker_embedding",
        ],
    )

    unrelated_path = (
        storage.root
        / "turns"
        / "turn_000002"
        / "embeddings"
        / "other.npy"
    )
    unrelated_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    unrelated_path.touch()

    def add_embedding(record):
        record["embeddings"]["other"] = {
            "encoder": "test",
            "representation": "review",
            "path": (
                "turns/turn_000002/"
                "embeddings/other.npy"
            ),
            "dimension": 3,
            "metadata": {
                "model": "test-model",
                "sha256": "test",
            },
        }
        return record

    storage.update_turn(
        "turn_000002",
        add_embedding,
    )

    def trim_turn(record):
        record["source_end"] = 2.8
        return record

    storage.update_turn(
        "turn_000002",
        trim_turn,
    )

    representations.materialize_turns(
        storage=storage,
        source_id="source_001",
        source=source_audio,
        representation_name="speaker",
        kind="center",
        purposes=[
            "speaker_embedding",
        ],
    )

    turn = storage.get_turn(
        "turn_000002"
    )

    assert turn is not None
    assert "other" in turn["embeddings"]
    assert unrelated_path.exists()


def test_failed_representation_rebuild_preserves_existing_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, source_audio = make_storage(tmp_path)
    calls = []

    prepare_audio_mocks(
        monkeypatch,
        calls,
    )

    representations.materialize_turns(
        storage=storage,
        source_id="source_001",
        source=source_audio,
        representation_name="speaker",
        kind="center",
        purposes=[
            "speaker_embedding",
        ],
    )

    turn = storage.get_turn(
        "turn_000002"
    )
    assert turn is not None

    old_representation = dict(
        turn["representations"]["speaker"]
    )

    speaker_path = (
        storage.root
        / old_representation["path"]
    )

    speaker_path.write_bytes(
        b"old-speaker-audio"
    )

    embedding_path = (
        storage.root
        / "turns"
        / "turn_000002"
        / "embeddings"
        / "ecapa_speaker.npy"
    )
    embedding_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    embedding_path.write_bytes(
        b"old-embedding"
    )

    def add_embedding(record):
        record["embeddings"][
            "ecapa_speaker"
        ] = {
            "encoder": "ecapa",
            "representation": "speaker",
            "path": (
                "turns/turn_000002/"
                "embeddings/ecapa_speaker.npy"
            ),
            "dimension": 3,
            "metadata": {
                "model": "test-model",
                "sha256": "test",
            },
        }
        return record

    storage.update_turn(
        "turn_000002",
        add_embedding,
    )

    def trim_turn(record):
        record["source_end"] = 2.8
        return record

    storage.update_turn(
        "turn_000002",
        trim_turn,
    )

    def fail_extraction(
        source,
        destination,
        start,
        end,
    ):
        raise RuntimeError(
            "simulated extraction failure"
        )

    monkeypatch.setattr(
        representations,
        "extract_audio_region",
        fail_extraction,
    )

    with pytest.raises(
        RuntimeError,
        match="simulated extraction failure",
    ):
        representations.materialize_turns(
            storage=storage,
            source_id="source_001",
            source=source_audio,
            representation_name="speaker",
            kind="center",
            purposes=[
                "speaker_embedding",
            ],
        )

    turn = storage.get_turn(
        "turn_000002"
    )
    assert turn is not None

    assert (
        turn["representations"]["speaker"]
        == old_representation
    )

    assert (
        "ecapa_speaker"
        in turn["embeddings"]
    )

    assert speaker_path.read_bytes() == (
        b"old-speaker-audio"
    )

    assert embedding_path.read_bytes() == (
        b"old-embedding"
    )


def test_failed_metadata_update_rolls_back_rebuilt_representation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, source_audio = make_storage(tmp_path)
    calls = []

    prepare_audio_mocks(
        monkeypatch,
        calls,
    )

    representations.materialize_turns(
        storage=storage,
        source_id="source_001",
        source=source_audio,
        representation_name="speaker",
        kind="center",
        purposes=[
            "speaker_embedding",
        ],
    )

    turn = storage.get_turn(
        "turn_000002"
    )
    assert turn is not None

    speaker_path = (
        storage.root
        / turn["representations"]["speaker"]["path"]
    )
    speaker_path.write_bytes(
        b"old-speaker-audio"
    )

    def trim_turn(record):
        record["source_end"] = 2.8
        return record

    storage.update_turn(
        "turn_000002",
        trim_turn,
    )

    original_update_turn = storage.update_turn

    def fail_update_turn(
        turn_id,
        updater,
    ):
        if turn_id == "turn_000002":
            raise RuntimeError(
                "simulated metadata failure"
            )

        return original_update_turn(
            turn_id,
            updater,
        )

    monkeypatch.setattr(
        storage,
        "update_turn",
        fail_update_turn,
    )

    with pytest.raises(
        RuntimeError,
        match="simulated metadata failure",
    ):
        representations.materialize_turns(
            storage=storage,
            source_id="source_001",
            source=source_audio,
            representation_name="speaker",
            kind="center",
            purposes=[
                "speaker_embedding",
            ],
        )

    assert speaker_path.read_bytes() == (
        b"old-speaker-audio"
    )

    backup = speaker_path.with_name(
        f".{speaker_path.name}.backup"
    )

    assert not backup.exists()

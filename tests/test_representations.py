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

    monkeypatch.setattr(
        representations,
        "_boundary_requires_padding",
        lambda source, boundary, *, side, **kwargs: True,
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


def test_manual_boundary_turn_audio_is_not_padded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, source_audio = make_storage(tmp_path)
    calls = []

    prepare_audio_mocks(
        monkeypatch,
        calls,
    )

    def mark_manual(record):
        record["metadata"]["boundary_curation"] = {
            "method": "manual",
        }
        return record

    storage.update_turn(
        "turn_000002",
        mark_manual,
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

    # A human-curated boundary is authoritative.
    # Signal continuity must not reintroduce padding.
    assert calls[1]["start"] == pytest.approx(
        2.1
    )
    assert calls[1]["end"] == pytest.approx(
        3.0
    )

    turn = storage.get_turn(
        "turn_000002"
    )

    assert turn is not None

    metadata = turn["representations"][
        "speech"
    ]["metadata"]

    assert metadata[
        "padding_before"
    ] == pytest.approx(0.0)

    assert metadata[
        "padding_after"
    ] == pytest.approx(0.0)

def make_region_storage(
    tmp_path: Path,
) -> tuple[DatasetStorage, Path]:
    storage = DatasetStorage(
        tmp_path / "region-dataset"
    )

    source_audio = (
        tmp_path / "region-source.wav"
    )
    source_audio.touch()

    storage.regions.append(
        {
            "schema_version": 2,
            "record_type": "candidate_region",
            "id": "region_000001",
            "source_id": "source_001",
            "source_start": 1.0,
            "source_end": 2.0,
            "detector": "test",
            "detector_label": "speaker_0",
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

    return storage, source_audio


def test_materialize_regions_rebuilds_stale_region_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, source_audio = (
        make_region_storage(tmp_path)
    )

    calls = []

    prepare_audio_mocks(
        monkeypatch,
        calls,
    )

    first = representations.materialize_regions(
        storage=storage,
        source_id="source_001",
        source=source_audio,
        representation_name="center",
        kind="center",
        purposes=["asr"],
    )

    assert first.created == 1
    assert first.skipped == 0

    calls.clear()

    def trim_region(record):
        record["source_end"] = 1.8
        return record

    storage.update_region(
        "region_000001",
        trim_region,
    )

    second = (
        representations.materialize_regions(
            storage=storage,
            source_id="source_001",
            source=source_audio,
            representation_name="center",
            kind="center",
            purposes=["asr"],
        )
    )

    assert second.created == 1
    assert second.skipped == 0

    assert len(calls) == 1

    assert calls[0][
        "start"
    ] == pytest.approx(1.0)

    assert calls[0][
        "end"
    ] == pytest.approx(1.8)

    region = storage.get_region(
        "region_000001"
    )

    metadata = region["representations"][
        "center"
    ]["metadata"]

    assert metadata[
        "canonical_start"
    ] == pytest.approx(1.0)

    assert metadata[
        "canonical_end"
    ] == pytest.approx(1.8)


def test_materialize_regions_accepts_legacy_geometry_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, source_audio = (
        make_region_storage(tmp_path)
    )

    calls = []

    prepare_audio_mocks(
        monkeypatch,
        calls,
    )

    relative_path = (
        "regions/region_000001/"
        "center.wav"
    )

    destination = (
        storage.root / relative_path
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    destination.touch()

    def add_legacy(record):
        record["representations"][
            "center"
        ] = {
            "path": relative_path,
            "kind": "center",
            "sample_rate": 48000,
            "channels": 1,
            "purposes": ["asr"],
            "metadata": {
                "source": str(
                    source_audio.resolve()
                ),
            },
        }

        return record

    storage.update_region(
        "region_000001",
        add_legacy,
    )

    result = representations.materialize_regions(
        storage=storage,
        source_id="source_001",
        source=source_audio,
        representation_name="center",
        kind="center",
        purposes=["asr"],
    )

    assert result.created == 0
    assert result.skipped == 1
    assert calls == []


def test_rebuilding_region_representation_invalidates_dependent_transcripts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, source_audio = (
        make_region_storage(tmp_path)
    )

    calls = []

    prepare_audio_mocks(
        monkeypatch,
        calls,
    )

    representations.materialize_regions(
        storage=storage,
        source_id="source_001",
        source=source_audio,
        representation_name="center",
        kind="center",
        purposes=["asr"],
    )

    def add_transcripts(record):
        record["transcripts"]["whisper"] = {
            "text": "old transcript",
            "language": "en",
            "model": "large-v3",
            "representation": "center",
            "confidence": None,
            "metadata": {
                "language_override": "en",
            },
        }

        record["transcripts"]["other"] = {
            "text": "keep me",
            "language": "en",
            "model": "test",
            "representation": "other",
            "confidence": None,
            "metadata": {},
        }

        return record

    storage.update_region(
        "region_000001",
        add_transcripts,
    )

    def trim_region(record):
        record["source_end"] = 1.8
        return record

    storage.update_region(
        "region_000001",
        trim_region,
    )

    representations.materialize_regions(
        storage=storage,
        source_id="source_001",
        source=source_audio,
        representation_name="center",
        kind="center",
        purposes=["asr"],
    )

    region = storage.get_region(
        "region_000001"
    )

    assert region is not None

    assert (
        "whisper"
        not in region["transcripts"]
    )

    assert (
        "other"
        in region["transcripts"]
    )


def test_boundary_padding_detects_clean_and_clipped_edges(
    tmp_path: Path,
) -> None:
    import numpy as np
    import soundfile as sf

    sample_rate = 16000
    duration = 2.0

    samples = np.zeros(
        int(sample_rate * duration),
        dtype=np.float32,
    )

    def fill_signal(
        start: float,
        end: float,
    ) -> None:
        start_frame = int(
            start * sample_rate
        )
        end_frame = int(
            end * sample_rate
        )

        times = (
            np.arange(
                end_frame - start_frame,
                dtype=np.float32,
            )
            / sample_rate
        )

        samples[
            start_frame:end_frame
        ] = (
            0.2
            * np.sin(
                2.0
                * np.pi
                * 440.0
                * times
            )
        )

    # 0.40–0.70:
    # clean onset at 0.40, clean ending at 0.70.
    fill_signal(
        0.40,
        0.70,
    )

    # 1.00–1.40:
    # boundaries 1.20 and 1.30 are both inside
    # continuous signal.
    fill_signal(
        1.00,
        1.40,
    )

    source = tmp_path / "boundary-source.wav"

    sf.write(
        source,
        samples,
        sample_rate,
        subtype="FLOAT",
    )

    # Silence immediately before the onset:
    # no reason to extend the turn backwards.
    assert not representations._boundary_requires_padding(
        source,
        0.40,
        side="start",
    )

    # Signal exists on both sides of the boundary:
    # this looks like a turn beginning mid-sound.
    assert representations._boundary_requires_padding(
        source,
        1.20,
        side="start",
    )

    # Silence immediately after the clean ending:
    # no reason to extend the turn forwards.
    assert not representations._boundary_requires_padding(
        source,
        0.70,
        side="end",
    )

    # Signal exists on both sides:
    # this looks like a turn ending mid-sound.
    assert representations._boundary_requires_padding(
        source,
        1.30,
        side="end",
    )


def test_review_turn_padding_is_selected_per_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, source_audio = make_storage(tmp_path)
    calls = []

    prepare_audio_mocks(
        monkeypatch,
        calls,
    )

    decisions = []

    def boundary_requires_padding(
        source,
        boundary,
        *,
        side,
        **kwargs,
    ):
        decisions.append(
            (
                float(boundary),
                side,
            )
        )

        # The middle turn starts cleanly but ends
        # inside continuous signal.
        if (
            boundary == pytest.approx(2.1)
            and side == "start"
        ):
            return False

        if (
            boundary == pytest.approx(3.0)
            and side == "end"
        ):
            return True

        return False

    monkeypatch.setattr(
        representations,
        "_boundary_requires_padding",
        boundary_requires_padding,
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

    # Turn 2: clean start, clipped-looking end.
    assert calls[1]["start"] == pytest.approx(
        2.1
    )
    assert calls[1]["end"] == pytest.approx(
        3.25
    )

    turn = storage.get_turn(
        "turn_000002"
    )

    assert turn is not None

    metadata = turn["representations"][
        "speech"
    ]["metadata"]

    assert metadata[
        "padding_before"
    ] == pytest.approx(0.0)

    assert metadata[
        "padding_after"
    ] == pytest.approx(0.25)

    assert (
        2.1,
        "start",
    ) in decisions

    assert (
        3.0,
        "end",
    ) in decisions


def test_boundary_padding_detects_quiet_continuous_signal(
    tmp_path: Path,
) -> None:
    import numpy as np
    import soundfile as sf

    sample_rate = 16000
    samples = np.zeros(
        sample_rate,
        dtype=np.float32,
    )

    start_frame = int(
        0.30 * sample_rate
    )
    end_frame = int(
        0.70 * sample_rate
    )

    times = (
        np.arange(
            end_frame - start_frame,
            dtype=np.float32,
        )
        / sample_rate
    )

    # RMS is roughly 0.0042: deliberately below the
    # old absolute 0.01 threshold, but continuous
    # across the test boundary.
    samples[
        start_frame:end_frame
    ] = (
        0.006
        * np.sin(
            2.0
            * np.pi
            * 440.0
            * times
        )
    )

    source = tmp_path / "quiet-continuous.wav"

    sf.write(
        source,
        samples,
        sample_rate,
        subtype="FLOAT",
    )

    assert representations._boundary_requires_padding(
        source,
        0.50,
        side="start",
    )

    assert representations._boundary_requires_padding(
        source,
        0.50,
        side="end",
    )


def test_boundary_padding_rejects_strong_level_transition(
    tmp_path: Path,
) -> None:
    import numpy as np
    import soundfile as sf

    sample_rate = 16000
    samples = np.zeros(
        sample_rate,
        dtype=np.float32,
    )

    boundary_frame = int(
        0.50 * sample_rate
    )

    left_start = int(
        0.30 * sample_rate
    )

    right_end = int(
        0.70 * sample_rate
    )

    left_times = (
        np.arange(
            boundary_frame - left_start,
            dtype=np.float32,
        )
        / sample_rate
    )

    right_times = (
        np.arange(
            right_end - boundary_frame,
            dtype=np.float32,
        )
        / sample_rate
    )

    # Both sides contain real signal, but the level
    # changes by much more than our 0.25 continuity
    # ratio. Treat this as an onset/offset rather than
    # a clipped continuous sound.
    samples[
        left_start:boundary_frame
    ] = (
        0.010
        * np.sin(
            2.0
            * np.pi
            * 440.0
            * left_times
        )
    )

    samples[
        boundary_frame:right_end
    ] = (
        0.080
        * np.sin(
            2.0
            * np.pi
            * 440.0
            * right_times
        )
    )

    source = tmp_path / "level-transition.wav"

    sf.write(
        source,
        samples,
        sample_rate,
        subtype="FLOAT",
    )

    assert not representations._boundary_requires_padding(
        source,
        0.50,
        side="start",
    )

    assert not representations._boundary_requires_padding(
        source,
        0.50,
        side="end",
    )

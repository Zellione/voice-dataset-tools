from pathlib import Path

import pytest

from voice_dataset.schema import (
    AudioRepresentation,
    SourceRecord,
)
from voice_dataset.storage import DatasetStorage


def test_add_and_get_source(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)

    source = SourceRecord(
        id="arcane-s01e09-test",
        media_path="/media/arcane.mkv",
        representations={
            "center": AudioRepresentation(
                path="/scratch/arcane/center.wav",
                kind="center",
                sample_rate=48000,
                channels=1,
                duration=60.0,
                media_start=600.0,
                purposes=[
                    "speaker_embedding",
                    "boundary_analysis",
                    "context",
                ],
                stream_index=1,
                channel_mode="center",
            ),
            "speech": AudioRepresentation(
                path=(
                    "/scratch/arcane/"
                    "speech_estimate.wav"
                ),
                kind="separated_speech",
                processor="bandit",
                sample_rate=48000,
                channels=1,
                purposes=[
                    "asr",
                    "review",
                    "tts_candidate",
                ],
            ),
        },
        metadata={
            "episode": "S01E09",
        },
    )

    storage.add_source(source)

    result = storage.get_source(
        "arcane-s01e09-test"
    )

    assert result is not None
    assert result["record_type"] == "source"
    assert result["schema_version"] == 1

    assert result["id"] == "arcane-s01e09-test"
    assert result["media_path"] == "/media/arcane.mkv"

    assert (
        result["representations"]["center"]["kind"]
        == "center"
    )

    assert (
        result["representations"]["center"]["path"]
        == "/scratch/arcane/center.wav"
    )

    assert (
        result["representations"]["center"]["stream_index"]
        == 1
    )
    assert (
        result["representations"]["center"]["channel_mode"]
        == "center"
    )

    assert (
        result["representations"]["center"][
            "duration"
        ]
        == 60.0
    )

    assert (
        result["representations"]["center"][
            "media_start"
        ]
        == 600.0
    )

    assert (
        "context"
        in result["representations"]["center"][
            "purposes"
        ]
    )

    assert (
        result["representations"]["speech"][
            "processor"
        ]
        == "bandit"
    )

    assert result["metadata"] == {
        "episode": "S01E09",
    }


def test_get_unknown_source(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)

    assert (
        storage.get_source("does-not-exist")
        is None
    )


def test_duplicate_source_is_rejected(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)

    source = SourceRecord(
        id="source-test",
    )

    storage.add_source(source)

    before = storage.sources.path.read_bytes()

    with pytest.raises(
        ValueError,
        match="Record already exists",
    ):
        storage.add_source(source)

    assert storage.sources.path.read_bytes() == before


def test_update_source_preserves_other_fields(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)

    source = SourceRecord(
        id="source-1",
        media_path="/media/episode.mkv",
        representations={
            "center": AudioRepresentation(
                path="/scratch/center.wav",
                kind="center",
                purposes=["context"],
            ),
        },
        metadata={
            "episode": "S01E09",
        },
    )

    storage.add_source(source)

    def update(record):
        representation = record[
            "representations"
        ]["center"]

        representation["media_start"] = 600.0
        representation["stream_index"] = 1
        representation["channel_mode"] = "center"

        return record

    updated = storage.update_source(
        "source-1",
        update,
    )

    assert updated["media_path"] == (
        "/media/episode.mkv"
    )
    assert updated["metadata"] == {
        "episode": "S01E09",
    }

    representation = updated[
        "representations"
    ]["center"]

    assert representation["media_start"] == 600.0
    assert representation["stream_index"] == 1
    assert representation["channel_mode"] == "center"

    stored = storage.get_source("source-1")

    assert stored == updated

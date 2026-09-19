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
                purposes=[
                    "speaker_embedding",
                    "boundary_analysis",
                    "context",
                ],
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

from __future__ import annotations

from pathlib import Path

import pytest

from voice_dataset.region_asr import (
    WHISPER_MODEL,
    transcribe_regions,
)
from voice_dataset.schema import (
    AudioRepresentation,
    CandidateRegion,
    SourceRecord,
)
from voice_dataset.storage import DatasetStorage


def make_storage(
    tmp_path: Path,
) -> DatasetStorage:
    storage = DatasetStorage(tmp_path)

    audio = tmp_path / "source.wav"
    audio.write_bytes(b"placeholder")

    storage.add_source(
        SourceRecord(
            id="source_001",
            representations={
                "center": AudioRepresentation(
                    path=str(audio),
                    kind="center",
                    purposes=["asr"],
                ),
            },
        )
    )

    storage.add_region(
        CandidateRegion(
            id="region_000001",
            source_id="source_001",
            source_start=1.0,
            source_end=2.0,
            detector="test",
        )
    )

    return storage


def test_transcribe_regions_skips_matching_transcript(
    tmp_path,
    monkeypatch,
):
    storage = make_storage(tmp_path)

    def add_existing(record):
        record["representations"][
            "center"
        ] = {
            "path": (
                "regions/region_000001/"
                "center.wav"
            ),
            "kind": "center",
            "purposes": ["asr"],
        }

        record["transcripts"][
            "whisper"
        ] = {
            "text": "hello",
            "language": "en",
            "model": WHISPER_MODEL,
            "representation": "center",
            "confidence": None,
            "metadata": {
                "language_override": "en",
            },
        }

        return record

    storage.update_region(
        "region_000001",
        add_existing,
    )

    path = (
        storage.root
        / "regions"
        / "region_000001"
        / "center.wav"
    )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_bytes(b"placeholder")

    monkeypatch.setattr(
        "voice_dataset.region_asr."
        "materialize_regions",
        lambda **kwargs: None,
    )

    result = transcribe_regions(
        storage,
        "source_001",
        representation_name="center",
        transcript_name="whisper",
        language="en",
    )

    assert result.transcribed == 0
    assert result.skipped == 1


@pytest.mark.parametrize(
    (
        "model",
        "representation",
        "language_override",
    ),
    [
        (
            "old-whisper",
            "center",
            "en",
        ),
        (
            WHISPER_MODEL,
            "old-representation",
            "en",
        ),
        (
            WHISPER_MODEL,
            "center",
            None,
        ),
    ],
)
def test_transcribe_regions_rejects_stale_transcript_provenance(
    tmp_path,
    monkeypatch,
    model,
    representation,
    language_override,
):
    storage = make_storage(tmp_path)

    def add_existing(record):
        record["representations"][
            "center"
        ] = {
            "path": (
                "regions/region_000001/"
                "center.wav"
            ),
            "kind": "center",
            "purposes": ["asr"],
        }

        record["transcripts"][
            "whisper"
        ] = {
            "text": "hello",
            "language": "en",
            "model": model,
            "representation": representation,
            "confidence": None,
            "metadata": {
                "language_override": (
                    language_override
                ),
            },
        }

        return record

    storage.update_region(
        "region_000001",
        add_existing,
    )

    path = (
        storage.root
        / "regions"
        / "region_000001"
        / "center.wav"
    )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_bytes(b"placeholder")

    monkeypatch.setattr(
        "voice_dataset.region_asr."
        "materialize_regions",
        lambda **kwargs: None,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Region transcript already exists "
            "with different provenance"
        ),
    ):
        transcribe_regions(
            storage,
            "source_001",
            representation_name="center",
            transcript_name="whisper",
            language="en",
        )

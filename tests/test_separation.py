from __future__ import annotations

import wave
from pathlib import Path

import pytest

from voice_dataset.schema import (
    AudioRepresentation,
    SourceRecord,
)
from voice_dataset.separation import (
    BANDIT_CHECKPOINT_SHA256,
    BANDIT_PROCESSOR,
    BANDIT_REVISION,
    SEPARATION_PROVENANCE_VERSION,
    separate_source,
)
from voice_dataset.storage import DatasetStorage


def _wav(path: Path) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(
            b"\x00\x00" * 160
        )


def _storage(
    tmp_path: Path,
    *,
    provenance=None,
) -> DatasetStorage:
    storage = DatasetStorage(tmp_path)

    center = tmp_path / "center.wav"
    speech = (
        tmp_path
        / "audio"
        / "source_001"
        / "speech"
        / "speech.wav"
    )

    _wav(center)
    _wav(speech)

    metadata = {
        "checkpoint_sha256": (
            BANDIT_CHECKPOINT_SHA256
        ),
        "parameters": {
            "chunk": 30.0,
            "overlap": 2.0,
        },
    }

    if provenance is not None:
        metadata["provenance"] = provenance

    storage.add_source(
        SourceRecord(
            id="source_001",
            representations={
                "center": AudioRepresentation(
                    path=str(center),
                    kind="center",
                    purposes=[
                        "boundary_analysis",
                    ],
                ),
                "speech": AudioRepresentation(
                    path=(
                        "audio/source_001/"
                        "speech/speech.wav"
                    ),
                    kind="separated_speech",
                    processor=BANDIT_PROCESSOR,
                    processor_version=BANDIT_REVISION,
                    purposes=[
                        "review",
                        "tts_candidate",
                    ],
                    metadata=metadata,
                ),
            },
        )
    )

    return storage


def test_separate_source_accepts_legacy_provenance(
    tmp_path,
):
    storage = _storage(
        tmp_path,
        provenance=None,
    )

    result = separate_source(
        storage,
        "source_001",
        input_representation="center",
    )

    assert result.representation_name == "speech"


def test_separate_source_accepts_current_parent_provenance(
    tmp_path,
):
    storage = _storage(
        tmp_path,
        provenance={
            "version": (
                SEPARATION_PROVENANCE_VERSION
            ),
            "input_representation": "center",
        },
    )

    result = separate_source(
        storage,
        "source_001",
        input_representation="center",
    )

    assert result.representation_name == "speech"


def test_separate_source_rejects_different_parent_provenance(
    tmp_path,
):
    storage = _storage(
        tmp_path,
        provenance={
            "version": (
                SEPARATION_PROVENANCE_VERSION
            ),
            "input_representation": "other",
        },
    )

    with pytest.raises(
        ValueError,
        match="different provenance",
    ):
        separate_source(
            storage,
            "source_001",
            input_representation="center",
        )


def test_separate_source_rejects_stale_provenance_version(
    tmp_path,
):
    storage = _storage(
        tmp_path,
        provenance={
            "version": (
                SEPARATION_PROVENANCE_VERSION
                - 1
            ),
            "input_representation": "center",
        },
    )

    with pytest.raises(
        ValueError,
        match="different provenance",
    ):
        separate_source(
            storage,
            "source_001",
            input_representation="center",
        )

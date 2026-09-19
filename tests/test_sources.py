from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice_dataset.schema import (
    AudioRepresentation,
    SourceRecord,
)
from voice_dataset.sources import (
    probe_audio_representation,
    set_representation_provenance,
)
from voice_dataset.storage import DatasetStorage


def make_storage(
    tmp_path: Path,
) -> DatasetStorage:
    storage = DatasetStorage(tmp_path)

    audio_path = tmp_path / "center.wav"

    sample_rate = 48000
    duration = 2.0

    audio = np.zeros(
        round(sample_rate * duration),
        dtype=np.float32,
    )

    sf.write(
        audio_path,
        audio,
        sample_rate,
        subtype="PCM_16",
    )

    storage.add_source(
        SourceRecord(
            id="source-1",
            media_path="/media/episode.mkv",
            representations={
                "center": AudioRepresentation(
                    path=str(audio_path),
                    kind="center",
                    purposes=["context"],
                ),
            },
        )
    )

    return storage


def test_probe_audio_representation(
    tmp_path: Path,
):
    storage = make_storage(tmp_path)

    source = storage.get_source("source-1")

    assert source is not None

    path = Path(
        source["representations"]["center"]["path"]
    )

    result = probe_audio_representation(path)

    assert result == {
        "sample_rate": 48000,
        "channels": 1,
        "duration": 2.0,
    }


def test_set_representation_provenance(
    tmp_path: Path,
):
    storage = make_storage(tmp_path)

    result = set_representation_provenance(
        storage,
        "source-1",
        "center",
        media_start=600.0,
        stream_index=1,
        channel_mode="center",
    )

    representation = result[
        "representations"
    ]["center"]

    assert representation["media_start"] == 600.0
    assert representation["stream_index"] == 1
    assert representation["channel_mode"] == "center"

    assert representation["sample_rate"] == 48000
    assert representation["channels"] == 1
    assert representation["duration"] == 2.0

    stored = storage.get_source("source-1")

    assert stored == result


def test_set_representation_provenance_rejects_unknown_representation(
    tmp_path: Path,
):
    storage = make_storage(tmp_path)

    with pytest.raises(
        KeyError,
        match="Source representation does not exist",
    ):
        set_representation_provenance(
            storage,
            "source-1",
            "missing",
            media_start=600.0,
            stream_index=1,
            channel_mode="center",
        )


@pytest.mark.parametrize(
    (
        "media_start",
        "stream_index",
        "channel_mode",
        "message",
    ),
    [
        (
            -1.0,
            1,
            "center",
            "media_start must not be negative",
        ),
        (
            600.0,
            -1,
            "center",
            "stream_index must not be negative",
        ),
        (
            600.0,
            1,
            "invalid",
            "Unsupported channel_mode",
        ),
    ],
)
def test_set_representation_provenance_rejects_invalid_values(
    tmp_path: Path,
    media_start: float,
    stream_index: int,
    channel_mode: str,
    message: str,
):
    storage = make_storage(tmp_path)

    with pytest.raises(
        ValueError,
        match=message,
    ):
        set_representation_provenance(
            storage,
            "source-1",
            "center",
            media_start=media_start,
            stream_index=stream_index,
            channel_mode=channel_mode,
        )

from pathlib import Path

import pytest

from voice_dataset.schema import (
    AudioRepresentation,
    SourceRecord,
)
from voice_dataset.sources import (
    set_representation_provenance,
)
from voice_dataset.storage import DatasetStorage


def make_storage(
    tmp_path: Path,
) -> DatasetStorage:
    storage = DatasetStorage(tmp_path)

    storage.add_source(
        SourceRecord(
            id="source-1",
            media_path="/media/episode.mkv",
            representations={
                "center": AudioRepresentation(
                    path="/scratch/center.wav",
                    kind="center",
                    purposes=["context"],
                ),
            },
        )
    )

    return storage


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

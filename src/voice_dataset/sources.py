from __future__ import annotations

from pathlib import Path
from typing import Any

import soundfile as sf

from .storage import DatasetStorage


SUPPORTED_CHANNEL_MODES = {
    "mono",
    "center",
}


def probe_audio_representation(
    path: Path,
) -> dict[str, Any]:
    path = path.resolve()

    if not path.is_file():
        raise ValueError(
            "Representation audio does not exist: "
            f"{path}"
        )

    with sf.SoundFile(path) as audio:
        sample_rate = int(audio.samplerate)
        channels = int(audio.channels)
        frames = int(audio.frames)

    if sample_rate <= 0:
        raise ValueError(
            "Representation has invalid sample rate: "
            f"{path}"
        )

    if channels <= 0:
        raise ValueError(
            "Representation has invalid channel count: "
            f"{path}"
        )

    if frames < 0:
        raise ValueError(
            "Representation has invalid frame count: "
            f"{path}"
        )

    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "duration": frames / sample_rate,
    }


def resolve_source_representation(
    storage: DatasetStorage,
    source_id: str,
    representation_name: str,
) -> tuple[dict[str, Any], Path]:
    source = storage.get_source(source_id)

    if source is None:
        raise KeyError(
            f"Source does not exist: {source_id}"
        )

    representations = source.get(
        "representations",
        {},
    )

    if not isinstance(representations, dict):
        raise ValueError(
            "Source has invalid representations: "
            f"{source_id}"
        )

    representation = representations.get(
        representation_name
    )

    if not isinstance(representation, dict):
        raise KeyError(
            "Source representation does not exist: "
            f"{representation_name}"
        )

    path_value = representation.get("path")

    if not isinstance(path_value, str) or not path_value:
        raise ValueError(
            "Source representation has no valid path: "
            f"{representation_name}"
        )

    path = Path(path_value)

    if not path.is_absolute():
        path = storage.root / path

    path = path.resolve()

    if not path.is_file():
        raise ValueError(
            "Source representation audio does not exist: "
            f"{path}"
        )

    return representation, path


def set_representation_provenance(
    storage: DatasetStorage,
    source_id: str,
    representation_name: str,
    *,
    media_start: float,
    stream_index: int,
    channel_mode: str,
) -> dict[str, Any]:
    media_start = float(media_start)

    if media_start < 0:
        raise ValueError(
            "media_start must not be negative"
        )

    if stream_index < 0:
        raise ValueError(
            "stream_index must not be negative"
        )

    if channel_mode not in SUPPORTED_CHANNEL_MODES:
        raise ValueError(
            "Unsupported channel_mode: "
            f"{channel_mode}"
        )

    _, representation_path = (
        resolve_source_representation(
            storage,
            source_id,
            representation_name,
        )
    )

    audio_info = probe_audio_representation(
        representation_path
    )

    def update(
        record: dict[str, Any],
    ) -> dict[str, Any]:
        current_representations = record.get(
            "representations",
            {},
        )

        current = current_representations[
            representation_name
        ]

        current["media_start"] = media_start
        current["stream_index"] = stream_index
        current["channel_mode"] = channel_mode

        current["sample_rate"] = audio_info[
            "sample_rate"
        ]
        current["channels"] = audio_info[
            "channels"
        ]
        current["duration"] = audio_info[
            "duration"
        ]

        return record

    return storage.update_source(
        source_id,
        update,
    )

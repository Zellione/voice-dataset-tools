from __future__ import annotations

from typing import Any

from .storage import DatasetStorage


SUPPORTED_CHANNEL_MODES = {
    "mono",
    "center",
}


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

    def update(
        source: dict[str, Any],
    ) -> dict[str, Any]:
        representations = source.get(
            "representations",
            {},
        )

        if representation_name not in representations:
            raise KeyError(
                "Source representation does not exist: "
                f"{representation_name}"
            )

        representation = representations[
            representation_name
        ]

        representation["media_start"] = media_start
        representation["stream_index"] = stream_index
        representation["channel_mode"] = channel_mode

        return source

    return storage.update_source(
        source_id,
        update,
    )

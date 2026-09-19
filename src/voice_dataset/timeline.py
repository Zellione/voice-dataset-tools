from __future__ import annotations

from typing import Any


def representation_media_start(
    representation: dict[str, Any],
) -> float:
    media_start = representation.get(
        "media_start"
    )

    if media_start is None:
        raise ValueError(
            "Representation has no known "
            "media timeline mapping"
        )

    media_start = float(media_start)

    if media_start < 0:
        raise ValueError(
            "Representation media_start "
            "must not be negative"
        )

    return media_start


def representation_time_to_media_time(
    representation: dict[str, Any],
    time: float,
) -> float:
    time = float(time)

    if time < 0:
        raise ValueError(
            "Representation time must not "
            "be negative"
        )

    return (
        representation_media_start(
            representation
        )
        + time
    )


def media_time_to_representation_time(
    representation: dict[str, Any],
    time: float,
) -> float:
    time = float(time)

    if time < 0:
        raise ValueError(
            "Media time must not be negative"
        )

    representation_time = (
        time
        - representation_media_start(
            representation
        )
    )

    if representation_time < 0:
        raise ValueError(
            "Media time is before the start "
            "of the representation"
        )

    return representation_time

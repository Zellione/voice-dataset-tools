from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .media import (
    extract_audio_region,
    probe_audio_streams,
)
from .schema import AudioRepresentation
from .storage import DatasetStorage


@dataclass
class MaterializeRegionsResult:
    created: int
    skipped: int


def probe_representation_source(
    source: Path,
) -> tuple[int, int]:
    streams = probe_audio_streams(
        source
    )

    if len(streams) != 1:
        raise ValueError(
            "Representation source must contain "
            "exactly one audio stream; "
            f"found {len(streams)}: {source}"
        )

    stream = streams[0]

    if stream.sample_rate is None:
        raise ValueError(
            "Representation source has no "
            f"sample rate: {source}"
        )

    if stream.channels is None:
        raise ValueError(
            "Representation source has no "
            f"channel count: {source}"
        )

    return (
        stream.sample_rate,
        stream.channels,
    )


def materialize_region_audio(
    storage: DatasetStorage,
    region_id: str,
    source: Path,
    representation_name: str,
    kind: str,
    purposes: list[str],
    sample_rate: int | None = None,
    channels: int | None = None,
) -> AudioRepresentation:
    source = source.resolve()

    if not source.is_file():
        raise ValueError(
            f"Audio source does not exist: {source}"
        )

    if (
        sample_rate is None
        or channels is None
    ):
        (
            detected_sample_rate,
            detected_channels,
        ) = probe_representation_source(
            source
        )

        if sample_rate is None:
            sample_rate = (
                detected_sample_rate
            )

        if channels is None:
            channels = detected_channels

    region = storage.get_region(
        region_id
    )

    if region is None:
        raise KeyError(
            f"Region does not exist: {region_id}"
        )

    representations = region.get(
        "representations",
        {},
    )

    if not isinstance(
        representations,
        dict,
    ):
        raise ValueError(
            f"Invalid representations "
            f"for {region_id}"
        )

    existing = representations.get(
        representation_name
    )

    if existing is not None:
        raise ValueError(
            f"Representation already exists: "
            f"{region_id}/{representation_name}"
        )

    start = region.get(
        "source_start"
    )

    end = region.get(
        "source_end"
    )

    if not isinstance(
        start,
        (int, float),
    ):
        raise ValueError(
            f"Invalid source_start for {region_id}"
        )

    if not isinstance(
        end,
        (int, float),
    ):
        raise ValueError(
            f"Invalid source_end for {region_id}"
        )

    relative_path = (
        Path("regions")
        / region_id
        / f"{representation_name}.wav"
    )

    destination = (
        storage.root
        / relative_path
    )

    temporary = destination.with_name(
        f".{destination.name}.tmp.wav"
    )

    if destination.exists():
        raise ValueError(
            f"Representation file already exists: "
            f"{destination}"
        )

    temporary.unlink(
        missing_ok=True
    )

    try:
        extract_audio_region(
            source=source,
            destination=temporary,
            start=float(start),
            end=float(end),
        )

        os.replace(
            temporary,
            destination,
        )

        representation = AudioRepresentation(
            path=relative_path.as_posix(),
            kind=kind,
            sample_rate=sample_rate,
            channels=channels,
            purposes=list(purposes),
            metadata={
                "source": str(source),
            },
        )

        def update(
            record: dict[str, Any],
        ) -> dict[str, Any]:
            current = record.get(
                "representations"
            )

            if not isinstance(
                current,
                dict,
            ):
                raise ValueError(
                    f"Invalid representations "
                    f"for {region_id}"
                )

            if representation_name in current:
                raise ValueError(
                    f"Representation already exists: "
                    f"{region_id}/"
                    f"{representation_name}"
                )

            current[
                representation_name
            ] = representation.to_dict()

            return record

        storage.update_region(
            region_id,
            update,
        )

    except Exception:
        temporary.unlink(
            missing_ok=True
        )

        destination.unlink(
            missing_ok=True
        )

        raise

    return representation


def materialize_regions(
    storage: DatasetStorage,
    source_id: str,
    source: Path,
    representation_name: str,
    kind: str,
    purposes: list[str],
) -> MaterializeRegionsResult:
    source = source.resolve()

    if not source.is_file():
        raise ValueError(
            f"Audio source does not exist: {source}"
        )

    (
        sample_rate,
        channels,
    ) = probe_representation_source(
        source
    )

    regions = storage.regions.load()

    created = 0
    skipped = 0

    for region in regions:
        if region.get("source_id") != source_id:
            continue

        region_id = region.get("id")

        if not isinstance(
            region_id,
            str,
        ) or not region_id:
            raise ValueError(
                "Region has invalid id"
            )

        representations = region.get(
            "representations"
        )

        if not isinstance(
            representations,
            dict,
        ):
            raise ValueError(
                f"Invalid representations "
                f"for {region_id}"
            )

        existing = representations.get(
            representation_name
        )

        expected_relative_path = (
            Path("regions")
            / region_id
            / f"{representation_name}.wav"
        )

        expected_destination = (
            storage.root
            / expected_relative_path
        )

        if existing is not None:
            if not isinstance(
                existing,
                dict,
            ):
                raise ValueError(
                    f"Invalid representation "
                    f"{region_id}/"
                    f"{representation_name}"
                )

            existing_path = existing.get(
                "path"
            )

            if (
                existing_path
                != expected_relative_path.as_posix()
            ):
                raise ValueError(
                    f"Unexpected representation path "
                    f"for {region_id}/"
                    f"{representation_name}: "
                    f"{existing_path}"
                )

            if not expected_destination.is_file():
                raise ValueError(
                    f"Representation metadata exists "
                    f"but file is missing: "
                    f"{expected_destination}"
                )

            skipped += 1
            continue

        if expected_destination.exists():
            raise ValueError(
                f"Representation file exists "
                f"without metadata: "
                f"{expected_destination}"
            )

        materialize_region_audio(
            storage=storage,
            region_id=region_id,
            source=source,
            representation_name=(
                representation_name
            ),
            kind=kind,
            purposes=purposes,
            sample_rate=sample_rate,
            channels=channels,
        )

        created += 1

    return MaterializeRegionsResult(
        created=created,
        skipped=skipped,
    )

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import CandidateRegion
from .storage import DatasetStorage


DETECTOR_OUTPUT_FORMAT = (
    "voice-dataset-detector-output"
)

DETECTOR_OUTPUT_VERSION = 1


@dataclass
class DetectorOutput:
    detector_name: str
    detector_model: str | None

    source_path: Path
    source_sample_rate: int | None
    source_channels: int | None
    source_duration: float | None

    regions: list[dict[str, Any]]


@dataclass
class ValidatedRegion:
    start: float
    end: float
    label: str | None


def _is_number(
    value: Any,
) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def load_detector_output(
    path: Path,
) -> DetectorOutput:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        document = json.load(file)

    if not isinstance(document, dict):
        raise ValueError(
            "Detector output must be a JSON object"
        )

    if document.get("format") != (
        DETECTOR_OUTPUT_FORMAT
    ):
        raise ValueError(
            "Unsupported detector output format: "
            f"{document.get('format')!r}"
        )

    if document.get("version") != (
        DETECTOR_OUTPUT_VERSION
    ):
        raise ValueError(
            "Unsupported detector output version: "
            f"{document.get('version')!r}"
        )

    detector = document.get("detector")

    if not isinstance(detector, dict):
        raise ValueError(
            "Missing detector object"
        )

    detector_name = detector.get("name")

    if not isinstance(
        detector_name,
        str,
    ) or not detector_name:
        raise ValueError(
            "Detector name must be a non-empty string"
        )

    detector_model = detector.get("model")

    if (
        detector_model is not None
        and not isinstance(detector_model, str)
    ):
        raise ValueError(
            "Detector model must be a string or null"
        )

    source = document.get("source")

    if not isinstance(source, dict):
        raise ValueError(
            "Missing source object"
        )

    source_path_value = source.get("path")

    if not isinstance(
        source_path_value,
        str,
    ) or not source_path_value:
        raise ValueError(
            "Source path must be a non-empty string"
        )

    sample_rate = source.get(
        "sample_rate"
    )

    if (
        sample_rate is not None
        and (
            not isinstance(sample_rate, int)
            or isinstance(sample_rate, bool)
            or sample_rate <= 0
        )
    ):
        raise ValueError(
            "Source sample rate must be "
            "a positive integer or null"
        )

    channels = source.get("channels")

    if (
        channels is not None
        and (
            not isinstance(channels, int)
            or isinstance(channels, bool)
            or channels <= 0
        )
    ):
        raise ValueError(
            "Source channels must be "
            "a positive integer or null"
        )

    duration = source.get("duration")

    if (
        duration is not None
        and (
            not _is_number(duration)
            or duration <= 0
        )
    ):
        raise ValueError(
            "Source duration must be "
            "a positive finite number or null"
        )

    regions = document.get("regions")

    if not isinstance(regions, list):
        raise ValueError(
            "Regions must be a list"
        )

    return DetectorOutput(
        detector_name=detector_name,
        detector_model=detector_model,
        source_path=Path(source_path_value),
        source_sample_rate=sample_rate,
        source_channels=channels,
        source_duration=(
            float(duration)
            if duration is not None
            else None
        ),
        regions=regions,
    )


def _validate_regions(
    detector_output: DetectorOutput,
) -> list[ValidatedRegion]:
    validated = []

    for index, item in enumerate(
        detector_output.regions,
        start=1,
    ):
        if not isinstance(item, dict):
            raise ValueError(
                "Detector region "
                f"{index} must be an object"
            )

        start = item.get("start")
        end = item.get("end")
        label = item.get("label")

        if not _is_number(start):
            raise ValueError(
                "Detector region "
                f"{index} has invalid start"
            )

        if not _is_number(end):
            raise ValueError(
                "Detector region "
                f"{index} has invalid end"
            )

        start = float(start)
        end = float(end)

        if start < 0:
            raise ValueError(
                "Detector region "
                f"{index} starts before zero"
            )

        if end <= start:
            raise ValueError(
                "Detector region "
                f"{index} has end <= start"
            )

        if (
            detector_output.source_duration
            is not None
            and end
            > detector_output.source_duration
        ):
            raise ValueError(
                "Detector region "
                f"{index} exceeds source duration"
            )

        if (
            label is not None
            and not isinstance(label, str)
        ):
            raise ValueError(
                "Detector region "
                f"{index} has invalid label"
            )

        validated.append(
            ValidatedRegion(
                start=start,
                end=end,
                label=label,
            )
        )

    return validated


def import_detector_regions(
    storage: DatasetStorage,
    detector_output: DetectorOutput,
    source_id: str,
) -> list[CandidateRegion]:
    validated = _validate_regions(
        detector_output
    )

    imported = []

    for index, item in enumerate(
        validated,
        start=1,
    ):
        region = CandidateRegion(
            id=storage.next_region_id(),
            source_id=source_id,
            source_start=item.start,
            source_end=item.end,
            detector=(
                detector_output.detector_name
            ),
            detector_label=item.label,
            metadata={
                "detector_model": (
                    detector_output.detector_model
                ),
                "detector_region_index": index,
            },
        )

        storage.add_region(region)
        imported.append(region)

    return imported

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
    detector_revision: str | None
    detector_parameters: dict[str, Any]

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

    detector_revision = detector.get(
        "revision"
    )

    if (
        detector_revision is not None
        and not isinstance(
            detector_revision,
            str,
        )
    ):
        raise ValueError(
            "Detector revision must be "
            "a string or null"
        )

    detector_parameters = detector.get(
        "parameters",
        {},
    )

    if not isinstance(
        detector_parameters,
        dict,
    ):
        raise ValueError(
            "Detector parameters must be "
            "an object"
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
        detector_revision=detector_revision,
        detector_parameters=(
            detector_parameters
        ),
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


@dataclass
class DetectorImportResult:
    imported: list[CandidateRegion]
    skipped: int


def _parameter_identity(
    parameters: dict[str, Any],
) -> str:
    return json.dumps(
        parameters,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _region_identity(
    *,
    source_id: str,
    detector: str,
    detector_model: str | None,
    detector_revision: str | None,
    detector_parameters: dict[str, Any],
    start: float,
    end: float,
    label: str | None,
) -> tuple[
    str,
    str,
    str | None,
    str | None,
    str,
    float,
    float,
    str | None,
]:
    return (
        source_id,
        detector,
        detector_model,
        detector_revision,
        _parameter_identity(
            detector_parameters
        ),
        start,
        end,
        label,
    )


def import_detector_regions(
    storage: DatasetStorage,
    detector_output: DetectorOutput,
    source_id: str,
) -> DetectorImportResult:
    validated = _validate_regions(
        detector_output
    )

    existing_identities = set()

    for record in storage.regions.load():
        metadata = record.get(
            "metadata",
            {},
        )

        if not isinstance(metadata, dict):
            metadata = {}

        source_id_value = record.get(
            "source_id"
        )

        detector = record.get(
            "detector"
        )

        detector_model = metadata.get(
            "detector_model"
        )

        detector_revision = metadata.get(
            "detector_revision"
        )

        detector_parameters = metadata.get(
            "detector_parameters",
            {},
        )

        label = record.get(
            "detector_label"
        )

        start = record.get(
            "source_start"
        )

        end = record.get(
            "source_end"
        )

        if (
            not isinstance(
                source_id_value,
                str,
            )
            or not source_id_value
            or not isinstance(
                detector,
                str,
            )
            or not detector
            or (
                detector_model is not None
                and not isinstance(
                    detector_model,
                    str,
                )
            )
            or (
                detector_revision is not None
                and not isinstance(
                    detector_revision,
                    str,
                )
            )
            or not isinstance(
                detector_parameters,
                dict,
            )
            or (
                label is not None
                and not isinstance(
                    label,
                    str,
                )
            )
            or not _is_number(start)
            or not _is_number(end)
        ):
            continue

        existing_identities.add(
            _region_identity(
                source_id=source_id_value,
                detector=detector,
                detector_model=detector_model,
                detector_revision=(
                    detector_revision
                ),
                detector_parameters=(
                    detector_parameters
                ),
                start=float(start),
                end=float(end),
                label=label,
            )
        )

    imported = []
    skipped = 0

    for index, item in enumerate(
        validated,
        start=1,
    ):
        identity = _region_identity(
            source_id=source_id,
            detector=(
                detector_output.detector_name
            ),
            detector_model=(
                detector_output.detector_model
            ),
            detector_revision=(
                detector_output.detector_revision
            ),
            detector_parameters=(
                detector_output.detector_parameters
            ),
            start=item.start,
            end=item.end,
            label=item.label,
        )

        if identity in existing_identities:
            skipped += 1
            continue

        metadata = {
            "detector_model": (
                detector_output.detector_model
            ),
            "detector_region_index": index,
        }

        if (
            detector_output.detector_revision
            is not None
        ):
            metadata["detector_revision"] = (
                detector_output.detector_revision
            )

        if detector_output.detector_parameters:
            metadata["detector_parameters"] = (
                detector_output.detector_parameters
            )

        region = CandidateRegion(
            id=storage.next_region_id(),
            source_id=source_id,
            source_start=item.start,
            source_end=item.end,
            detector=(
                detector_output.detector_name
            ),
            detector_label=item.label,
            metadata=metadata,
        )

        storage.add_region(region)
        imported.append(region)

        existing_identities.add(
            identity
        )

    return DetectorImportResult(
        imported=imported,
        skipped=skipped,
    )

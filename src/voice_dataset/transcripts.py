from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import TranscriptHypothesis
from .storage import DatasetStorage


TRANSCRIPT_OUTPUT_FORMAT = (
    "voice-dataset-transcript-output"
)

TRANSCRIPT_OUTPUT_VERSION = 1


@dataclass
class ValidatedTranscript:
    region_id: str
    text: str | None
    language: str | None
    confidence: float | None
    metadata: dict[str, Any]


@dataclass
class TranscriptOutput:
    transcriber_name: str
    transcriber_model: str | None
    representation: str
    hypotheses: list[ValidatedTranscript]


@dataclass
class TranscriptImportResult:
    imported: int
    skipped: int


def _validate_optional_string(
    value: str | None,
    field_name: str,
) -> None:
    if (
        value is not None
        and not isinstance(value, str)
    ):
        raise ValueError(
            f"{field_name} must be a string or null"
        )


def _validate_confidence(
    confidence: float | None,
) -> None:
    if confidence is None:
        return

    if (
        isinstance(confidence, bool)
        or not isinstance(
            confidence,
            (int, float),
        )
        or not math.isfinite(confidence)
    ):
        raise ValueError(
            "confidence must be a finite number or null"
        )


def attach_transcript(
    storage: DatasetStorage,
    region_id: str,
    name: str,
    *,
    text: str | None = None,
    language: str | None = None,
    model: str | None = None,
    representation: str | None = None,
    confidence: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(name, str) or not name:
        raise ValueError(
            "Transcript name must be a non-empty string"
        )

    _validate_optional_string(
        text,
        "text",
    )

    _validate_optional_string(
        language,
        "language",
    )

    _validate_optional_string(
        model,
        "model",
    )

    _validate_optional_string(
        representation,
        "representation",
    )

    _validate_confidence(
        confidence
    )

    if (
        metadata is not None
        and not isinstance(metadata, dict)
    ):
        raise ValueError(
            "metadata must be an object"
        )

    region = storage.get_region(
        region_id
    )

    if region is None:
        raise KeyError(
            f"Region does not exist: {region_id}"
        )

    if representation is not None:
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

        if representation not in representations:
            raise ValueError(
                f"Representation does not exist: "
                f"{region_id}/{representation}"
            )

    hypothesis = TranscriptHypothesis(
        text=text,
        language=language,
        model=model,
        representation=representation,
        confidence=(
            float(confidence)
            if confidence is not None
            else None
        ),
        metadata=(
            dict(metadata)
            if metadata is not None
            else {}
        ),
    )

    def update(
        record: dict[str, Any],
    ) -> dict[str, Any]:
        transcripts = record.get(
            "transcripts"
        )

        if not isinstance(
            transcripts,
            dict,
        ):
            raise ValueError(
                f"Invalid transcripts for {region_id}"
            )

        if name in transcripts:
            raise ValueError(
                f"Transcript already exists: "
                f"{region_id}/{name}"
            )

        transcripts[name] = (
            hypothesis.to_dict()
        )

        return record

    return storage.update_region(
        region_id,
        update,
    )


def load_transcript_output(
    path: Path,
) -> TranscriptOutput:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        document = json.load(file)

    if not isinstance(document, dict):
        raise ValueError(
            "Transcript output must be a JSON object"
        )

    if document.get("format") != (
        TRANSCRIPT_OUTPUT_FORMAT
    ):
        raise ValueError(
            "Unsupported transcript output format: "
            f"{document.get('format')!r}"
        )

    if document.get("version") != (
        TRANSCRIPT_OUTPUT_VERSION
    ):
        raise ValueError(
            "Unsupported transcript output version: "
            f"{document.get('version')!r}"
        )

    transcriber = document.get(
        "transcriber"
    )

    if not isinstance(transcriber, dict):
        raise ValueError(
            "Missing transcriber object"
        )

    transcriber_name = transcriber.get(
        "name"
    )

    if (
        not isinstance(
            transcriber_name,
            str,
        )
        or not transcriber_name
    ):
        raise ValueError(
            "Transcriber name must be "
            "a non-empty string"
        )

    transcriber_model = transcriber.get(
        "model"
    )

    _validate_optional_string(
        transcriber_model,
        "Transcriber model",
    )

    representation = document.get(
        "representation"
    )

    if (
        not isinstance(
            representation,
            str,
        )
        or not representation
    ):
        raise ValueError(
            "Representation must be "
            "a non-empty string"
        )

    hypotheses_value = document.get(
        "hypotheses"
    )

    if not isinstance(
        hypotheses_value,
        list,
    ):
        raise ValueError(
            "Hypotheses must be a list"
        )

    hypotheses = []
    seen_region_ids = set()

    for index, item in enumerate(
        hypotheses_value,
        start=1,
    ):
        if not isinstance(item, dict):
            raise ValueError(
                f"Hypothesis {index} "
                "must be an object"
            )

        region_id = item.get(
            "region_id"
        )

        if (
            not isinstance(region_id, str)
            or not region_id
        ):
            raise ValueError(
                f"Hypothesis {index} "
                "has invalid region_id"
            )

        if region_id in seen_region_ids:
            raise ValueError(
                "Duplicate region_id in "
                f"transcript output: {region_id}"
            )

        seen_region_ids.add(
            region_id
        )

        text = item.get("text")
        language = item.get("language")
        confidence = item.get(
            "confidence"
        )
        metadata = item.get(
            "metadata",
            {},
        )

        _validate_optional_string(
            text,
            f"Hypothesis {index} text",
        )

        _validate_optional_string(
            language,
            f"Hypothesis {index} language",
        )

        _validate_confidence(
            confidence
        )

        if not isinstance(metadata, dict):
            raise ValueError(
                f"Hypothesis {index} "
                "metadata must be an object"
            )

        hypotheses.append(
            ValidatedTranscript(
                region_id=region_id,
                text=text,
                language=language,
                confidence=(
                    float(confidence)
                    if confidence is not None
                    else None
                ),
                metadata=dict(metadata),
            )
        )

    return TranscriptOutput(
        transcriber_name=transcriber_name,
        transcriber_model=transcriber_model,
        representation=representation,
        hypotheses=hypotheses,
    )


def import_transcripts(
    storage: DatasetStorage,
    output: TranscriptOutput,
    name: str,
) -> TranscriptImportResult:
    if not isinstance(name, str) or not name:
        raise ValueError(
            "Transcript name must be a non-empty string"
        )

    regions = storage.regions.load()

    regions_by_id = {}

    for region in regions:
        region_id = region.get("id")

        if (
            not isinstance(region_id, str)
            or not region_id
        ):
            raise ValueError(
                "Dataset contains region "
                "with invalid id"
            )

        if region_id in regions_by_id:
            raise RuntimeError(
                f"Duplicate region id: {region_id}"
            )

        regions_by_id[region_id] = region

    #
    # Validate the complete import before
    # performing any writes.
    #
    for hypothesis in output.hypotheses:
        region = regions_by_id.get(
            hypothesis.region_id
        )

        if region is None:
            raise ValueError(
                "Transcript references unknown region: "
                f"{hypothesis.region_id}"
            )

        representations = region.get(
            "representations"
        )

        if not isinstance(
            representations,
            dict,
        ):
            raise ValueError(
                "Invalid representations for "
                f"{hypothesis.region_id}"
            )

        if (
            output.representation
            not in representations
        ):
            raise ValueError(
                "Representation does not exist: "
                f"{hypothesis.region_id}/"
                f"{output.representation}"
            )

        transcripts = region.get(
            "transcripts"
        )

        if not isinstance(
            transcripts,
            dict,
        ):
            raise ValueError(
                "Invalid transcripts for "
                f"{hypothesis.region_id}"
            )

    imported = 0
    skipped = 0

    for hypothesis in output.hypotheses:
        region = regions_by_id[
            hypothesis.region_id
        ]

        transcripts = region[
            "transcripts"
        ]

        if name in transcripts:
            skipped += 1
            continue

        attach_transcript(
            storage,
            hypothesis.region_id,
            name,
            text=hypothesis.text,
            language=hypothesis.language,
            model=output.transcriber_model,
            representation=output.representation,
            confidence=hypothesis.confidence,
            metadata={
                **hypothesis.metadata,
                "transcriber": (
                    output.transcriber_name
                ),
            },
        )

        imported += 1

    return TranscriptImportResult(
        imported=imported,
        skipped=skipped,
    )

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .continuous_asr import transcribe_source_qwen3
from .detectors import detect_source_regions
from .ingest import ingest
from .region_asr import transcribe_regions
from .separation import separate_source
from .sources import (
    resolve_source_representation_for_purpose,
)
from .storage import DatasetStorage
from .utterance_pipeline import (
    apply_source_utterance_turns,
    build_source_utterances,
    prepare_source_review_audio,
    prepare_source_speaker_evidence,
    source_utterance_reconciliation_is_complete,
    source_utterance_reconciliation_is_curated,
)
from .media import (
    choose_channel_mode,
    probe_audio_streams,
    select_audio_stream,
)


@dataclass(frozen=True)
class ProcessResult:
    source_id: str
    detector_ran: bool
    qwen_ran: bool
    turns_created: int
    turns_skipped: int
    turns_review: int


def _source_regions(
    storage: DatasetStorage,
    source_id: str,
) -> list[dict[str, Any]]:
    return [
        region
        for region in storage.regions.load()
        if region.get("source_id") == source_id
    ]


def _has_continuous_asr(
    storage: DatasetStorage,
    source_id: str,
    evidence_name: str,
) -> bool:
    source = storage.get_source(source_id)

    if source is None:
        return False

    metadata = source.get("metadata")

    if not isinstance(metadata, dict):
        return False

    continuous_asr = metadata.get("continuous_asr")

    if not isinstance(continuous_asr, dict):
        return False

    evidence = continuous_asr.get(evidence_name)

    return isinstance(evidence, dict)


def _validate_existing_source(
    record: dict[str, Any],
    *,
    source: Path,
    audio_stream: int | None,
    audio_language: str | None,
    channel: str,
    start: float | None,
    duration: float | None,
) -> None:
    media_path = record.get("media_path")

    if not isinstance(media_path, str) or not media_path:
        raise ValueError(
            "Existing source has invalid media_path"
        )

    existing_media = Path(media_path).resolve()

    if existing_media != source.resolve():
        raise ValueError(
            "Existing source refers to different media: "
            f"{existing_media} != {source.resolve()}"
        )

    metadata = record.get("metadata")

    if not isinstance(metadata, dict):
        return

    ingest_metadata = metadata.get("ingest")

    if not isinstance(ingest_metadata, dict):
        return

    requested_slice = {
        "requested_start": start,
        "requested_duration": duration,
    }

    for key, value in requested_slice.items():
        existing_value = ingest_metadata.get(key)

        if existing_value != value:
            raise ValueError(
                "Existing source has incompatible "
                f"{key}: "
                f"{existing_value!r} != {value!r}"
            )

    streams = probe_audio_streams(source)

    _, selected_stream = select_audio_stream(
        streams,
        audio_stream=audio_stream,
        language=audio_language,
    )

    selected_channel = choose_channel_mode(
        selected_stream,
        requested=channel,
    )

    existing_stream_index = ingest_metadata.get(
        "stream_index"
    )

    if existing_stream_index != selected_stream.index:
        raise ValueError(
            "Existing source uses incompatible "
            "audio stream: "
            f"{existing_stream_index!r} != "
            f"{selected_stream.index!r}"
        )

    existing_channel_mode = ingest_metadata.get(
        "channel_mode"
    )

    if existing_channel_mode != selected_channel:
        raise ValueError(
            "Existing source uses incompatible "
            "channel mode: "
            f"{existing_channel_mode!r} != "
            f"{selected_channel!r}"
        )


def process_source(
    source: Path,
    output: Path,
    *,
    source_id: str,
    audio_stream: int | None = None,
    audio_language: str | None = None,
    language: str | None = None,
    channel: str = "auto",
    start: float | None = None,
    duration: float | None = None,
) -> ProcessResult:
    source = source.resolve()
    output = output.resolve()

    storage = DatasetStorage(output)

    existing_source = storage.get_source(source_id)

    if existing_source is None:
        ingest(
            source,
            output,
            source_id,
            audio_stream=audio_stream,
            language=audio_language,
            channel=channel,
            start=start,
            duration=duration,
        )
    else:
        _validate_existing_source(
            existing_source,
            source=source,
            audio_stream=audio_stream,
            audio_language=audio_language,
            channel=channel,
            start=start,
            duration=duration,
        )

    boundary_name, _, _ = (
        resolve_source_representation_for_purpose(
            storage,
            source_id,
            "boundary_analysis",
        )
    )

    separate_source(
        storage,
        source_id,
        input_representation=boundary_name,
        output_representation="speech",
    )

    asr_name, _, _ = (
        resolve_source_representation_for_purpose(
            storage,
            source_id,
            "asr",
        )
    )

    detector_ran = False

    if not _source_regions(storage, source_id):
        detect_source_regions(
            storage,
            source_id,
            boundary_name,
        )
        detector_ran = True

    transcribe_regions(
        storage,
        source_id,
        representation_name=asr_name,
        transcript_name="whisper",
        language=language,
    )

    qwen_ran = False

    if not _has_continuous_asr(
        storage,
        source_id,
        "qwen3",
    ):
        transcribe_source_qwen3(
            storage,
            source_id,
            representation_name=asr_name,
            evidence_name="qwen3",
            language=language,
        )
        qwen_ran = True

    reconciliation_complete = (
        source_utterance_reconciliation_is_complete(
            storage,
            source_id,
            asr_evidence_name="qwen3",
        )
    )

    reconciliation_curated = (
        source_utterance_reconciliation_is_curated(
            storage,
            source_id,
            asr_evidence_name="qwen3",
        )
    )

    if reconciliation_complete or reconciliation_curated:
        turns_created = 0
        turns_skipped = 0
        turns_review = 0
    else:
        pipeline = build_source_utterances(
            storage,
            source_id,
            asr_evidence_name="qwen3",
        )

        turns = apply_source_utterance_turns(
            storage,
            source_id,
            pipeline,
            language=language,
            asr_evidence_name="qwen3",
        )

        turns_created = turns.created
        turns_skipped = turns.skipped
        turns_review = turns.review

    prepare_source_review_audio(
        storage,
        source_id,
    )

    prepare_source_speaker_evidence(
        storage,
        source_id,
    )

    return ProcessResult(
        source_id=source_id,
        detector_ran=detector_ran,
        qwen_ran=qwen_ran,
        turns_created=turns_created,
        turns_skipped=turns_skipped,
        turns_review=turns_review,
    )

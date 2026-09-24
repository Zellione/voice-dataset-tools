from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .storage import DatasetStorage


SORTFORMER_ACTIVITY_WINDOW = 0.160


@dataclass(frozen=True)
class SpeakerActivityEvidence:
    start_frame: int
    end_frame: int
    frame_start: float
    frame_end: float
    activity: tuple[float, ...]
    dominant_speaker: int
    dominant_activity: float


@dataclass(frozen=True)
class TurnBoundaryEvidence:
    left_region_id: str
    right_region_id: str

    boundary: float
    right_start: float
    gap: float
    touching: bool

    left_speaker: str | None
    right_speaker: str | None
    community_same_speaker: bool

    left_word_index: int | None
    right_word_index: int | None
    left_word: str | None
    right_word: str | None

    shared_word_indices: tuple[int, ...]
    qwen_word_crosses_left_end: bool

    left_utterance_index: int | None
    right_utterance_index: int | None
    qwen_same_utterance: bool

    suspicious_word_indices: tuple[int, ...]

    whisper_left_text: str | None
    whisper_right_text: str | None
    whisper_left_language: str | None
    whisper_right_language: str | None

    sortformer_before: SpeakerActivityEvidence
    sortformer_after: SpeakerActivityEvidence
    sortformer_same_dominant: bool
    sortformer_gap: SpeakerActivityEvidence | None


def _positive_overlap(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    return max(
        0.0,
        min(end_a, end_b) - max(start_a, start_b),
    )


def _word_indices_for_range(
    words: list[dict[str, Any]],
    start: float,
    end: float,
) -> tuple[int, ...]:
    return tuple(
        index
        for index, word in enumerate(words)
        if _positive_overlap(
            start,
            end,
            float(word["start"]),
            float(word["end"]),
        )
        > 0
    )


def _utterance_for_word(
    utterances: list[dict[str, Any]],
    word_index: int | None,
) -> int | None:
    if word_index is None:
        return None

    for index, utterance in enumerate(utterances):
        if (
            int(utterance["word_start"])
            <= word_index
            < int(utterance["word_end"])
        ):
            return index

    return None


def _activity_window(
    activity: np.ndarray,
    *,
    frame_duration: float,
    start: float,
    end: float,
) -> SpeakerActivityEvidence:
    frame_count = activity.shape[0]

    start_frame = max(
        0,
        int(start / frame_duration),
    )

    end_frame = min(
        frame_count,
        max(
            start_frame + 1,
            int(end / frame_duration),
        ),
    )

    selected = activity[
        start_frame:end_frame
    ]

    if selected.shape[0] == 0:
        raise ValueError(
            "SortFormer activity window is empty"
        )

    mean = selected.mean(axis=0)
    dominant_speaker = int(mean.argmax())

    return SpeakerActivityEvidence(
        start_frame=start_frame,
        end_frame=end_frame,
        frame_start=(
            start_frame * frame_duration
        ),
        frame_end=(
            end_frame * frame_duration
        ),
        activity=tuple(
            float(value)
            for value in mean
        ),
        dominant_speaker=dominant_speaker,
        dominant_activity=float(
            mean[dominant_speaker]
        ),
    )


def collect_turn_boundary_evidence(
    storage: DatasetStorage,
    source_id: str,
    *,
    asr_evidence_name: str = "qwen3",
    diarization_evidence_name: str = "sortformer",
    activity_window: float = SORTFORMER_ACTIVITY_WINDOW,
) -> list[TurnBoundaryEvidence]:
    source = storage.get_source(source_id)

    if source is None:
        raise KeyError(
            f"Source does not exist: {source_id}"
        )

    metadata = source.get("metadata", {})

    continuous_asr = metadata.get(
        "continuous_asr",
        {},
    )

    if not isinstance(continuous_asr, dict):
        raise ValueError(
            f"{source_id}: invalid continuous_asr metadata"
        )

    evidence = continuous_asr.get(
        asr_evidence_name
    )

    if not isinstance(evidence, dict):
        raise ValueError(
            "Continuous ASR evidence does not exist: "
            f"{source_id}/{asr_evidence_name}"
        )

    words = evidence.get("words")
    utterances = evidence.get("utterances")

    if not isinstance(words, list):
        raise ValueError(
            "Continuous ASR words must be a list"
        )

    if not isinstance(utterances, list):
        raise ValueError(
            "Continuous ASR utterances must be a list"
        )

    diarization = metadata.get(
        "diarization",
        {},
    )

    if not isinstance(diarization, dict):
        raise ValueError(
            f"{source_id}: invalid diarization metadata"
        )

    diarization_evidence = diarization.get(
        diarization_evidence_name
    )

    if not isinstance(
        diarization_evidence,
        dict,
    ):
        raise ValueError(
            "Diarization evidence does not exist: "
            f"{source_id}/"
            f"{diarization_evidence_name}"
        )

    activity_reference = (
        diarization_evidence.get("activity")
    )

    if not isinstance(
        activity_reference,
        dict,
    ):
        raise ValueError(
            "Diarization activity reference "
            "must be an object"
        )

    relative_path = activity_reference.get(
        "path"
    )
    frame_duration = activity_reference.get(
        "frame_duration"
    )

    if not isinstance(relative_path, str):
        raise ValueError(
            "Diarization activity path "
            "must be a string"
        )

    frame_duration = float(frame_duration)

    if frame_duration <= 0:
        raise ValueError(
            "Diarization frame duration "
            "must be positive"
        )

    if activity_window <= 0:
        raise ValueError(
            "Diarization activity window "
            "must be positive"
        )

    activity_path = (
        storage.root / Path(relative_path)
    )

    activity_raw = np.load(
        activity_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    if (
        activity_raw.ndim != 3
        or activity_raw.shape[0] != 1
    ):
        raise ValueError(
            "SortFormer activity must have shape "
            "(1, frames, speakers)"
        )

    activity = activity_raw[0]

    regions = [
        region
        for region in storage.regions.load()
        if region.get("source_id") == source_id
    ]

    regions.sort(
        key=lambda region: (
            float(region["source_start"]),
            float(region["source_end"]),
            region["id"],
        )
    )

    results: list[TurnBoundaryEvidence] = []

    for left, right in zip(
        regions,
        regions[1:],
    ):
        left_start = float(
            left["source_start"]
        )
        boundary = float(
            left["source_end"]
        )
        right_start = float(
            right["source_start"]
        )
        right_end = float(
            right["source_end"]
        )

        left_words = _word_indices_for_range(
            words,
            left_start,
            boundary,
        )
        right_words = _word_indices_for_range(
            words,
            right_start,
            right_end,
        )

        left_word_index = (
            left_words[-1]
            if left_words
            else None
        )
        right_word_index = (
            right_words[0]
            if right_words
            else None
        )

        shared_word_indices = tuple(
            sorted(
                set(left_words)
                & set(right_words)
            )
        )

        crossing_word_indices = tuple(
            index
            for index, word in enumerate(words)
            if (
                float(word["start"])
                < boundary
                < float(word["end"])
            )
        )

        relevant_word_indices = set(
            left_words
        )
        relevant_word_indices.update(
            right_words
        )
        relevant_word_indices.update(
            crossing_word_indices
        )

        suspicious_word_indices = tuple(
            sorted(
                index
                for index
                in relevant_word_indices
                if (
                    float(words[index]["end"])
                    - float(words[index]["start"])
                    <= 0
                    or
                    float(words[index]["end"])
                    - float(words[index]["start"])
                    > 2.0
                )
            )
        )

        left_utterance_index = (
            _utterance_for_word(
                utterances,
                left_word_index,
            )
        )
        right_utterance_index = (
            _utterance_for_word(
                utterances,
                right_word_index,
            )
        )

        left_speaker = left.get(
            "detector_label"
        )
        right_speaker = right.get(
            "detector_label"
        )

        left_whisper = (
            left.get("transcripts", {})
            .get("whisper", {})
        )
        right_whisper = (
            right.get("transcripts", {})
            .get("whisper", {})
        )

        before = _activity_window(
            activity,
            frame_duration=frame_duration,
            start=max(
                0.0,
                boundary - activity_window,
            ),
            end=boundary,
        )

        after = _activity_window(
            activity,
            frame_duration=frame_duration,
            start=boundary,
            end=boundary + activity_window,
        )

        gap = right_start - boundary

        sortformer_gap = (
            _activity_window(
                activity,
                frame_duration=frame_duration,
                start=boundary,
                end=right_start,
            )
            if gap > 0
            else None
        )

        results.append(
            TurnBoundaryEvidence(
                left_region_id=left["id"],
                right_region_id=right["id"],
                boundary=boundary,
                right_start=right_start,
                gap=gap,
                touching=(
                    right_start == boundary
                ),
                left_speaker=left_speaker,
                right_speaker=right_speaker,
                community_same_speaker=(
                    left_speaker == right_speaker
                ),
                left_word_index=left_word_index,
                right_word_index=right_word_index,
                left_word=(
                    str(
                        words[
                            left_word_index
                        ]["text"]
                    )
                    if left_word_index
                    is not None
                    else None
                ),
                right_word=(
                    str(
                        words[
                            right_word_index
                        ]["text"]
                    )
                    if right_word_index
                    is not None
                    else None
                ),
                shared_word_indices=(
                    shared_word_indices
                ),
                qwen_word_crosses_left_end=bool(
                    crossing_word_indices
                ),
                left_utterance_index=(
                    left_utterance_index
                ),
                right_utterance_index=(
                    right_utterance_index
                ),
                qwen_same_utterance=(
                    left_utterance_index
                    is not None
                    and left_utterance_index
                    == right_utterance_index
                ),
                suspicious_word_indices=(
                    suspicious_word_indices
                ),
                whisper_left_text=left_whisper.get("text"),
                whisper_right_text=right_whisper.get("text"),
                whisper_left_language=left_whisper.get("language"),
                whisper_right_language=right_whisper.get("language"),
                sortformer_before=before,
                sortformer_after=after,
                sortformer_same_dominant=(
                    before.dominant_speaker
                    == after.dominant_speaker
                ),
                sortformer_gap=sortformer_gap,
            )
        )

    return results

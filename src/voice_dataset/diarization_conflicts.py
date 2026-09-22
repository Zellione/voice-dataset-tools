from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .storage import DatasetStorage


SORTFORMER_ACTIVITY_WINDOW = 0.160


@dataclass(frozen=True)
class DiarizationActivityWindow:
    start_frame: int
    end_frame: int
    frame_start: float
    frame_end: float
    activity: tuple[float, ...]
    dominant_speaker: int
    dominant_activity: float


@dataclass(frozen=True)
class DiarizationBoundaryConflict:
    left_region_id: str
    right_region_id: str
    left_speaker: str | None
    right_speaker: str | None
    boundary: float
    region_gap: float
    crossing_word: str
    word_start: float
    word_end: float
    word_boundary_offset: float
    word_boundary_fraction: float
    before: DiarizationActivityWindow
    after: DiarizationActivityWindow


@dataclass(frozen=True)
class DiarizationBoundaryConflictRun:
    region_ids: tuple[str, ...]
    conflicts: tuple[
        DiarizationBoundaryConflict,
        ...,
    ]


def group_diarization_boundary_conflicts(
    conflicts: list[DiarizationBoundaryConflict],
) -> list[DiarizationBoundaryConflictRun]:
    if not conflicts:
        return []

    runs: list[
        DiarizationBoundaryConflictRun
    ] = []

    current: list[
        DiarizationBoundaryConflict
    ] = []

    def append_current() -> None:
        if not current:
            return

        region_ids = (
            current[0].left_region_id,
            *(
                conflict.right_region_id
                for conflict in current
            ),
        )

        runs.append(
            DiarizationBoundaryConflictRun(
                region_ids=region_ids,
                conflicts=tuple(current),
            )
        )

    for conflict in conflicts:
        if (
            current
            and current[-1].right_region_id
            != conflict.left_region_id
        ):
            append_current()
            current = []

        current.append(conflict)

    append_current()

    return runs


def _activity_window(
    activity: np.ndarray,
    *,
    frame_duration: float,
    start: float,
    end: float,
) -> DiarizationActivityWindow:
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

    return DiarizationActivityWindow(
        start_frame=start_frame,
        end_frame=end_frame,
        frame_start=start_frame * frame_duration,
        frame_end=end_frame * frame_duration,
        activity=tuple(
            float(value)
            for value in mean
        ),
        dominant_speaker=dominant_speaker,
        dominant_activity=float(
            mean[dominant_speaker]
        ),
    )


def analyze_diarization_boundary_conflicts(
    storage: DatasetStorage,
    source_id: str,
    *,
    asr_evidence_name: str = "qwen3",
    diarization_evidence_name: str = "sortformer",
    activity_window: float = (
        SORTFORMER_ACTIVITY_WINDOW
    ),
) -> list[DiarizationBoundaryConflict]:
    source = storage.get_source(source_id)

    if source is None:
        raise KeyError(
            f"Source does not exist: {source_id}"
        )

    metadata = source.get("metadata", {})

    if not isinstance(metadata, dict):
        raise ValueError(
            f"Source has invalid metadata: {source_id}"
        )

    continuous_asr = metadata.get(
        "continuous_asr",
        {},
    )

    if not isinstance(continuous_asr, dict):
        raise ValueError(
            "Source has invalid continuous_asr metadata: "
            f"{source_id}"
        )

    asr_evidence = continuous_asr.get(
        asr_evidence_name
    )

    if not isinstance(asr_evidence, dict):
        raise ValueError(
            "Continuous ASR evidence does not exist: "
            f"{source_id}/{asr_evidence_name}"
        )

    words = asr_evidence.get("words")

    if not isinstance(words, list):
        raise ValueError(
            "Continuous ASR words must be a list"
        )

    diarization = metadata.get(
        "diarization",
        {},
    )

    if not isinstance(diarization, dict):
        raise ValueError(
            "Source has invalid diarization metadata: "
            f"{source_id}"
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

    results: list[
        DiarizationBoundaryConflict
    ] = []

    for left, right in zip(
        regions,
        regions[1:],
    ):
        left_speaker = left.get(
            "detector_label"
        )
        right_speaker = right.get(
            "detector_label"
        )

        if left_speaker == right_speaker:
            continue

        boundary = float(
            left["source_end"]
        )
        right_start = float(
            right["source_start"]
        )

        crossing_words: list[
            dict[str, Any]
        ] = []

        for word in words:
            word_start = float(word["start"])
            word_end = float(word["end"])

            if (
                word_start
                < boundary
                < word_end
            ):
                crossing_words.append(word)

        for word in crossing_words:
            word_start = float(word["start"])
            word_end = float(word["end"])
            word_duration = (
                word_end - word_start
            )

            if word_duration <= 0:
                continue

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

            results.append(
                DiarizationBoundaryConflict(
                    left_region_id=left["id"],
                    right_region_id=right["id"],
                    left_speaker=left_speaker,
                    right_speaker=right_speaker,
                    boundary=boundary,
                    region_gap=(
                        right_start - boundary
                    ),
                    crossing_word=str(
                        word["text"]
                    ),
                    word_start=word_start,
                    word_end=word_end,
                    word_boundary_offset=(
                        boundary - word_start
                    ),
                    word_boundary_fraction=(
                        (
                            boundary
                            - word_start
                        )
                        / word_duration
                    ),
                    before=before,
                    after=after,
                )
            )

    return results

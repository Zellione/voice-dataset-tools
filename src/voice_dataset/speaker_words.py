from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .storage import DatasetStorage

from .diarization_conflicts import (
    analyze_continuous_word_region_conflicts,
)


@dataclass(frozen=True)
class WordRegionOverlap:
    region_id: str
    speaker: str | None
    start: float
    end: float
    overlap: float


@dataclass(frozen=True)
class SpeakerAttributedWord:
    index: int
    text: str
    start: float
    end: float

    overlaps: tuple[WordRegionOverlap, ...]

    speaker: str | None
    assignment_method: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class UnresolvedWordRun:
    word_indices: tuple[int, ...]
    start: float
    end: float
    text: str

    left_word_index: int | None
    right_word_index: int | None

    left_speaker: str | None
    right_speaker: str | None

    left_gap: float | None
    right_gap: float | None

    assignment_methods: tuple[str, ...]
    overlapping_region_ids: tuple[str, ...]
    overlapping_speakers: tuple[str, ...]


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


def _load_continuous_asr(
    storage: DatasetStorage,
    source_id: str,
    evidence_name: str,
) -> dict[str, Any]:
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

    evidence = continuous_asr.get(evidence_name)

    if not isinstance(evidence, dict):
        raise ValueError(
            "Continuous ASR evidence does not exist: "
            f"{source_id}/{evidence_name}"
        )

    words = evidence.get("words")

    if not isinstance(words, list):
        raise ValueError(
            "Continuous ASR words must be a list"
        )

    return evidence


def _source_regions(
    storage: DatasetStorage,
    source_id: str,
) -> list[dict[str, Any]]:
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

    return regions


def attribute_speakers_to_words(
    storage: DatasetStorage,
    source_id: str,
    *,
    evidence_name: str = "qwen3",
    words: list[dict[str, Any]] | None = None,
) -> list[SpeakerAttributedWord]:
    if words is None:
        evidence = _load_continuous_asr(
            storage,
            source_id,
            evidence_name,
        )
        words = evidence["words"]
    regions = _source_regions(
        storage,
        source_id,
    )

    results: list[SpeakerAttributedWord] = []

    for index, word in enumerate(words):
        text = word["text"]
        start = float(word["start"])
        end = float(word["end"])

        overlaps: list[WordRegionOverlap] = []

        for region in regions:
            region_start = float(
                region["source_start"]
            )
            region_end = float(
                region["source_end"]
            )

            overlap = _positive_overlap(
                start,
                end,
                region_start,
                region_end,
            )

            if overlap <= 0.0:
                continue

            overlaps.append(
                WordRegionOverlap(
                    region_id=region["id"],
                    speaker=region.get(
                        "detector_label"
                    ),
                    start=region_start,
                    end=region_end,
                    overlap=overlap,
                )
            )

        speakers = {
            overlap.speaker
            for overlap in overlaps
            if overlap.speaker is not None
        }

        if len(speakers) == 1:
            speaker = next(iter(speakers))

            if len(overlaps) == 1:
                assignment_method = (
                    "single_overlapping_region"
                )
            else:
                assignment_method = (
                    "consistent_overlapping_regions"
                )

        elif not overlaps:
            speaker = None
            assignment_method = (
                "no_overlapping_region"
            )

        elif not speakers:
            speaker = None
            assignment_method = (
                "no_speaker_label"
            )

        else:
            speaker = None
            assignment_method = (
                "conflicting_overlapping_speakers"
            )

        results.append(
            SpeakerAttributedWord(
                index=index,
                text=text,
                start=start,
                end=end,
                overlaps=tuple(overlaps),
                speaker=speaker,
                assignment_method=assignment_method,
            )
        )

    return results


def collect_unresolved_word_runs(
    words: list[SpeakerAttributedWord],
) -> list[UnresolvedWordRun]:
    runs: list[UnresolvedWordRun] = []

    index = 0

    while index < len(words):
        if words[index].speaker is not None:
            index += 1
            continue

        first = index

        while (
            index + 1 < len(words)
            and words[index + 1].speaker is None
        ):
            index += 1

        last = index

        run_words = words[first : last + 1]

        left = (
            words[first - 1]
            if first > 0
            else None
        )

        right = (
            words[last + 1]
            if last + 1 < len(words)
            else None
        )

        region_ids: list[str] = []
        speakers: list[str] = []

        for word in run_words:
            for overlap in word.overlaps:
                if overlap.region_id not in region_ids:
                    region_ids.append(
                        overlap.region_id
                    )

                if (
                    overlap.speaker is not None
                    and overlap.speaker not in speakers
                ):
                    speakers.append(
                        overlap.speaker
                    )

        runs.append(
            UnresolvedWordRun(
                word_indices=tuple(
                    word.index
                    for word in run_words
                ),
                start=run_words[0].start,
                end=run_words[-1].end,
                text=" ".join(
                    word.text
                    for word in run_words
                ),
                left_word_index=(
                    left.index
                    if left is not None
                    else None
                ),
                right_word_index=(
                    right.index
                    if right is not None
                    else None
                ),
                left_speaker=(
                    left.speaker
                    if left is not None
                    else None
                ),
                right_speaker=(
                    right.speaker
                    if right is not None
                    else None
                ),
                left_gap=(
                    max(
                        0.0,
                        run_words[0].start
                        - left.end,
                    )
                    if left is not None
                    else None
                ),
                right_gap=(
                    max(
                        0.0,
                        right.start
                        - run_words[-1].end,
                    )
                    if right is not None
                    else None
                ),
                assignment_methods=tuple(
                    dict.fromkeys(
                        word.assignment_method
                        for word in run_words
                    )
                ),
                overlapping_region_ids=tuple(
                    region_ids
                ),
                overlapping_speakers=tuple(
                    speakers
                ),
            )
        )

        index += 1

    return runs


def resolve_consistent_speaker_context(
    words: list[SpeakerAttributedWord],
) -> list[SpeakerAttributedWord]:
    resolved = list(words)

    runs = collect_unresolved_word_runs(words)

    for run in runs:
        if run.left_speaker is None:
            continue

        if run.right_speaker is None:
            continue

        if run.left_speaker != run.right_speaker:
            continue

        speaker = run.left_speaker

        for word_index in run.word_indices:
            word = resolved[word_index]

            resolved[word_index] = (
                SpeakerAttributedWord(
                    index=word.index,
                    text=word.text,
                    start=word.start,
                    end=word.end,
                    overlaps=word.overlaps,
                    speaker=speaker,
                    assignment_method=(
                        "consistent_speaker_context"
                    ),
                )
            )

    return resolved


def resolve_fragmented_speaker_words(
    storage: DatasetStorage,
    source_id: str,
    words: list[SpeakerAttributedWord],
    *,
    evidence_name: str = "qwen3",
    geometry_words: list[dict[str, Any]] | None = None,
    boundary_after_word_indices: set[int] | None = None,
) -> list[SpeakerAttributedWord]:
    resolved = list(words)

    if boundary_after_word_indices is None:
        boundary_after_word_indices = set()

    conflicts = analyze_continuous_word_region_conflicts(
        storage,
        source_id,
        asr_evidence_name=evidence_name,
        words=geometry_words,
    )

    conflicts_by_span = {
        (
            conflict.word,
            conflict.word_start,
            conflict.word_end,
        ): conflict
        for conflict in conflicts
    }

    for index, word in enumerate(words):
        if word.speaker is not None:
            continue

        if (
            word.assignment_method
            != "conflicting_overlapping_speakers"
        ):
            continue

        conflict = conflicts_by_span.get(
            (
                word.text,
                word.start,
                word.end,
            )
        )

        if conflict is None:
            continue

        conflict_speakers = {
            speaker
            for speaker in conflict.region_speakers
            if speaker is not None
        }

        left_speaker = (
            resolved[index - 1].speaker
            if index > 0
            else None
        )

        right_speaker = (
            words[index + 1].speaker
            if index + 1 < len(words)
            else None
        )

        candidates: list[
            tuple[str, str]
        ] = []

        if (
            left_speaker is not None
            and left_speaker in conflict_speakers
        ):
            candidates.append(
                (
                    left_speaker,
                    "fragmentation_left_context",
                )
            )

        if (
            right_speaker is not None
            and right_speaker in conflict_speakers
        ):
            candidates.append(
                (
                    right_speaker,
                    "fragmentation_right_context",
                )
            )

        candidate_speakers = {
            speaker
            for speaker, _ in candidates
        }

        if len(candidate_speakers) != 1:
            continue

        speaker = next(iter(candidate_speakers))

        methods = {
            method
            for candidate, method in candidates
            if candidate == speaker
        }

        if len(methods) == 2:
            assignment_method = (
                "fragmentation_consistent_context"
            )
        else:
            assignment_method = next(
                iter(methods)
            )

        resolved[index] = (
            SpeakerAttributedWord(
                index=word.index,
                text=word.text,
                start=word.start,
                end=word.end,
                overlaps=word.overlaps,
                speaker=speaker,
                assignment_method=assignment_method,
            )
        )

    for conflict in conflicts:
        conflict_position = next(
            (
                position
                for position, word in enumerate(resolved)
                if (
                    word.text == conflict.word
                    and word.start == conflict.word_start
                    and word.end == conflict.word_end
                )
            ),
            None,
        )

        if conflict_position is None:
            continue

        conflict_speakers = {
            speaker
            for speaker in conflict.region_speakers
            if speaker is not None
        }

        if len(conflict_speakers) != 2:
            continue

        span_start = conflict_position

        while span_start > 0:
            previous = resolved[span_start - 1]

            if (
                previous.index
                in boundary_after_word_indices
            ):
                break

            span_start -= 1

        span_end = conflict_position

        while span_end + 1 < len(resolved):
            current = resolved[span_end]

            if (
                current.index
                in boundary_after_word_indices
            ):
                break

            span_end += 1

        span = resolved[
            span_start : span_end + 1
        ]

        speakers_before = {
            word.speaker
            for word in span[
                : conflict_position - span_start
            ]
            if word.speaker is not None
        }

        speakers_after = {
            word.speaker
            for word in span[
                conflict_position - span_start + 1 :
            ]
            if word.speaker is not None
        }

        if len(speakers_before) != 1:
            continue

        if len(speakers_after) != 1:
            continue

        before_speaker = next(
            iter(speakers_before)
        )
        after_speaker = next(
            iter(speakers_after)
        )

        if before_speaker == after_speaker:
            continue

        if before_speaker not in conflict_speakers:
            continue

        if after_speaker not in conflict_speakers:
            continue

        next_speaker = next(
            (
                word.speaker
                for word in resolved[span_end + 1 :]
                if word.speaker is not None
            ),
            None,
        )

        if next_speaker != before_speaker:
            continue

        for position in range(
            conflict_position,
            span_end + 1,
        ):
            word = resolved[position]

            if (
                word.speaker is not None
                and word.speaker
                not in conflict_speakers
            ):
                continue

            resolved[position] = (
                SpeakerAttributedWord(
                    index=word.index,
                    text=word.text,
                    start=word.start,
                    end=word.end,
                    overlaps=word.overlaps,
                    speaker=before_speaker,
                    assignment_method=(
                        "fragmentation_sat_context"
                    ),
                )
            )

    return resolved

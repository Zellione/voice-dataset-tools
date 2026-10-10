from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TurnTranscriptProjection:
    source_start: float
    source_end: float
    word_indices: tuple[int, ...]
    complete_utterance_indices: tuple[int, ...]
    intersecting_utterance_indices: tuple[int, ...]
    suspicious_word_indices: tuple[int, ...]


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


def project_continuous_asr_to_range(
    evidence: dict[str, Any],
    source_start: float,
    source_end: float,
) -> TurnTranscriptProjection:
    if source_end <= source_start:
        raise ValueError(
            "Projection range must have positive duration"
        )

    words = evidence["words"]
    utterances = evidence["utterances"]

    word_indices: list[int] = []
    suspicious_word_indices: list[int] = []

    for index, word in enumerate(words):
        word_start = float(word["start"])
        word_end = float(word["end"])

        word_duration = word_end - word_start

        overlap = _positive_overlap(
            source_start,
            source_end,
            word_start,
            word_end,
        )

        zero_duration_in_range = (
            word_duration == 0
            and source_start
            <= word_start + 1e-6
            and word_start
            <= source_end + 1e-6
        )

        if (
            overlap <= 0
            and not zero_duration_in_range
        ):
            continue

        word_indices.append(index)

        # Diagnostic only. This deliberately does not decide whether
        # the word belongs to the turn. It merely exposes obviously
        # weak temporal evidence.
        if (
            word_duration <= 0
            or word_duration > 2.0
        ):
            suspicious_word_indices.append(index)

    selected_words = set(word_indices)

    complete_utterances: list[int] = []
    intersecting_utterances: list[int] = []

    for index, utterance in enumerate(utterances):
        word_start = int(utterance["word_start"])
        word_end = int(utterance["word_end"])

        utterance_words = set(
            range(word_start, word_end)
        )

        if not (
            utterance_words
            & selected_words
        ):
            continue

        intersecting_utterances.append(index)

        if utterance_words <= selected_words:
            complete_utterances.append(index)

    return TurnTranscriptProjection(
        source_start=source_start,
        source_end=source_end,
        word_indices=tuple(word_indices),
        complete_utterance_indices=tuple(
            complete_utterances
        ),
        intersecting_utterance_indices=tuple(
            intersecting_utterances
        ),
        suspicious_word_indices=tuple(
            suspicious_word_indices
        ),
    )

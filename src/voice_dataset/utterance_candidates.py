from __future__ import annotations

from dataclasses import dataclass

from .speaker_words import SpeakerAttributedWord


MAX_WORD_ALIGNMENT_DURATION = 5.0
MAX_INTER_WORD_ALIGNMENT_GAP = 5.0


@dataclass(frozen=True)
class UtteranceCandidate:
    start_word_index: int
    end_word_index: int

    start: float
    end: float

    text: str
    speaker: str | None

    word_indices: tuple[int, ...]
    unresolved_word_indices: tuple[int, ...]
    known_speakers: tuple[str, ...]

    alignment_issue_word_indices: tuple[int, ...]

    end_boundary: str


def _known_speakers(
    words: list[SpeakerAttributedWord],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            word.speaker
            for word in words
            if word.speaker is not None
        )
    )


def _alignment_issue_word_indices(
    words: list[SpeakerAttributedWord],
) -> tuple[int, ...]:
    issue_indices: list[int] = []

    def add(index: int) -> None:
        if index not in issue_indices:
            issue_indices.append(index)

    for position, word in enumerate(words):
        if word.duration > MAX_WORD_ALIGNMENT_DURATION:
            add(word.index)

        if position == 0:
            continue

        previous = words[position - 1]
        gap = word.start - previous.end

        if gap > MAX_INTER_WORD_ALIGNMENT_GAP:
            add(previous.index)
            add(word.index)

    return tuple(issue_indices)


def _make_candidate(
    words: list[SpeakerAttributedWord],
    *,
    end_boundary: str,
) -> UtteranceCandidate:
    if not words:
        raise ValueError(
            "Cannot create empty utterance candidate"
        )

    speakers = _known_speakers(words)

    return UtteranceCandidate(
        start_word_index=words[0].index,
        end_word_index=words[-1].index,
        start=words[0].start,
        end=words[-1].end,
        text=" ".join(
            word.text
            for word in words
        ),
        speaker=(
            speakers[0]
            if len(speakers) == 1
            else None
        ),
        word_indices=tuple(
            word.index
            for word in words
        ),
        unresolved_word_indices=tuple(
            word.index
            for word in words
            if word.speaker is None
        ),
        known_speakers=speakers,
        alignment_issue_word_indices=(
            _alignment_issue_word_indices(words)
        ),
        end_boundary=end_boundary,
    )


def build_utterance_candidates(
    words: list[SpeakerAttributedWord],
    *,
    boundary_after_word_indices: set[int],
) -> list[UtteranceCandidate]:
    if not words:
        return []

    candidates: list[UtteranceCandidate] = []
    current: list[SpeakerAttributedWord] = []

    for position, word in enumerate(words):
        current.append(word)

        is_last = position == len(words) - 1

        if is_last:
            candidates.append(
                _make_candidate(
                    current,
                    end_boundary="source_end",
                )
            )
            break

        next_word = words[position + 1]

        direct_speaker_change = (
            word.speaker is not None
            and next_word.speaker is not None
            and word.speaker != next_word.speaker
        )

        sat_boundary = (
            word.index
            in boundary_after_word_indices
        )

        alignment_gap = (
            next_word.start - word.end
            > MAX_INTER_WORD_ALIGNMENT_GAP
        )

        if (
            not direct_speaker_change
            and not sat_boundary
            and not alignment_gap
        ):
            continue

        boundary_reasons = []

        if direct_speaker_change:
            boundary_reasons.append(
                "speaker_change"
            )

        if sat_boundary:
            boundary_reasons.append(
                "sat"
            )

        if alignment_gap:
            boundary_reasons.append(
                "alignment_gap"
            )

        end_boundary = "+".join(
            boundary_reasons
        )

        candidates.append(
            _make_candidate(
                current,
                end_boundary=end_boundary,
            )
        )

        current = []

    return candidates

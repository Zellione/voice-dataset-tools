from __future__ import annotations

from dataclasses import dataclass

from .speaker_similarity import (
    VoiceMatch,
    VoiceTurnMatch,
)


@dataclass(frozen=True)
class EmbeddingVoiceCandidate:
    voice_id: str
    embedding_name: str
    score: float
    support: int
    matches: tuple[VoiceTurnMatch, ...]


@dataclass(frozen=True)
class SpeakerCandidate:
    voice_id: str
    score: float
    encoder_count: int
    embedding_scores: tuple[
        EmbeddingVoiceCandidate,
        ...,
    ]


def aggregate_voice_matches(
    matches: list[VoiceMatch],
    *,
    embedding_name: str,
    top_k: int = 3,
) -> list[EmbeddingVoiceCandidate]:
    if top_k <= 0:
        raise ValueError(
            "top_k must be positive"
        )

    candidates: list[EmbeddingVoiceCandidate] = []

    for voice_match in matches:
        selected = voice_match.matches[:top_k]

        if not selected:
            continue

        score = sum(
            match.similarity
            for match in selected
        ) / len(selected)

        candidates.append(
            EmbeddingVoiceCandidate(
                voice_id=voice_match.voice_id,
                embedding_name=embedding_name,
                score=score,
                support=len(voice_match.matches),
                matches=voice_match.matches,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            -candidate.score,
            candidate.voice_id,
        )
    )

    return candidates


def combine_embedding_candidates(
    *candidate_groups: list[EmbeddingVoiceCandidate],
) -> list[SpeakerCandidate]:
    by_voice: dict[
        str,
        list[EmbeddingVoiceCandidate],
    ] = {}

    for candidates in candidate_groups:
        for candidate in candidates:
            existing = by_voice.setdefault(
                candidate.voice_id,
                [],
            )

            if any(
                item.embedding_name
                == candidate.embedding_name
                for item in existing
            ):
                raise ValueError(
                    "duplicate embedding candidate "
                    f"for voice {candidate.voice_id}: "
                    f"{candidate.embedding_name}"
                )

            existing.append(candidate)

    results: list[SpeakerCandidate] = []

    for voice_id, embedding_scores in by_voice.items():
        ordered_scores = sorted(
            embedding_scores,
            key=lambda candidate: (
                candidate.embedding_name,
            ),
        )

        score = sum(
            candidate.score
            for candidate in ordered_scores
        ) / len(ordered_scores)

        results.append(
            SpeakerCandidate(
                voice_id=voice_id,
                score=score,
                encoder_count=len(ordered_scores),
                embedding_scores=tuple(ordered_scores),
            )
        )

    results.sort(
        key=lambda candidate: (
            -candidate.score,
            candidate.voice_id,
        )
    )

    return results

import pytest

from voice_dataset.speaker_candidates import (
    aggregate_voice_matches,
)
from voice_dataset.speaker_similarity import (
    VoiceMatch,
    VoiceTurnMatch,
)


def test_aggregate_voice_matches_uses_top_three_mean():
    matches = [
        VoiceMatch(
            voice_id="voice_001",
            matches=(
                VoiceTurnMatch(
                    turn_id="turn_000001",
                    similarity=0.90,
                ),
                VoiceTurnMatch(
                    turn_id="turn_000002",
                    similarity=0.80,
                ),
                VoiceTurnMatch(
                    turn_id="turn_000003",
                    similarity=0.70,
                ),
                VoiceTurnMatch(
                    turn_id="turn_000004",
                    similarity=0.10,
                ),
            ),
        ),
    ]

    candidates = aggregate_voice_matches(
        matches,
        embedding_name="ecapa",
    )

    assert len(candidates) == 1

    candidate = candidates[0]

    assert candidate.voice_id == "voice_001"
    assert candidate.embedding_name == "ecapa"
    assert candidate.score == pytest.approx(0.80)
    assert candidate.support == 4
    assert tuple(
        match.turn_id
        for match in candidate.matches
    ) == (
        "turn_000001",
        "turn_000002",
        "turn_000003",
        "turn_000004",
    )


def test_aggregate_voice_matches_uses_available_references():
    matches = [
        VoiceMatch(
            voice_id="voice_001",
            matches=(
                VoiceTurnMatch(
                    turn_id="turn_000001",
                    similarity=0.70,
                ),
                VoiceTurnMatch(
                    turn_id="turn_000002",
                    similarity=0.50,
                ),
            ),
        ),
    ]

    candidates = aggregate_voice_matches(
        matches,
        embedding_name="wespeaker",
    )

    assert candidates[0].score == pytest.approx(0.60)
    assert candidates[0].support == 2


def test_aggregate_voice_matches_ranks_voices_by_score():
    matches = [
        VoiceMatch(
            voice_id="voice_001",
            matches=(
                VoiceTurnMatch(
                    turn_id="turn_000001",
                    similarity=0.60,
                ),
            ),
        ),
        VoiceMatch(
            voice_id="voice_002",
            matches=(
                VoiceTurnMatch(
                    turn_id="turn_000002",
                    similarity=0.80,
                ),
            ),
        ),
    ]

    candidates = aggregate_voice_matches(
        matches,
        embedding_name="ecapa",
    )

    assert [
        candidate.voice_id
        for candidate in candidates
    ] == [
        "voice_002",
        "voice_001",
    ]


def test_aggregate_voice_matches_rejects_invalid_top_k():
    with pytest.raises(
        ValueError,
        match="top_k must be positive",
    ):
        aggregate_voice_matches(
            [],
            embedding_name="ecapa",
            top_k=0,
        )


def test_aggregate_voice_matches_breaks_ties_by_voice_id():
    matches = [
        VoiceMatch(
            voice_id="voice_002",
            matches=(
                VoiceTurnMatch(
                    turn_id="turn_000002",
                    similarity=0.75,
                ),
            ),
        ),
        VoiceMatch(
            voice_id="voice_001",
            matches=(
                VoiceTurnMatch(
                    turn_id="turn_000001",
                    similarity=0.75,
                ),
            ),
        ),
    ]

    candidates = aggregate_voice_matches(
        matches,
        embedding_name="ecapa",
    )

    assert [
        candidate.voice_id
        for candidate in candidates
    ] == [
        "voice_001",
        "voice_002",
    ]


def test_aggregate_voice_matches_skips_empty_voice():
    matches = [
        VoiceMatch(
            voice_id="voice_001",
            matches=(),
        ),
    ]

    candidates = aggregate_voice_matches(
        matches,
        embedding_name="ecapa",
    )

    assert candidates == []


def test_combine_embedding_candidates_averages_encoder_scores():
    from voice_dataset.speaker_candidates import (
        EmbeddingVoiceCandidate,
        combine_embedding_candidates,
    )

    ecapa = [
        EmbeddingVoiceCandidate(
            voice_id="voice_001",
            embedding_name="ecapa_speaker",
            score=0.80,
            support=4,
            matches=(),
        ),
        EmbeddingVoiceCandidate(
            voice_id="voice_002",
            embedding_name="ecapa_speaker",
            score=0.60,
            support=3,
            matches=(),
        ),
    ]

    wespeaker = [
        EmbeddingVoiceCandidate(
            voice_id="voice_001",
            embedding_name="wespeaker_speaker",
            score=0.70,
            support=4,
            matches=(),
        ),
        EmbeddingVoiceCandidate(
            voice_id="voice_002",
            embedding_name="wespeaker_speaker",
            score=0.90,
            support=3,
            matches=(),
        ),
    ]

    candidates = combine_embedding_candidates(
        ecapa,
        wespeaker,
    )

    assert [
        candidate.voice_id
        for candidate in candidates
    ] == [
        "voice_001",
        "voice_002",
    ]

    assert candidates[0].score == pytest.approx(0.75)
    assert candidates[1].score == pytest.approx(0.75)

    assert candidates[0].encoder_count == 2

    assert {
        score.embedding_name: score.score
        for score in candidates[1].embedding_scores
    } == {
        "ecapa_speaker": pytest.approx(0.60),
        "wespeaker_speaker": pytest.approx(0.90),
    }


def test_combine_embedding_candidates_keeps_single_encoder_candidate():
    from voice_dataset.speaker_candidates import (
        EmbeddingVoiceCandidate,
        combine_embedding_candidates,
    )

    ecapa = [
        EmbeddingVoiceCandidate(
            voice_id="voice_001",
            embedding_name="ecapa_speaker",
            score=0.72,
            support=2,
            matches=(),
        ),
    ]

    candidates = combine_embedding_candidates(
        ecapa,
        [],
    )

    assert len(candidates) == 1
    assert candidates[0].voice_id == "voice_001"
    assert candidates[0].score == pytest.approx(0.72)
    assert candidates[0].encoder_count == 1


def test_combine_embedding_candidates_breaks_score_ties_by_voice_id():
    from voice_dataset.speaker_candidates import (
        EmbeddingVoiceCandidate,
        combine_embedding_candidates,
    )

    ecapa = [
        EmbeddingVoiceCandidate(
            voice_id="voice_002",
            embedding_name="ecapa_speaker",
            score=0.75,
            support=1,
            matches=(),
        ),
        EmbeddingVoiceCandidate(
            voice_id="voice_001",
            embedding_name="ecapa_speaker",
            score=0.75,
            support=1,
            matches=(),
        ),
    ]

    candidates = combine_embedding_candidates(
        ecapa,
        [],
    )

    assert [
        candidate.voice_id
        for candidate in candidates
    ] == [
        "voice_001",
        "voice_002",
    ]


def test_combine_embedding_candidates_rejects_duplicate_embedding_for_voice():
    from voice_dataset.speaker_candidates import (
        EmbeddingVoiceCandidate,
        combine_embedding_candidates,
    )

    first = [
        EmbeddingVoiceCandidate(
            voice_id="voice_001",
            embedding_name="ecapa_speaker",
            score=0.80,
            support=3,
            matches=(),
        ),
    ]

    duplicate = [
        EmbeddingVoiceCandidate(
            voice_id="voice_001",
            embedding_name="ecapa_speaker",
            score=0.70,
            support=2,
            matches=(),
        ),
    ]

    with pytest.raises(
        ValueError,
        match="duplicate embedding candidate",
    ):
        combine_embedding_candidates(
            first,
            duplicate,
        )

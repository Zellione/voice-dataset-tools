import pytest
from voice_dataset.reviewer_view import (
    build_reviewer_turn_view,
)
from voice_dataset.speaker_candidates import (
    EmbeddingVoiceCandidate,
    SpeakerCandidate,
)


def test_build_reviewer_turn_view_maps_core_turn_state():
    turn = {
        "id": "turn_000013",
        "source_id": "source_001",
        "source_start": 207.426,
        "source_end": 208.786,
        "source_regions": [
            "region_000020",
        ],
        "language": "en",
        "transcript": "They're right not to trust us",
        "representations": {
            "review": {},
            "raw": {},
        },
        "assignment": {
            "status": "unknown",
            "voice_id": None,
            "method": None,
            "confidence": None,
        },
        "review": {
            "status": "pending",
            "boundary": {
                "status": "complete",
            },
        },
        "metadata": {
            "automatic_pipeline": {
                "status": "review",
                "review_reasons": [
                    "speaker_ambiguous",
                ],
            },
            "boundary_evidence": {
                "near_source_start": True,
                "near_source_end": False,
            },
        },
    }

    view = build_reviewer_turn_view(
        turn,
        position=13,
        total=28,
        voices={},
        speaker_candidates=[],
    )

    assert view.turn_id == "turn_000013"
    assert view.source_id == "source_001"
    assert view.position == 13
    assert view.total == 28

    assert view.start == 207.426
    assert view.end == 208.786
    assert view.duration == pytest.approx(1.36)

    assert view.transcript == (
        "They're right not to trust us"
    )
    assert view.language == "en"

    assert view.region_ids == (
        "region_000020",
    )
    assert view.representation_names == (
        "review",
        "raw",
    )

    assert view.review_status == "pending"

    assert view.boundary.status == "complete"
    assert view.boundary.near_source_start is True
    assert view.boundary.near_source_end is False

    assert view.assignment.status == "unknown"
    assert view.assignment.voice_id is None

    assert view.automatic_pipeline.status == "review"
    assert view.automatic_pipeline.review_reasons == (
        "speaker_ambiguous",
    )


def test_build_reviewer_turn_view_maps_speaker_candidates():
    candidate = SpeakerCandidate(
        voice_id="voice_004",
        score=0.81,
        encoder_count=2,
        embedding_scores=(
            EmbeddingVoiceCandidate(
                voice_id="voice_004",
                embedding_name="ecapa_speaker",
                score=0.79,
                support=6,
                matches=(),
            ),
            EmbeddingVoiceCandidate(
                voice_id="voice_004",
                embedding_name="wespeaker_speaker",
                score=0.83,
                support=6,
                matches=(),
            ),
        ),
    )

    voices = {
        "voice_004": {
            "id": "voice_004",
            "character": "Vander",
            "language": "en",
        },
    }

    view = build_reviewer_turn_view(
        {
            "id": "turn_000001",
            "source_id": "source_001",
            "source_start": 1.0,
            "source_end": 2.0,
        },
        position=1,
        total=1,
        voices=voices,
        speaker_candidates=[candidate],
    )

    assert len(view.speaker_candidates) == 1

    speaker = view.speaker_candidates[0]

    assert speaker.voice_id == "voice_004"
    assert speaker.character == "Vander"
    assert speaker.score == 0.81
    assert speaker.encoder_count == 2

    assert [
        item.embedding_name
        for item in speaker.embedding_scores
    ] == [
        "ecapa_speaker",
        "wespeaker_speaker",
    ]

    assert [
        item.support
        for item in speaker.embedding_scores
    ] == [
        6,
        6,
    ]


def test_build_reviewer_turn_view_handles_missing_optional_state():
    view = build_reviewer_turn_view(
        {
            "id": "turn_000001",
            "source_id": "source_001",
            "source_start": 1.0,
            "source_end": 2.0,
        },
        position=1,
        total=1,
        voices={},
        speaker_candidates=[],
    )

    assert view.transcript is None
    assert view.language is None
    assert view.region_ids == ()
    assert view.representation_names == ()

    assert view.review_status == "pending"

    assert view.boundary.status == "unknown"
    assert view.boundary.near_source_start is False
    assert view.boundary.near_source_end is False

    assert view.assignment.status == "unknown"
    assert view.assignment.voice_id is None

    assert view.automatic_pipeline.status is None
    assert view.automatic_pipeline.review_reasons == ()

    assert view.speaker_candidates == ()
    assert view.speaker_evidence.speaker is None
    assert view.speaker_evidence.known_speakers == ()
    assert (
        view.speaker_evidence.unresolved_word_indices
        == ()
    )

    assert view.alignment.status is None
    assert view.alignment.issue_word_indices == ()
    assert view.alignment.recovery is None

    assert view.edge_recovery is None


def test_build_reviewer_turn_view_maps_speaker_evidence():
    turn = {
        "id": "turn_000001",
        "source_id": "source_001",
        "source_start": 1.0,
        "source_end": 2.0,
        "metadata": {
            "speaker_evidence": {
                "speaker": "SPEAKER_04",
                "known_speakers": [
                    "SPEAKER_04",
                    "SPEAKER_01",
                ],
                "unresolved_word_indices": [
                    3,
                    7,
                ],
            },
        },
    }

    view = build_reviewer_turn_view(
        turn,
        position=1,
        total=1,
        voices={},
        speaker_candidates=[],
    )

    assert view.speaker_evidence.speaker == "SPEAKER_04"
    assert view.speaker_evidence.known_speakers == (
        "SPEAKER_04",
        "SPEAKER_01",
    )
    assert (
        view.speaker_evidence.unresolved_word_indices
        == (
            3,
            7,
        )
    )


def test_build_reviewer_turn_view_maps_alignment_evidence():
    turn = {
        "id": "turn_000001",
        "source_id": "source_001",
        "source_start": 1.0,
        "source_end": 2.0,
        "metadata": {
            "alignment_evidence": {
                "status": "review",
                "issue_word_indices": [
                    2,
                    3,
                ],
                "recovery": {
                    "status": "candidate",
                    "source_start": 1.15,
                    "source_end": 1.82,
                    "speaker": "SPEAKER_04",
                    "region_ids": [
                        "region_000020",
                    ],
                    "matched_token_count": 4,
                    "candidate_token_count": 5,
                },
            },
        },
    }

    view = build_reviewer_turn_view(
        turn,
        position=1,
        total=1,
        voices={},
        speaker_candidates=[],
    )

    assert view.alignment.status == "review"
    assert view.alignment.issue_word_indices == (
        2,
        3,
    )

    recovery = view.alignment.recovery

    assert recovery is not None
    assert recovery.status == "candidate"
    assert recovery.source_start == 1.15
    assert recovery.source_end == 1.82
    assert recovery.speaker == "SPEAKER_04"
    assert recovery.region_ids == (
        "region_000020",
    )
    assert recovery.matched_token_count == 4
    assert recovery.candidate_token_count == 5


def test_build_reviewer_turn_view_handles_missing_alignment_recovery():
    turn = {
        "id": "turn_000001",
        "source_id": "source_001",
        "source_start": 1.0,
        "source_end": 2.0,
        "metadata": {
            "alignment_evidence": {
                "status": "valid",
                "issue_word_indices": [],
            },
        },
    }

    view = build_reviewer_turn_view(
        turn,
        position=1,
        total=1,
        voices={},
        speaker_candidates=[],
    )

    assert view.alignment.status == "valid"
    assert view.alignment.issue_word_indices == ()
    assert view.alignment.recovery is None


def test_build_reviewer_turn_view_maps_end_edge_recovery():
    turn = {
        "id": "turn_000001",
        "source_id": "source_001",
        "source_start": 1.0,
        "source_end": 2.0,
        "metadata": {
            "edge_evidence": {
                "status": "suggested",
                "method": (
                    "community_whisper_"
                    "edge_extension"
                ),
                "edge": "end",
                "source_end": 2.4,
                "region_id": "region_000020",
                "speaker": "SPEAKER_04",
                "candidate_token": "kid",
                "whisper_token": "kiddo",
                "candidate_text": "It gets easier kid",
                "whisper_text": "It gets easier kiddo",
            },
        },
    }

    view = build_reviewer_turn_view(
        turn,
        position=1,
        total=1,
        voices={},
        speaker_candidates=[],
    )

    edge = view.edge_recovery

    assert edge is not None
    assert edge.status == "suggested"
    assert edge.method == (
        "community_whisper_edge_extension"
    )
    assert edge.edge == "end"

    assert edge.source_start is None
    assert edge.source_end == 2.4

    assert edge.region_id == "region_000020"
    assert edge.conflicting_region_id is None
    assert edge.speaker == "SPEAKER_04"

    assert edge.candidate_token == "kid"
    assert edge.whisper_token == "kiddo"

    assert edge.candidate_text == (
        "It gets easier kid"
    )
    assert edge.whisper_text == (
        "It gets easier kiddo"
    )
    assert edge.conflicting_whisper_text is None


def test_build_reviewer_turn_view_maps_start_edge_recovery():
    turn = {
        "id": "turn_000001",
        "source_id": "source_001",
        "source_start": 1.0,
        "source_end": 2.0,
        "metadata": {
            "edge_evidence": {
                "status": "suggested",
                "method": (
                    "community_whisper_"
                    "start_conflict"
                ),
                "edge": "start",
                "source_start": 1.2,
                "region_id": "region_000021",
                "conflicting_region_id": (
                    "region_000020"
                ),
                "speaker": "SPEAKER_01",
                "candidate_text": "You're walking",
                "whisper_text": "You're walking",
                "conflicting_whisper_text": (
                    "Something else"
                ),
            },
        },
    }

    view = build_reviewer_turn_view(
        turn,
        position=1,
        total=1,
        voices={},
        speaker_candidates=[],
    )

    edge = view.edge_recovery

    assert edge is not None
    assert edge.status == "suggested"
    assert edge.method == (
        "community_whisper_start_conflict"
    )
    assert edge.edge == "start"

    assert edge.source_start == 1.2
    assert edge.source_end is None

    assert edge.region_id == "region_000021"
    assert edge.conflicting_region_id == (
        "region_000020"
    )
    assert edge.speaker == "SPEAKER_01"

    assert edge.candidate_token is None
    assert edge.whisper_token is None

    assert edge.candidate_text == "You're walking"
    assert edge.whisper_text == "You're walking"
    assert edge.conflicting_whisper_text == (
        "Something else"
    )

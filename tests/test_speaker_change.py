import numpy as np
import pytest

from voice_dataset.speaker_change import (
    SpeakerEmbeddingChange,
    SpeakerChangeBoundary,
    cosine_similarity,
    is_suspected_speaker_change,
    speaker_change_window_kind,
    plan_turn_speaker_change_boundaries,
    score_speaker_change,
)


def test_plan_turn_speaker_change_boundaries_uses_non_overlapping_flanks():
    turn = {
        "id": "turn_001",
        "source_start": 10.0,
        "source_end": 13.0,
    }

    boundaries = (
        plan_turn_speaker_change_boundaries(
            turn,
            window=1.0,
            step=0.25,
        )
    )

    assert [
        boundary.boundary
        for boundary in boundaries
    ] == pytest.approx([
        11.0,
        11.25,
        11.5,
        11.75,
        12.0,
    ])

    first = boundaries[0]

    assert first.left_start == pytest.approx(
        10.0
    )
    assert first.left_end == pytest.approx(
        11.0
    )
    assert first.right_start == pytest.approx(
        11.0
    )
    assert first.right_end == pytest.approx(
        12.0
    )


def test_plan_turn_speaker_change_boundaries_skips_short_turn():
    boundaries = (
        plan_turn_speaker_change_boundaries(
            {
                "id": "turn_001",
                "source_start": 10.0,
                "source_end": 11.5,
            },
            window=1.0,
            step=0.25,
        )
    )

    assert boundaries == ()


def test_plan_turn_speaker_change_boundaries_handles_exact_minimum_duration():
    boundaries = (
        plan_turn_speaker_change_boundaries(
            {
                "id": "turn_001",
                "source_start": 10.0,
                "source_end": 12.0,
            },
            window=1.0,
            step=0.25,
        )
    )

    assert len(boundaries) == 1
    assert boundaries[0].boundary == pytest.approx(
        11.0
    )


def test_cosine_similarity_normalizes_inputs():
    similarity = cosine_similarity(
        np.array([2.0, 0.0]),
        np.array([3.0, 0.0]),
    )

    assert similarity == pytest.approx(1.0)


def test_score_speaker_change_combines_encoders():
    boundary = SpeakerChangeBoundary(
        turn_id="turn_001",
        boundary=11.0,
        left_start=10.0,
        left_end=11.0,
        right_start=11.0,
        right_end=12.0,
    )

    change = score_speaker_change(
        boundary,
        ecapa_left=np.array([1.0, 0.0]),
        ecapa_right=np.array([0.0, 1.0]),
        wespeaker_left=np.array([1.0, 0.0]),
        wespeaker_right=np.array([0.8, 0.6]),
    )

    assert change.ecapa_similarity == pytest.approx(
        0.0
    )

    assert (
        change.wespeaker_similarity
        == pytest.approx(0.8)
    )

    assert change.mean_similarity == pytest.approx(
        0.4
    )

    assert (
        change.encoder_disagreement
        == pytest.approx(0.8)
    )

    assert change.divergence == pytest.approx(
        0.6
    )


def test_analyze_speaker_change_boundaries_handles_empty_input():
    from voice_dataset.speaker_change import (
        analyze_speaker_change_boundaries,
    )

    assert (
        analyze_speaker_change_boundaries(
            storage=None,
            source_id="source",
            boundaries=[],
        )
        == ()
    )


def test_adjacent_boundaries_reuse_shared_window_geometry():
    first = SpeakerChangeBoundary(
        turn_id="turn_a",
        boundary=11.0,
        left_start=10.0,
        left_end=11.0,
        right_start=11.0,
        right_end=12.0,
    )

    second = SpeakerChangeBoundary(
        turn_id="turn_b",
        boundary=12.0,
        left_start=11.0,
        left_end=12.0,
        right_start=12.0,
        right_end=13.0,
    )

    assert (
        first.right_start,
        first.right_end,
    ) == (
        second.left_start,
        second.left_end,
    )


def test_speech_aware_planning_rejects_pause_only_side():
    words = [
        {
            "text": "right",
            "start": 11.5,
            "end": 12.0,
        },
    ]

    boundaries = (
        plan_turn_speaker_change_boundaries(
            {
                "id": "turn_001",
                "source_start": 9.0,
                "source_end": 12.0,
            },
            words=words,
            window=1.0,
            step=0.25,
            minimum_window=0.5,
            minimum_speech_ratio=0.5,
        )
    )

    assert boundaries == ()


def test_speech_aware_planning_accepts_short_valid_side():
    words = [
        {
            "text": "left",
            "start": 10.0,
            "end": 11.0,
        },
        {
            "text": "right",
            "start": 11.0,
            "end": 11.64,
        },
    ]

    boundaries = (
        plan_turn_speaker_change_boundaries(
            {
                "id": "turn_001",
                "source_start": 10.0,
                "source_end": 11.64,
            },
            words=words,
            window=1.0,
            step=0.25,
            minimum_window=0.5,
            minimum_speech_ratio=0.5,
        )
    )

    assert len(boundaries) >= 1

    assert (
        boundaries[0].left_end
        - boundaries[0].left_start
    ) >= 0.5

    assert (
        boundaries[0].right_end
        - boundaries[0].right_start
    ) >= 0.5


def test_speech_aware_planning_shifts_away_from_pause():
    words = [
        {
            "text": "speech",
            "start": 10.0,
            "end": 10.8,
        },
        {
            "text": "speech",
            "start": 11.2,
            "end": 12.0,
        },
    ]

    boundaries = (
        plan_turn_speaker_change_boundaries(
            {
                "id": "turn_001",
                "source_start": 9.5,
                "source_end": 12.5,
            },
            words=words,
            window=1.0,
            step=1.0,
            minimum_window=0.5,
            max_shift=0.75,
            shift_step=0.05,
            minimum_speech_ratio=0.5,
        )
    )

    assert boundaries

    boundary = boundaries[0]

    assert boundary.left_start >= 9.5
    assert boundary.right_end <= 12.5


def test_long_turn_uses_only_full_speaker_change_windows():
    words = [
        {
            "text": "speech",
            "start": 10.0,
            "end": 16.0,
        },
    ]

    boundaries = (
        plan_turn_speaker_change_boundaries(
            {
                "id": "turn_001",
                "source_start": 10.0,
                "source_end": 16.0,
            },
            words=words,
            window=1.0,
            step=0.5,
            minimum_window=0.5,
            minimum_speech_ratio=0.5,
        )
    )

    assert boundaries

    assert boundaries[0].boundary == pytest.approx(
        11.0
    )
    assert boundaries[-1].boundary == pytest.approx(
        15.0
    )

    for boundary in boundaries:
        assert (
            boundary.left_end
            - boundary.left_start
        ) == pytest.approx(1.0)

        assert (
            boundary.right_end
            - boundary.right_start
        ) == pytest.approx(1.0)


def test_short_turn_uses_single_adaptive_center_boundary():
    words = [
        {
            "text": "speech",
            "start": 10.0,
            "end": 11.6,
        },
    ]

    boundaries = (
        plan_turn_speaker_change_boundaries(
            {
                "id": "turn_001",
                "source_start": 10.0,
                "source_end": 11.6,
            },
            words=words,
            window=1.0,
            step=0.25,
            minimum_window=0.5,
            minimum_speech_ratio=0.5,
        )
    )

    assert len(boundaries) == 1
    assert boundaries[0].boundary == pytest.approx(
        10.8
    )


def test_speaker_change_window_kind_full():
    boundary = SpeakerChangeBoundary(
        turn_id="turn_001",
        boundary=11.0,
        left_start=10.0,
        left_end=11.0,
        right_start=11.0,
        right_end=12.0,
    )

    assert (
        speaker_change_window_kind(
            boundary
        )
        == "full"
    )


def test_speaker_change_window_kind_adaptive():
    boundary = SpeakerChangeBoundary(
        turn_id="turn_001",
        boundary=11.0,
        left_start=10.0,
        left_end=11.0,
        right_start=11.0,
        right_end=11.7,
    )

    assert (
        speaker_change_window_kind(
            boundary
        )
        == "adaptive"
    )


def test_suspected_speaker_change_uses_full_threshold():
    boundary = SpeakerChangeBoundary(
        turn_id="turn_001",
        boundary=11.0,
        left_start=10.0,
        left_end=11.0,
        right_start=11.0,
        right_end=12.0,
    )

    change = SpeakerEmbeddingChange(
        turn_id="turn_001",
        boundary=11.0,
        ecapa_similarity=0.08,
        wespeaker_similarity=0.06,
        mean_similarity=0.07,
        encoder_disagreement=0.02,
        divergence=0.93,
    )

    assert is_suspected_speaker_change(
        change,
        boundary,
    )


def test_suspected_speaker_change_uses_adaptive_threshold():
    boundary = SpeakerChangeBoundary(
        turn_id="turn_001",
        boundary=11.0,
        left_start=10.0,
        left_end=11.0,
        right_start=11.0,
        right_end=11.7,
    )

    change = SpeakerEmbeddingChange(
        turn_id="turn_001",
        boundary=11.0,
        ecapa_similarity=0.08,
        wespeaker_similarity=0.06,
        mean_similarity=0.07,
        encoder_disagreement=0.02,
        divergence=0.93,
    )

    assert is_suspected_speaker_change(
        change,
        boundary,
    )

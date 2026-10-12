import numpy as np
import pytest

from voice_dataset.speaker_change import (
    SpeakerChangeBoundary,
    cosine_similarity,
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

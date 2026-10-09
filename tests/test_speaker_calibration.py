import pytest

from voice_dataset.speaker_calibration import (
    build_calibration_observation,
    evaluate_calibration_rule,
    filter_calibration_observations,
    speaker_calibration_observation_to_dict,
    summarize_calibration_observations,
    summarize_speaker_candidates,
    SpeakerCalibrationObservation,
)
from voice_dataset.speaker_candidates import (
    EmbeddingVoiceCandidate,
    SpeakerCandidate,
)
from voice_dataset.speaker_similarity import (
    VoiceTurnMatch,
)


def test_calibration_observation_serializes_for_storage():
    observation = SpeakerCalibrationObservation(
        turn_id="turn_001",
        source_id="episode_01",
        confirmed_voice_id="voice_001",
        predicted_voice_id="voice_001",
        correct=True,
        top_score=0.81,
        runner_up_score=0.43,
        margin=0.38,
        encoder_count=2,
        embedding_scores={
            "ecapa_speaker": 0.78,
            "wespeaker_speaker": 0.84,
        },
        support={
            "ecapa_speaker": 5,
            "wespeaker_speaker": 7,
        },
        source_support={
            "ecapa_speaker": 2,
            "wespeaker_speaker": 3,
        },
        total_source_support=3,
    )

    result = (
        speaker_calibration_observation_to_dict(
            observation
        )
    )

    assert result == {
        "schema_version": 1,
        "turn_id": "turn_001",
        "source_id": "episode_01",
        "confirmed_voice_id": "voice_001",
        "predicted_voice_id": "voice_001",
        "correct": True,
        "top_score": pytest.approx(0.81),
        "runner_up_score": pytest.approx(0.43),
        "margin": pytest.approx(0.38),
        "encoder_count": 2,
        "embedding_scores": {
            "ecapa_speaker": pytest.approx(0.78),
            "wespeaker_speaker": pytest.approx(0.84),
        },
        "support": {
            "ecapa_speaker": 5,
            "wespeaker_speaker": 7,
        },
        "source_support": {
            "ecapa_speaker": 2,
            "wespeaker_speaker": 3,
        },
        "total_source_support": 3,
    }


def test_summarize_speaker_candidates_records_top1_and_margin():
    candidates = [
        SpeakerCandidate(
            voice_id="voice_001",
            score=0.81,
            encoder_count=2,
            embedding_scores=(
                EmbeddingVoiceCandidate(
                    voice_id="voice_001",
                    embedding_name="ecapa_speaker",
                    score=0.78,
                    support=5,
                    matches=(
                        VoiceTurnMatch(
                            turn_id="turn_001",
                            similarity=0.90,
                            source_id="episode_01",
                        ),
                        VoiceTurnMatch(
                            turn_id="turn_002",
                            similarity=0.88,
                            source_id="episode_01",
                        ),
                        VoiceTurnMatch(
                            turn_id="turn_003",
                            similarity=0.85,
                            source_id="episode_02",
                        ),
                    ),
                ),
                EmbeddingVoiceCandidate(
                    voice_id="voice_001",
                    embedding_name="wespeaker_speaker",
                    score=0.84,
                    support=7,
                    matches=(
                        VoiceTurnMatch(
                            turn_id="turn_004",
                            similarity=0.92,
                            source_id="episode_01",
                        ),
                        VoiceTurnMatch(
                            turn_id="turn_005",
                            similarity=0.89,
                            source_id="episode_02",
                        ),
                        VoiceTurnMatch(
                            turn_id="turn_006",
                            similarity=0.87,
                            source_id="episode_03",
                        ),
                    ),
                ),
            ),
        ),
        SpeakerCandidate(
            voice_id="voice_002",
            score=0.43,
            encoder_count=2,
            embedding_scores=(
                EmbeddingVoiceCandidate(
                    voice_id="voice_002",
                    embedding_name="ecapa_speaker",
                    score=0.40,
                    support=3,
                    matches=(),
                ),
                EmbeddingVoiceCandidate(
                    voice_id="voice_002",
                    embedding_name="wespeaker_speaker",
                    score=0.46,
                    support=4,
                    matches=(),
                ),
            ),
        ),
    ]

    summary = summarize_speaker_candidates(
        candidates
    )

    assert summary.top_voice_id == "voice_001"
    assert summary.top_score == pytest.approx(0.81)

    assert summary.runner_up_voice_id == "voice_002"
    assert summary.runner_up_score == pytest.approx(0.43)

    assert summary.margin == pytest.approx(0.38)

    assert summary.encoder_count == 2

    assert summary.embedding_scores == {
        "ecapa_speaker": pytest.approx(0.78),
        "wespeaker_speaker": pytest.approx(0.84),
    }

    assert summary.support == {
        "ecapa_speaker": 5,
        "wespeaker_speaker": 7,
    }

    assert summary.source_support == {
        "ecapa_speaker": 2,
        "wespeaker_speaker": 3,
    }

    assert summary.total_source_support == 3


def test_summarize_speaker_candidates_handles_empty_list():
    summary = summarize_speaker_candidates(
        []
    )

    assert summary.top_voice_id is None
    assert summary.top_score is None
    assert summary.runner_up_voice_id is None
    assert summary.runner_up_score is None
    assert summary.margin is None
    assert summary.encoder_count == 0
    assert summary.embedding_scores == {}
    assert summary.support == {}
    assert summary.total_source_support == 0


def test_summarize_speaker_candidates_handles_single_candidate():
    candidate = SpeakerCandidate(
        voice_id="voice_001",
        score=0.72,
        encoder_count=1,
        embedding_scores=(
            EmbeddingVoiceCandidate(
                voice_id="voice_001",
                embedding_name="ecapa_speaker",
                score=0.72,
                support=4,
                matches=(),
            ),
        ),
    )

    summary = summarize_speaker_candidates(
        [candidate]
    )

    assert summary.top_voice_id == "voice_001"
    assert summary.top_score == pytest.approx(0.72)

    assert summary.runner_up_voice_id is None
    assert summary.runner_up_score is None
    assert summary.margin is None

    assert summary.encoder_count == 1
    assert summary.embedding_scores == {
        "ecapa_speaker": pytest.approx(0.72),
    }
    assert summary.support == {
        "ecapa_speaker": 4,
    }
    assert summary.total_source_support == 0


def test_source_support_counts_distinct_sources_only():
    candidate = SpeakerCandidate(
        voice_id="voice_001",
        score=0.80,
        encoder_count=1,
        embedding_scores=(
            EmbeddingVoiceCandidate(
                voice_id="voice_001",
                embedding_name="ecapa_speaker",
                score=0.80,
                support=4,
                matches=(
                    VoiceTurnMatch(
                        turn_id="turn_001",
                        similarity=0.91,
                        source_id="episode_01",
                    ),
                    VoiceTurnMatch(
                        turn_id="turn_002",
                        similarity=0.89,
                        source_id="episode_01",
                    ),
                    VoiceTurnMatch(
                        turn_id="turn_003",
                        similarity=0.86,
                        source_id="episode_01",
                    ),
                    VoiceTurnMatch(
                        turn_id="turn_004",
                        similarity=0.82,
                        source_id="episode_02",
                    ),
                ),
            ),
        ),
    )

    summary = summarize_speaker_candidates(
        [candidate]
    )

    assert summary.support == {
        "ecapa_speaker": 4,
    }

    assert summary.source_support == {
        "ecapa_speaker": 2,
    }

    assert summary.total_source_support == 2


def test_build_calibration_observation_records_ground_truth():
    candidates = [
        SpeakerCandidate(
            voice_id="voice_001",
            score=0.82,
            encoder_count=2,
            embedding_scores=(
                EmbeddingVoiceCandidate(
                    voice_id="voice_001",
                    embedding_name="ecapa_speaker",
                    score=0.80,
                    support=6,
                    matches=(
                        VoiceTurnMatch(
                            turn_id="ref_001",
                            similarity=0.91,
                            source_id="episode_01",
                        ),
                        VoiceTurnMatch(
                            turn_id="ref_002",
                            similarity=0.88,
                            source_id="episode_02",
                        ),
                    ),
                ),
                EmbeddingVoiceCandidate(
                    voice_id="voice_001",
                    embedding_name="wespeaker_speaker",
                    score=0.84,
                    support=5,
                    matches=(
                        VoiceTurnMatch(
                            turn_id="ref_003",
                            similarity=0.90,
                            source_id="episode_01",
                        ),
                        VoiceTurnMatch(
                            turn_id="ref_004",
                            similarity=0.87,
                            source_id="episode_03",
                        ),
                    ),
                ),
            ),
        ),
        SpeakerCandidate(
            voice_id="voice_002",
            score=0.31,
            encoder_count=2,
            embedding_scores=(),
        ),
    ]

    observation = build_calibration_observation(
        turn_id="turn_000123",
        source_id="episode_04",
        confirmed_voice_id="voice_001",
        candidates=candidates,
    )

    assert observation.turn_id == "turn_000123"
    assert observation.source_id == "episode_04"

    assert observation.confirmed_voice_id == "voice_001"
    assert observation.predicted_voice_id == "voice_001"
    assert observation.correct is True

    assert observation.top_score == pytest.approx(0.82)
    assert observation.runner_up_score == pytest.approx(0.31)
    assert observation.margin == pytest.approx(0.51)

    assert observation.encoder_count == 2

    assert observation.support == {
        "ecapa_speaker": 6,
        "wespeaker_speaker": 5,
    }

    assert observation.source_support == {
        "ecapa_speaker": 2,
        "wespeaker_speaker": 2,
    }

    assert observation.total_source_support == 3


def test_build_calibration_observation_marks_wrong_prediction():
    candidates = [
        SpeakerCandidate(
            voice_id="voice_001",
            score=0.76,
            encoder_count=2,
            embedding_scores=(),
        ),
        SpeakerCandidate(
            voice_id="voice_002",
            score=0.61,
            encoder_count=2,
            embedding_scores=(),
        ),
    ]

    observation = build_calibration_observation(
        turn_id="turn_000999",
        source_id="episode_05",
        confirmed_voice_id="voice_002",
        candidates=candidates,
    )

    assert observation.predicted_voice_id == "voice_001"
    assert observation.confirmed_voice_id == "voice_002"
    assert observation.correct is False

    assert observation.top_score == pytest.approx(0.76)
    assert observation.runner_up_score == pytest.approx(0.61)
    assert observation.margin == pytest.approx(0.15)


def test_summarize_calibration_observations_reports_accuracy():
    observations = [
        build_calibration_observation(
            turn_id="turn_001",
            source_id="episode_01",
            confirmed_voice_id="voice_001",
            candidates=[
                SpeakerCandidate(
                    voice_id="voice_001",
                    score=0.82,
                    encoder_count=2,
                    embedding_scores=(),
                ),
                SpeakerCandidate(
                    voice_id="voice_002",
                    score=0.30,
                    encoder_count=2,
                    embedding_scores=(),
                ),
            ],
        ),
        build_calibration_observation(
            turn_id="turn_002",
            source_id="episode_01",
            confirmed_voice_id="voice_002",
            candidates=[
                SpeakerCandidate(
                    voice_id="voice_001",
                    score=0.70,
                    encoder_count=2,
                    embedding_scores=(),
                ),
                SpeakerCandidate(
                    voice_id="voice_002",
                    score=0.60,
                    encoder_count=2,
                    embedding_scores=(),
                ),
            ],
        ),
        build_calibration_observation(
            turn_id="turn_003",
            source_id="episode_02",
            confirmed_voice_id="voice_003",
            candidates=[],
        ),
    ]

    stats = summarize_calibration_observations(
        observations
    )

    assert stats.observation_count == 3
    assert stats.prediction_count == 2
    assert stats.correct_count == 1

    assert stats.top1_accuracy == pytest.approx(
        0.5
    )


def test_summarize_calibration_observations_handles_empty_list():
    stats = summarize_calibration_observations(
        []
    )

    assert stats.observation_count == 0
    assert stats.prediction_count == 0
    assert stats.correct_count == 0
    assert stats.top1_accuracy is None


def test_summarize_calibration_observations_handles_no_predictions():
    observations = [
        build_calibration_observation(
            turn_id="turn_001",
            source_id="episode_01",
            confirmed_voice_id="voice_001",
            candidates=[],
        ),
        build_calibration_observation(
            turn_id="turn_002",
            source_id="episode_02",
            confirmed_voice_id="voice_002",
            candidates=[],
        ),
    ]

    stats = summarize_calibration_observations(
        observations
    )

    assert stats.observation_count == 2
    assert stats.prediction_count == 0
    assert stats.correct_count == 0
    assert stats.top1_accuracy is None


def test_filter_calibration_observations_applies_evidence_thresholds():
    observations = [
        SpeakerCalibrationObservation(
            turn_id="turn_001",
            source_id="episode_01",
            confirmed_voice_id="voice_001",
            predicted_voice_id="voice_001",
            correct=True,
            top_score=0.85,
            runner_up_score=0.30,
            margin=0.55,
            encoder_count=2,
            embedding_scores={},
            support={},
            source_support={},
            total_source_support=3,
        ),
        SpeakerCalibrationObservation(
            turn_id="turn_002",
            source_id="episode_02",
            confirmed_voice_id="voice_002",
            predicted_voice_id="voice_001",
            correct=False,
            top_score=0.72,
            runner_up_score=0.60,
            margin=0.12,
            encoder_count=2,
            embedding_scores={},
            support={},
            source_support={},
            total_source_support=3,
        ),
        SpeakerCalibrationObservation(
            turn_id="turn_003",
            source_id="episode_03",
            confirmed_voice_id="voice_003",
            predicted_voice_id="voice_003",
            correct=True,
            top_score=0.80,
            runner_up_score=0.40,
            margin=0.40,
            encoder_count=1,
            embedding_scores={},
            support={},
            source_support={},
            total_source_support=4,
        ),
        SpeakerCalibrationObservation(
            turn_id="turn_004",
            source_id="episode_04",
            confirmed_voice_id="voice_004",
            predicted_voice_id="voice_004",
            correct=True,
            top_score=0.83,
            runner_up_score=0.44,
            margin=0.39,
            encoder_count=2,
            embedding_scores={},
            support={},
            source_support={},
            total_source_support=1,
        ),
    ]

    filtered = filter_calibration_observations(
        observations,
        minimum_margin=0.30,
        minimum_source_support=2,
        minimum_encoder_count=2,
    )

    assert [
        observation.turn_id
        for observation in filtered
    ] == [
        "turn_001",
    ]


def test_filter_calibration_observations_without_thresholds_keeps_predictions():
    observations = [
        SpeakerCalibrationObservation(
            turn_id="turn_001",
            source_id="episode_01",
            confirmed_voice_id="voice_001",
            predicted_voice_id="voice_001",
            correct=True,
            top_score=0.50,
            runner_up_score=None,
            margin=None,
            encoder_count=1,
            embedding_scores={},
            support={},
            source_support={},
            total_source_support=0,
        ),
        SpeakerCalibrationObservation(
            turn_id="turn_002",
            source_id="episode_01",
            confirmed_voice_id="voice_002",
            predicted_voice_id=None,
            correct=False,
            top_score=None,
            runner_up_score=None,
            margin=None,
            encoder_count=0,
            embedding_scores={},
            support={},
            source_support={},
            total_source_support=0,
        ),
    ]

    filtered = filter_calibration_observations(
        observations
    )

    assert [
        observation.turn_id
        for observation in filtered
    ] == [
        "turn_001",
    ]


def test_filter_calibration_observations_rejects_missing_margin_when_required():
    observation = SpeakerCalibrationObservation(
        turn_id="turn_001",
        source_id="episode_01",
        confirmed_voice_id="voice_001",
        predicted_voice_id="voice_001",
        correct=True,
        top_score=0.80,
        runner_up_score=None,
        margin=None,
        encoder_count=2,
        embedding_scores={},
        support={},
        source_support={},
        total_source_support=3,
    )

    filtered = filter_calibration_observations(
        [observation],
        minimum_margin=0.20,
    )

    assert filtered == []


def test_evaluate_calibration_rule_filters_and_summarizes():
    observations = [
        SpeakerCalibrationObservation(
            turn_id="turn_001",
            source_id="episode_01",
            confirmed_voice_id="voice_001",
            predicted_voice_id="voice_001",
            correct=True,
            top_score=0.86,
            runner_up_score=0.30,
            margin=0.56,
            encoder_count=2,
            embedding_scores={},
            support={},
            source_support={},
            total_source_support=3,
        ),
        SpeakerCalibrationObservation(
            turn_id="turn_002",
            source_id="episode_02",
            confirmed_voice_id="voice_002",
            predicted_voice_id="voice_001",
            correct=False,
            top_score=0.74,
            runner_up_score=0.66,
            margin=0.08,
            encoder_count=2,
            embedding_scores={},
            support={},
            source_support={},
            total_source_support=3,
        ),
        SpeakerCalibrationObservation(
            turn_id="turn_003",
            source_id="episode_03",
            confirmed_voice_id="voice_003",
            predicted_voice_id="voice_003",
            correct=True,
            top_score=0.81,
            runner_up_score=0.41,
            margin=0.40,
            encoder_count=2,
            embedding_scores={},
            support={},
            source_support={},
            total_source_support=2,
        ),
    ]

    stats = evaluate_calibration_rule(
        observations,
        minimum_margin=0.30,
        minimum_source_support=2,
        minimum_encoder_count=2,
    )

    assert stats.observation_count == 2
    assert stats.prediction_count == 2
    assert stats.correct_count == 2
    assert stats.top1_accuracy == pytest.approx(1.0)

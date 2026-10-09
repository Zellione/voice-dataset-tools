from __future__ import annotations

from dataclasses import dataclass

from .speaker_candidates import SpeakerCandidate


@dataclass(frozen=True)
class SpeakerCandidateSummary:
    top_voice_id: str | None
    top_score: float | None

    runner_up_voice_id: str | None
    runner_up_score: float | None

    margin: float | None

    encoder_count: int

    embedding_scores: dict[str, float]
    support: dict[str, int]
    source_support: dict[str, int]

    total_source_support: int


@dataclass(frozen=True)
class SpeakerCalibrationObservation:
    turn_id: str
    source_id: str

    confirmed_voice_id: str
    predicted_voice_id: str | None

    correct: bool

    top_score: float | None
    runner_up_score: float | None
    margin: float | None

    encoder_count: int

    embedding_scores: dict[str, float]
    support: dict[str, int]
    source_support: dict[str, int]

    total_source_support: int


@dataclass(frozen=True)
class SpeakerCalibrationStats:
    observation_count: int
    prediction_count: int
    correct_count: int
    top1_accuracy: float | None


def summarize_speaker_candidates(
    candidates: list[SpeakerCandidate],
) -> SpeakerCandidateSummary:
    if not candidates:
        return SpeakerCandidateSummary(
            top_voice_id=None,
            top_score=None,
            runner_up_voice_id=None,
            runner_up_score=None,
            margin=None,
            encoder_count=0,
            embedding_scores={},
            support={},
            source_support={},
            total_source_support=0,
        )

    top = candidates[0]

    runner_up = (
        candidates[1]
        if len(candidates) > 1
        else None
    )

    runner_up_score = (
        runner_up.score
        if runner_up is not None
        else None
    )

    margin = (
        top.score - runner_up_score
        if runner_up_score is not None
        else None
    )

    all_source_ids = {
        match.source_id
        for candidate in top.embedding_scores
        for match in candidate.matches
        if match.source_id is not None
    }

    return SpeakerCandidateSummary(
        top_voice_id=top.voice_id,
        top_score=top.score,
        runner_up_voice_id=(
            runner_up.voice_id
            if runner_up is not None
            else None
        ),
        runner_up_score=runner_up_score,
        margin=margin,
        encoder_count=top.encoder_count,
        embedding_scores={
            candidate.embedding_name:
                candidate.score
            for candidate
            in top.embedding_scores
        },
        support={
            candidate.embedding_name:
                candidate.support
            for candidate
            in top.embedding_scores
        },
        source_support={
            candidate.embedding_name: len({
                match.source_id
                for match in candidate.matches
                if match.source_id is not None
            })
            for candidate
            in top.embedding_scores
        },
        total_source_support=len(
            all_source_ids
        ),
    )


def build_calibration_observation(
    *,
    turn_id: str,
    source_id: str,
    confirmed_voice_id: str,
    candidates: list[SpeakerCandidate],
) -> SpeakerCalibrationObservation:
    summary = summarize_speaker_candidates(
        candidates
    )

    return SpeakerCalibrationObservation(
        turn_id=turn_id,
        source_id=source_id,
        confirmed_voice_id=confirmed_voice_id,
        predicted_voice_id=summary.top_voice_id,
        correct=(
            summary.top_voice_id
            == confirmed_voice_id
        ),
        top_score=summary.top_score,
        runner_up_score=summary.runner_up_score,
        margin=summary.margin,
        encoder_count=summary.encoder_count,
        embedding_scores=dict(
            summary.embedding_scores
        ),
        support=dict(
            summary.support
        ),
        source_support=dict(
            summary.source_support
        ),
        total_source_support=(
            summary.total_source_support
        ),
    )


def summarize_calibration_observations(
    observations: list[
        SpeakerCalibrationObservation
    ],
) -> SpeakerCalibrationStats:
    prediction_count = sum(
        observation.predicted_voice_id is not None
        for observation in observations
    )

    correct_count = sum(
        observation.correct
        for observation in observations
        if observation.predicted_voice_id is not None
    )

    top1_accuracy = (
        correct_count / prediction_count
        if prediction_count > 0
        else None
    )

    return SpeakerCalibrationStats(
        observation_count=len(observations),
        prediction_count=prediction_count,
        correct_count=correct_count,
        top1_accuracy=top1_accuracy,
    )


def filter_calibration_observations(
    observations: list[
        SpeakerCalibrationObservation
    ],
    *,
    minimum_margin: float | None = None,
    minimum_source_support: int | None = None,
    minimum_encoder_count: int | None = None,
) -> list[SpeakerCalibrationObservation]:
    results = []

    for observation in observations:
        if observation.predicted_voice_id is None:
            continue

        if (
            minimum_margin is not None
            and (
                observation.margin is None
                or observation.margin
                < minimum_margin
            )
        ):
            continue

        if (
            minimum_source_support is not None
            and observation.total_source_support
            < minimum_source_support
        ):
            continue

        if (
            minimum_encoder_count is not None
            and observation.encoder_count
            < minimum_encoder_count
        ):
            continue

        results.append(observation)

    return results


def evaluate_calibration_rule(
    observations: list[
        SpeakerCalibrationObservation
    ],
    *,
    minimum_margin: float | None = None,
    minimum_source_support: int | None = None,
    minimum_encoder_count: int | None = None,
) -> SpeakerCalibrationStats:
    filtered = filter_calibration_observations(
        observations,
        minimum_margin=minimum_margin,
        minimum_source_support=minimum_source_support,
        minimum_encoder_count=minimum_encoder_count,
    )

    return summarize_calibration_observations(
        filtered
    )

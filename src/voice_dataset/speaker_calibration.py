from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .speaker_candidates import SpeakerCandidate
from .storage import DatasetStorage


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


@dataclass(frozen=True)
class SpeakerCalibrationEvaluation:
    total_count: int
    eligible_count: int
    correct_count: int
    error_count: int
    precision: float | None
    coverage: float | None


@dataclass(frozen=True)
class SpeakerCalibrationRule:
    minimum_margin: float | None = None
    minimum_reference_support: int | None = None
    minimum_source_support: int | None = None
    minimum_encoder_count: int | None = None

    minimum_precision: float = 0.0
    minimum_eligible_count: int = 1


@dataclass(frozen=True)
class SpeakerCalibrationReportRow:
    rule: SpeakerCalibrationRule
    evaluation: SpeakerCalibrationEvaluation


SpeakerReviewMode = Literal[
    "none",
    "suggest",
    "prefill",
]


SPEAKER_CALIBRATION_SCHEMA_VERSION = 1


def speaker_calibration_observation_to_dict(
    observation: SpeakerCalibrationObservation,
) -> dict:
    return {
        "schema_version":
            SPEAKER_CALIBRATION_SCHEMA_VERSION,
        "turn_id": observation.turn_id,
        "source_id": observation.source_id,
        "confirmed_voice_id":
            observation.confirmed_voice_id,
        "predicted_voice_id":
            observation.predicted_voice_id,
        "correct": observation.correct,
        "top_score": observation.top_score,
        "runner_up_score":
            observation.runner_up_score,
        "margin": observation.margin,
        "encoder_count":
            observation.encoder_count,
        "embedding_scores": dict(
            observation.embedding_scores
        ),
        "support": dict(
            observation.support
        ),
        "source_support": dict(
            observation.source_support
        ),
        "total_source_support":
            observation.total_source_support,
    }


def _require_string(
    record: dict[str, Any],
    key: str,
) -> str:
    value = record.get(key)

    if not isinstance(value, str) or not value:
        raise ValueError(
            f"speaker calibration has invalid {key}"
        )

    return value


def _optional_string(
    record: dict[str, Any],
    key: str,
) -> str | None:
    value = record.get(key)

    if value is None:
        return None

    if not isinstance(value, str) or not value:
        raise ValueError(
            f"speaker calibration has invalid {key}"
        )

    return value


def _optional_float(
    record: dict[str, Any],
    key: str,
) -> float | None:
    value = record.get(key)

    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ValueError(
            f"speaker calibration has invalid {key}"
        )

    return float(value)


def _require_non_negative_int(
    record: dict[str, Any],
    key: str,
) -> int:
    value = record.get(key)

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"speaker calibration has invalid {key}"
        )

    return value


def _float_mapping(
    record: dict[str, Any],
    key: str,
) -> dict[str, float]:
    value = record.get(key)

    if not isinstance(value, dict):
        raise ValueError(
            f"speaker calibration has invalid {key}"
        )

    result: dict[str, float] = {}

    for name, score in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"speaker calibration has invalid {key}"
            )

        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
        ):
            raise ValueError(
                f"speaker calibration has invalid {key}"
            )

        result[name] = float(score)

    return result


def _int_mapping(
    record: dict[str, Any],
    key: str,
) -> dict[str, int]:
    value = record.get(key)

    if not isinstance(value, dict):
        raise ValueError(
            f"speaker calibration has invalid {key}"
        )

    result: dict[str, int] = {}

    for name, count in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"speaker calibration has invalid {key}"
            )

        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise ValueError(
                f"speaker calibration has invalid {key}"
            )

        result[name] = count

    return result


def speaker_calibration_observation_from_dict(
    record: dict[str, Any],
) -> SpeakerCalibrationObservation:
    if not isinstance(record, dict):
        raise ValueError(
            "speaker calibration must be an object"
        )

    schema_version = record.get(
        "schema_version"
    )

    if (
        schema_version
        != SPEAKER_CALIBRATION_SCHEMA_VERSION
    ):
        raise ValueError(
            "unsupported speaker calibration "
            f"schema version: {schema_version!r}"
        )

    turn_id = _require_string(
        record,
        "turn_id",
    )
    source_id = _require_string(
        record,
        "source_id",
    )
    confirmed_voice_id = _require_string(
        record,
        "confirmed_voice_id",
    )
    predicted_voice_id = _optional_string(
        record,
        "predicted_voice_id",
    )

    correct = record.get("correct")

    if not isinstance(correct, bool):
        raise ValueError(
            "speaker calibration has invalid correct"
        )

    expected_correct = (
        predicted_voice_id
        == confirmed_voice_id
    )

    if correct != expected_correct:
        raise ValueError(
            "speaker calibration correct flag "
            "does not match voice IDs"
        )

    observation = SpeakerCalibrationObservation(
        turn_id=turn_id,
        source_id=source_id,
        confirmed_voice_id=confirmed_voice_id,
        predicted_voice_id=predicted_voice_id,
        correct=correct,
        top_score=_optional_float(
            record,
            "top_score",
        ),
        runner_up_score=_optional_float(
            record,
            "runner_up_score",
        ),
        margin=_optional_float(
            record,
            "margin",
        ),
        encoder_count=_require_non_negative_int(
            record,
            "encoder_count",
        ),
        embedding_scores=_float_mapping(
            record,
            "embedding_scores",
        ),
        support=_int_mapping(
            record,
            "support",
        ),
        source_support=_int_mapping(
            record,
            "source_support",
        ),
        total_source_support=(
            _require_non_negative_int(
                record,
                "total_source_support",
            )
        ),
    )

    if (
        observation.predicted_voice_id is None
        and observation.top_score is not None
    ):
        raise ValueError(
            "speaker calibration without prediction "
            "must not have top_score"
        )

    if (
        observation.margin is not None
        and (
            observation.top_score is None
            or observation.runner_up_score is None
        )
    ):
        raise ValueError(
            "speaker calibration margin requires "
            "top and runner-up scores"
        )

    return observation


def load_calibration_observations(
    storage: DatasetStorage,
) -> list[SpeakerCalibrationObservation]:
    observations: list[
        SpeakerCalibrationObservation
    ] = []

    for turn in storage.turns.load():
        review = turn.get("review") or {}

        if not isinstance(review, dict):
            raise ValueError(
                f"{turn.get('id', '<unknown>')}: "
                "review must be an object"
            )

        if review.get("status") != "reviewed":
            continue

        calibration = review.get(
            "speaker_calibration"
        )

        if calibration is None:
            continue

        if not isinstance(calibration, dict):
            raise ValueError(
                f"{turn.get('id', '<unknown>')}: "
                "speaker calibration must be an object"
            )

        observation = (
            speaker_calibration_observation_from_dict(
                calibration
            )
        )

        turn_id = turn.get("id")
        source_id = turn.get("source_id")

        if observation.turn_id != turn_id:
            raise ValueError(
                f"{turn_id}: speaker calibration "
                "turn_id does not match turn"
            )

        if observation.source_id != source_id:
            raise ValueError(
                f"{turn_id}: speaker calibration "
                "source_id does not match turn"
            )

        assignment = turn.get("assignment") or {}

        if not isinstance(assignment, dict):
            raise ValueError(
                f"{turn_id}: assignment must be an object"
            )

        if assignment.get("status") != "assigned":
            raise ValueError(
                f"{turn_id}: reviewed speaker calibration "
                "requires assigned voice"
            )

        if assignment.get("method") != "manual":
            raise ValueError(
                f"{turn_id}: reviewed speaker calibration "
                "requires manual assignment"
            )

        if (
            assignment.get("voice_id")
            != observation.confirmed_voice_id
        ):
            raise ValueError(
                f"{turn_id}: speaker calibration "
                "confirmed voice does not match assignment"
            )

        observations.append(observation)

    return observations


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


def minimum_support_count(
    support: dict[str, int],
) -> int:
    if not support:
        return 0

    return min(support.values())


def speaker_summary_matches_rule(
    summary: SpeakerCandidateSummary,
    rule: SpeakerCalibrationRule,
) -> bool:
    if summary.top_voice_id is None:
        return False

    if (
        rule.minimum_margin is not None
        and (
            summary.margin is None
            or summary.margin
            < rule.minimum_margin
        )
    ):
        return False

    if (
        rule.minimum_reference_support is not None
        and minimum_support_count(
            summary.support
        ) < rule.minimum_reference_support
    ):
        return False

    if (
        rule.minimum_source_support is not None
        and summary.total_source_support
        < rule.minimum_source_support
    ):
        return False

    if (
        rule.minimum_encoder_count is not None
        and summary.encoder_count
        < rule.minimum_encoder_count
    ):
        return False

    return True


def calibration_rule_is_trusted(
    evaluation: SpeakerCalibrationEvaluation,
    rule: SpeakerCalibrationRule,
) -> bool:
    if (
        evaluation.eligible_count
        < rule.minimum_eligible_count
    ):
        return False

    if evaluation.precision is None:
        return False

    return (
        evaluation.precision
        >= rule.minimum_precision
    )


def filter_calibration_observations(
    observations: list[
        SpeakerCalibrationObservation
    ],
    *,
    minimum_margin: float | None = None,
    minimum_reference_support: int | None = None,
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
            minimum_reference_support is not None
            and minimum_support_count(
                observation.support
            ) < minimum_reference_support
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
    minimum_reference_support: int | None = None,
    minimum_source_support: int | None = None,
    minimum_encoder_count: int | None = None,
) -> SpeakerCalibrationEvaluation:
    eligible = filter_calibration_observations(
        observations,
        minimum_margin=minimum_margin,
        minimum_reference_support=(
            minimum_reference_support
        ),
        minimum_source_support=minimum_source_support,
        minimum_encoder_count=minimum_encoder_count,
    )

    total_count = len(observations)
    eligible_count = len(eligible)

    correct_count = sum(
        observation.correct
        for observation in eligible
    )

    error_count = (
        eligible_count - correct_count
    )

    precision = (
        correct_count / eligible_count
        if eligible_count > 0
        else None
    )

    coverage = (
        eligible_count / total_count
        if total_count > 0
        else None
    )

    return SpeakerCalibrationEvaluation(
        total_count=total_count,
        eligible_count=eligible_count,
        correct_count=correct_count,
        error_count=error_count,
        precision=precision,
        coverage=coverage,
    )


def evaluate_speaker_calibration_rule(
    observations: list[
        SpeakerCalibrationObservation
    ],
    rule: SpeakerCalibrationRule,
) -> SpeakerCalibrationEvaluation:
    return evaluate_calibration_rule(
        observations,
        minimum_margin=rule.minimum_margin,
        minimum_reference_support=(
            rule.minimum_reference_support
        ),
        minimum_source_support=(
            rule.minimum_source_support
        ),
        minimum_encoder_count=(
            rule.minimum_encoder_count
        ),
    )


def classify_speaker_review_mode(
    *,
    candidates: list[SpeakerCandidate],
    observations: list[
        SpeakerCalibrationObservation
    ],
    suggest_rule: SpeakerCalibrationRule,
    prefill_rule: SpeakerCalibrationRule,
) -> SpeakerReviewMode:
    summary = summarize_speaker_candidates(
        candidates
    )

    prefill_evaluation = (
        evaluate_speaker_calibration_rule(
            observations,
            prefill_rule,
        )
    )

    if (
        speaker_summary_matches_rule(
            summary,
            prefill_rule,
        )
        and calibration_rule_is_trusted(
            prefill_evaluation,
            prefill_rule,
        )
    ):
        return "prefill"

    suggest_evaluation = (
        evaluate_speaker_calibration_rule(
            observations,
            suggest_rule,
        )
    )

    if (
        speaker_summary_matches_rule(
            summary,
            suggest_rule,
        )
        and calibration_rule_is_trusted(
            suggest_evaluation,
            suggest_rule,
        )
    ):
        return "suggest"

    return "none"


def build_speaker_calibration_report(
    observations: list[
        SpeakerCalibrationObservation
    ],
    *,
    margins: tuple[float, ...] = (
        0.0,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
    ),
    reference_support_values: tuple[int, ...] = (
        1,
        2,
        4,
    ),
    source_support_values: tuple[int, ...] = (
        1,
        2,
        3,
    ),
    encoder_counts: tuple[int, ...] = (
        1,
        2,
    ),
) -> list[SpeakerCalibrationReportRow]:
    rows: list[
        SpeakerCalibrationReportRow
    ] = []

    for margin in margins:
        for reference_support in reference_support_values:
            for source_support in source_support_values:
                for encoder_count in encoder_counts:
                    rule = SpeakerCalibrationRule(
                        minimum_margin=margin,
                        minimum_reference_support=(
                            reference_support
                        ),
                        minimum_source_support=(
                            source_support
                        ),
                        minimum_encoder_count=(
                            encoder_count
                        ),
                    )

                    evaluation = (
                        evaluate_speaker_calibration_rule(
                            observations,
                            rule,
                        )
                    )

                    rows.append(
                        SpeakerCalibrationReportRow(
                            rule=rule,
                            evaluation=evaluation,
                        )
                    )

    rows.sort(
        key=lambda row: (
            -(
                row.evaluation.precision
                if row.evaluation.precision
                is not None
                else -1.0
            ),
            -(
                row.evaluation.coverage
                if row.evaluation.coverage
                is not None
                else -1.0
            ),
            -row.evaluation.eligible_count,
            row.rule.minimum_margin
            if row.rule.minimum_margin
            is not None
            else 0.0,
            row.rule.minimum_reference_support
            if row.rule.minimum_reference_support
            is not None
            else 0,
            row.rule.minimum_source_support
            if row.rule.minimum_source_support
            is not None
            else 0,
            row.rule.minimum_encoder_count
            if row.rule.minimum_encoder_count
            is not None
            else 0,
        )
    )

    return rows

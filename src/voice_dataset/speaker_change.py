from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .embeddings import embed_turns
from .representations import (
    materialize_turn_audio,
)
from .schema import TurnRecord
from .sources import (
    resolve_source_representation_for_purpose,
)
from .storage import DatasetStorage


DEFAULT_SPEAKER_CHANGE_WINDOW = 1.0
DEFAULT_SPEAKER_CHANGE_STEP = 0.25
DEFAULT_SPEAKER_CHANGE_MIN_WINDOW = 0.5
DEFAULT_SPEAKER_CHANGE_MAX_SHIFT = 0.75
DEFAULT_SPEAKER_CHANGE_SHIFT_STEP = 0.05
DEFAULT_SPEAKER_CHANGE_MIN_SPEECH_RATIO = 0.5

DEFAULT_SPEAKER_CHANGE_FULL_THRESHOLD = 0.92
DEFAULT_SPEAKER_CHANGE_ADAPTIVE_THRESHOLD = 0.92


@dataclass(frozen=True)
class SpeakerChangeBoundary:
    turn_id: str
    boundary: float

    left_start: float
    left_end: float

    right_start: float
    right_end: float


@dataclass(frozen=True)
class SpeakerEmbeddingChange:
    turn_id: str
    boundary: float

    ecapa_similarity: float
    wespeaker_similarity: float

    mean_similarity: float
    encoder_disagreement: float
    divergence: float


@dataclass(frozen=True)
class TurnSpeakerChangeEvidence:
    turn_id: str
    changes: tuple[
        SpeakerEmbeddingChange,
        ...,
    ]


@dataclass(frozen=True)
class _AnalysisPair:
    boundary: SpeakerChangeBoundary
    left_window_id: str
    right_window_id: str


def _speech_coverage(
    words: list[dict[str, Any]],
    start: float,
    end: float,
) -> float:
    total = 0.0

    for word in words:
        word_start = float(word["start"])
        word_end = float(word["end"])

        total += max(
            0.0,
            min(end, word_end)
            - max(start, word_start),
        )

    return total


def _best_speech_window(
    words: list[dict[str, Any]],
    *,
    boundary: float,
    range_start: float,
    range_end: float,
    side: str,
    target_window: float,
    minimum_window: float,
    max_shift: float,
    shift_step: float,
    minimum_speech_ratio: float,
) -> tuple[float, float] | None:
    if side not in {"left", "right"}:
        raise ValueError(
            f"Unsupported speaker-change side: {side}"
        )

    if side == "left":
        available = boundary - range_start
    else:
        available = range_end - boundary

    duration = min(
        target_window,
        available,
    )

    if duration < minimum_window:
        return None

    best: tuple[
        float,
        float,
        float,
        float,
    ] | None = None

    shift = 0.0

    while shift <= max_shift + 1e-9:
        if side == "left":
            end = boundary - shift
            start = end - duration

            if start < range_start:
                shift += shift_step
                continue

        else:
            start = boundary + shift
            end = start + duration

            if end > range_end:
                shift += shift_step
                continue

        coverage = _speech_coverage(
            words,
            start,
            end,
        )

        ratio = (
            coverage / duration
            if duration > 0
            else 0.0
        )

        candidate = (
            coverage,
            -shift,
            start,
            end,
        )

        if (
            best is None
            or candidate > best
        ):
            best = candidate

        shift += shift_step

    if best is None:
        return None

    coverage, _, start, end = best

    if (
        coverage / (end - start)
        < minimum_speech_ratio
    ):
        return None

    return (
        start,
        end,
    )


def plan_turn_speaker_change_boundaries(
    turn: dict[str, Any],
    *,
    words: list[dict[str, Any]] | None = None,
    window: float = DEFAULT_SPEAKER_CHANGE_WINDOW,
    step: float = DEFAULT_SPEAKER_CHANGE_STEP,
    minimum_window: float = (
        DEFAULT_SPEAKER_CHANGE_MIN_WINDOW
    ),
    max_shift: float = (
        DEFAULT_SPEAKER_CHANGE_MAX_SHIFT
    ),
    shift_step: float = (
        DEFAULT_SPEAKER_CHANGE_SHIFT_STEP
    ),
    minimum_speech_ratio: float = (
        DEFAULT_SPEAKER_CHANGE_MIN_SPEECH_RATIO
    ),
) -> tuple[SpeakerChangeBoundary, ...]:
    if window <= 0:
        raise ValueError(
            "window must be positive"
        )

    if step <= 0:
        raise ValueError(
            "step must be positive"
        )

    if minimum_window <= 0:
        raise ValueError(
            "minimum_window must be positive"
        )

    if minimum_window > window:
        raise ValueError(
            "minimum_window must not exceed window"
        )

    if max_shift < 0:
        raise ValueError(
            "max_shift must not be negative"
        )

    if shift_step <= 0:
        raise ValueError(
            "shift_step must be positive"
        )

    if not (
        0.0
        <= minimum_speech_ratio
        <= 1.0
    ):
        raise ValueError(
            "minimum_speech_ratio must be between "
            "zero and one"
        )

    turn_id = turn.get("id")

    if (
        not isinstance(turn_id, str)
        or not turn_id
    ):
        raise ValueError(
            "Turn has invalid id"
        )

    start = turn.get("source_start")
    end = turn.get("source_end")

    if not isinstance(
        start,
        (int, float),
    ):
        raise ValueError(
            f"{turn_id}: invalid source_start"
        )

    if not isinstance(
        end,
        (int, float),
    ):
        raise ValueError(
            f"{turn_id}: invalid source_end"
        )

    start = float(start)
    end = float(end)

    if start < 0 or end <= start:
        raise ValueError(
            f"{turn_id}: invalid source range"
        )

    duration = end - start

    if words is None:
        # Without speech timing evidence we cannot
        # safely shorten or reposition the windows.
        if duration < 2.0 * window:
            return ()

        first = start + window
        last = end - window

    elif duration >= 2.0 * window:
        # Long turns have enough room for full target
        # windows on both sides.
        first = start + window
        last = end - window

    elif duration >= 2.0 * minimum_window:
        # Short turns cannot support full target
        # windows on both sides. With speech timing
        # evidence available, analyze exactly one
        # central boundary using adaptive flanks.
        first = (
            start + end
        ) / 2.0
        last = first

    else:
        return ()

    boundaries = []

    boundary = first

    while boundary <= last + 1e-9:
        boundary = min(
            boundary,
            last,
        )

        if words is None:
            left_start = (
                boundary - window
            )
            left_end = boundary
            right_start = boundary
            right_end = (
                boundary + window
            )

        else:
            left = _best_speech_window(
                words,
                boundary=boundary,
                range_start=start,
                range_end=end,
                side="left",
                target_window=window,
                minimum_window=minimum_window,
                max_shift=max_shift,
                shift_step=shift_step,
                minimum_speech_ratio=(
                    minimum_speech_ratio
                ),
            )

            right = _best_speech_window(
                words,
                boundary=boundary,
                range_start=start,
                range_end=end,
                side="right",
                target_window=window,
                minimum_window=minimum_window,
                max_shift=max_shift,
                shift_step=shift_step,
                minimum_speech_ratio=(
                    minimum_speech_ratio
                ),
            )

            if left is None or right is None:
                if boundary >= last:
                    break

                boundary += step
                continue

            left_start, left_end = left
            right_start, right_end = right

        boundaries.append(
            SpeakerChangeBoundary(
                turn_id=turn_id,
                boundary=boundary,
                left_start=left_start,
                left_end=left_end,
                right_start=right_start,
                right_end=right_end,
            )
        )

        if boundary >= last:
            break

        boundary += step

    return tuple(boundaries)


def speaker_change_window_kind(
    boundary: SpeakerChangeBoundary,
) -> str:
    left_duration = (
        boundary.left_end
        - boundary.left_start
    )
    right_duration = (
        boundary.right_end
        - boundary.right_start
    )

    if (
        left_duration
        >= DEFAULT_SPEAKER_CHANGE_WINDOW - 1e-6
        and right_duration
        >= DEFAULT_SPEAKER_CHANGE_WINDOW - 1e-6
    ):
        return "full"

    return "adaptive"


def is_suspected_speaker_change(
    change: SpeakerEmbeddingChange,
    boundary: SpeakerChangeBoundary,
    *,
    full_threshold: float = (
        DEFAULT_SPEAKER_CHANGE_FULL_THRESHOLD
    ),
    adaptive_threshold: float = (
        DEFAULT_SPEAKER_CHANGE_ADAPTIVE_THRESHOLD
    ),
) -> bool:
    kind = speaker_change_window_kind(
        boundary
    )

    threshold = (
        full_threshold
        if kind == "full"
        else adaptive_threshold
    )

    return (
        change.divergence
        >= threshold
    )


def cosine_similarity(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    left = np.asarray(
        left,
        dtype=np.float32,
    ).reshape(-1)

    right = np.asarray(
        right,
        dtype=np.float32,
    ).reshape(-1)

    if left.shape != right.shape:
        raise ValueError(
            "Embedding dimensions do not match"
        )

    if left.size == 0:
        raise ValueError(
            "Embedding must not be empty"
        )

    if (
        not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError(
            "Embedding contains non-finite values"
        )

    left_norm = float(
        np.linalg.norm(left)
    )
    right_norm = float(
        np.linalg.norm(right)
    )

    if left_norm == 0 or right_norm == 0:
        raise ValueError(
            "Embedding has zero norm"
        )

    return float(
        np.dot(left, right)
        / (left_norm * right_norm)
    )


def score_speaker_change(
    boundary: SpeakerChangeBoundary,
    *,
    ecapa_left: np.ndarray,
    ecapa_right: np.ndarray,
    wespeaker_left: np.ndarray,
    wespeaker_right: np.ndarray,
) -> SpeakerEmbeddingChange:
    ecapa_similarity = cosine_similarity(
        ecapa_left,
        ecapa_right,
    )

    wespeaker_similarity = cosine_similarity(
        wespeaker_left,
        wespeaker_right,
    )

    mean_similarity = (
        ecapa_similarity
        + wespeaker_similarity
    ) / 2.0

    encoder_disagreement = abs(
        ecapa_similarity
        - wespeaker_similarity
    )

    return SpeakerEmbeddingChange(
        turn_id=boundary.turn_id,
        boundary=boundary.boundary,
        ecapa_similarity=ecapa_similarity,
        wespeaker_similarity=(
            wespeaker_similarity
        ),
        mean_similarity=mean_similarity,
        encoder_disagreement=(
            encoder_disagreement
        ),
        divergence=1.0 - mean_similarity,
    )


def _embedding_vector(
    storage: DatasetStorage,
    turn_id: str,
    name: str,
) -> np.ndarray:
    turn = storage.get_turn(turn_id)

    if turn is None:
        raise KeyError(
            f"Analysis window does not exist: "
            f"{turn_id}"
        )

    embeddings = turn.get(
        "embeddings"
    )

    if not isinstance(embeddings, dict):
        raise ValueError(
            f"{turn_id}: invalid embeddings"
        )

    reference = embeddings.get(name)

    if not isinstance(reference, dict):
        raise ValueError(
            f"{turn_id}: embedding does not exist: "
            f"{name}"
        )

    path_value = reference.get("path")

    if (
        not isinstance(path_value, str)
        or not path_value
    ):
        raise ValueError(
            f"{turn_id}/{name}: invalid path"
        )

    path = storage.root / Path(
        path_value
    )

    return np.load(
        path,
        allow_pickle=False,
    )


def _prepare_analysis_pairs(
    storage: DatasetStorage,
    source: Path,
    *,
    source_kind: str,
    boundaries: list[
        SpeakerChangeBoundary
    ],
) -> list[_AnalysisPair]:
    window_ids: dict[
        tuple[float, float],
        str,
    ] = {}

    pairs: list[_AnalysisPair] = []

    def window_id(
        start: float,
        end: float,
    ) -> str:
        key = (
            round(start, 9),
            round(end, 9),
        )

        existing = window_ids.get(key)

        if existing is not None:
            return existing

        record_id = (
            f"window_{len(window_ids) + 1:06d}"
        )

        storage.add_turn(
            TurnRecord(
                id=record_id,
                source_id="speaker-change-analysis",
                source_start=start,
                source_end=end,
            )
        )

        materialize_turn_audio(
            storage=storage,
            turn_id=record_id,
            source=source,
            representation_name=(
                "speaker_change"
            ),
            kind=source_kind,
            purposes=[
                "speaker_embedding",
            ],
        )

        window_ids[key] = record_id

        return record_id

    for boundary in boundaries:
        left_id = window_id(
            boundary.left_start,
            boundary.left_end,
        )

        right_id = window_id(
            boundary.right_start,
            boundary.right_end,
        )

        pairs.append(
            _AnalysisPair(
                boundary=boundary,
                left_window_id=left_id,
                right_window_id=right_id,
            )
        )

    return pairs


def analyze_speaker_change_boundaries(
    storage: DatasetStorage,
    source_id: str,
    boundaries: list[SpeakerChangeBoundary],
    *,
    capture_output: bool = True,
) -> tuple[SpeakerEmbeddingChange, ...]:
    if not boundaries:
        return ()

    (
        _,
        source_representation,
        source_path,
    ) = (
        resolve_source_representation_for_purpose(
            storage,
            source_id,
            "speaker_embedding",
        )
    )

    source_kind = source_representation.get(
        "kind"
    )

    if (
        not isinstance(source_kind, str)
        or not source_kind
    ):
        raise ValueError(
            "Speaker embedding source "
            "representation has invalid kind"
        )

    with tempfile.TemporaryDirectory(
        prefix="voice-dataset-speaker-change-"
    ) as temporary_directory:
        analysis_storage = DatasetStorage(
            Path(temporary_directory)
            / "dataset"
        )

        pairs = _prepare_analysis_pairs(
            analysis_storage,
            source_path,
            source_kind=source_kind,
            boundaries=boundaries,
        )

        embed_turns(
            analysis_storage,
            encoder="ecapa",
            representation="speaker_change",
            name="ecapa_speaker_change",
            capture_output=capture_output,
        )

        embed_turns(
            analysis_storage,
            encoder="wespeaker",
            representation="speaker_change",
            name="wespeaker_speaker_change",
            capture_output=capture_output,
        )

        changes = []

        for pair in pairs:
            changes.append(
                score_speaker_change(
                    pair.boundary,
                    ecapa_left=_embedding_vector(
                        analysis_storage,
                        pair.left_window_id,
                        "ecapa_speaker_change",
                    ),
                    ecapa_right=_embedding_vector(
                        analysis_storage,
                        pair.right_window_id,
                        "ecapa_speaker_change",
                    ),
                    wespeaker_left=_embedding_vector(
                        analysis_storage,
                        pair.left_window_id,
                        "wespeaker_speaker_change",
                    ),
                    wespeaker_right=_embedding_vector(
                        analysis_storage,
                        pair.right_window_id,
                        "wespeaker_speaker_change",
                    ),
                )
            )

    return tuple(changes)


def analyze_source_speaker_changes(
    storage: DatasetStorage,
    source_id: str,
    *,
    turn_ids: set[str] | None = None,
    window: float = DEFAULT_SPEAKER_CHANGE_WINDOW,
    step: float = DEFAULT_SPEAKER_CHANGE_STEP,
    capture_output: bool = True,
) -> tuple[
    TurnSpeakerChangeEvidence,
    ...,
]:
    turns = [
        turn
        for turn in storage.turns.load()
        if (
            turn.get("source_id")
            == source_id
            and (
                turn_ids is None
                or turn.get("id")
                in turn_ids
            )
        )
    ]

    turns.sort(
        key=lambda turn: (
            float(turn["source_start"]),
            float(turn["source_end"]),
            str(turn["id"]),
        )
    )

    source = storage.get_source(
        source_id
    )

    if source is None:
        raise KeyError(
            f"Source does not exist: {source_id}"
        )

    metadata = source.get(
        "metadata",
        {},
    )

    continuous_asr = (
        metadata.get("continuous_asr", {})
        if isinstance(metadata, dict)
        else {}
    )

    qwen = (
        continuous_asr.get("qwen3")
        if isinstance(
            continuous_asr,
            dict,
        )
        else None
    )

    words = (
        qwen.get("words")
        if isinstance(qwen, dict)
        else None
    )

    if not isinstance(words, list):
        raise ValueError(
            "Source has no valid qwen3 "
            "continuous ASR words"
        )

    boundaries = []

    for turn in turns:
        boundaries.extend(
            plan_turn_speaker_change_boundaries(
                turn,
                words=words,
                window=window,
                step=step,
            )
        )

    if not boundaries:
        return ()

    changes = analyze_speaker_change_boundaries(
        storage,
        source_id,
        boundaries,
        capture_output=capture_output,
    )

    changes_by_turn: dict[
        str,
        list[SpeakerEmbeddingChange],
    ] = {}

    for change in changes:
        changes_by_turn.setdefault(
            change.turn_id,
            [],
        ).append(change)

    return tuple(
        TurnSpeakerChangeEvidence(
            turn_id=turn["id"],
            changes=tuple(
                changes_by_turn.get(
                    turn["id"],
                    [],
                )
            ),
        )
        for turn in turns
        if turn["id"] in changes_by_turn
    )

@dataclass(frozen=True)
class PersistSpeakerChangeEvidenceResult:
    analyzed_turns: int
    comparisons: int
    suspected_turns: int


def _speaker_change_evidence_record(
    boundary: SpeakerChangeBoundary,
    change: SpeakerEmbeddingChange,
) -> dict[str, Any]:
    return {
        "boundary": change.boundary,
        "window_kind": (
            speaker_change_window_kind(
                boundary
            )
        ),
        "left_start": boundary.left_start,
        "left_end": boundary.left_end,
        "right_start": boundary.right_start,
        "right_end": boundary.right_end,
        "ecapa_similarity": (
            change.ecapa_similarity
        ),
        "wespeaker_similarity": (
            change.wespeaker_similarity
        ),
        "mean_similarity": (
            change.mean_similarity
        ),
        "encoder_disagreement": (
            change.encoder_disagreement
        ),
        "divergence": change.divergence,
        "suspected": (
            is_suspected_speaker_change(
                change,
                boundary,
            )
        ),
    }


def apply_source_speaker_change_evidence(
    storage: DatasetStorage,
    source_id: str,
    boundaries: list[SpeakerChangeBoundary],
    changes: tuple[
        SpeakerEmbeddingChange,
        ...,
    ],
    *,
    step: float,
    turn_ids: set[str] | None = None,
) -> PersistSpeakerChangeEvidenceResult:
    if len(boundaries) != len(changes):
        raise ValueError(
            "Speaker-change boundaries and "
            "changes must have equal length"
        )

    boundaries_by_key = {
        (
            boundary.turn_id,
            round(
                boundary.boundary,
                9,
            ),
        ): boundary
        for boundary in boundaries
    }

    if (
        len(boundaries_by_key)
        != len(boundaries)
    ):
        raise ValueError(
            "Duplicate speaker-change boundary"
        )

    changes_by_turn: dict[
        str,
        list[
            tuple[
                SpeakerChangeBoundary,
                SpeakerEmbeddingChange,
            ]
        ],
    ] = {}

    for change in changes:
        key = (
            change.turn_id,
            round(
                change.boundary,
                9,
            ),
        )

        boundary = boundaries_by_key.get(
            key
        )

        if boundary is None:
            raise ValueError(
                "Speaker-change result has no "
                "matching boundary: "
                f"{change.turn_id}/"
                f"{change.boundary}"
            )

        changes_by_turn.setdefault(
            change.turn_id,
            [],
        ).append(
            (
                boundary,
                change,
            )
        )

    turns = [
        turn
        for turn in storage.turns.load()
        if (
            turn.get("source_id")
            == source_id
            and (
                turn_ids is None
                or turn.get("id")
                in turn_ids
            )
        )
    ]

    suspected_turns = 0

    for turn in turns:
        turn_id = str(turn["id"])

        pairs = changes_by_turn.get(
            turn_id,
            [],
        )

        records = [
            _speaker_change_evidence_record(
                boundary,
                change,
            )
            for boundary, change in pairs
        ]

        records.sort(
            key=lambda item: float(
                item["boundary"]
            )
        )

        suspected = any(
            bool(record["suspected"])
            for record in records
        )

        if suspected:
            suspected_turns += 1

        if records:
            status = (
                "suspected"
                if suspected
                else "clear"
            )

            strongest = max(
                records,
                key=lambda item: float(
                    item["divergence"]
                ),
            )
        else:
            status = (
                "insufficient_evidence"
            )
            strongest = None

        evidence = {
            "status": status,
            "step": step,
            "full_threshold": (
                DEFAULT_SPEAKER_CHANGE_FULL_THRESHOLD
            ),
            "adaptive_threshold": (
                DEFAULT_SPEAKER_CHANGE_ADAPTIVE_THRESHOLD
            ),
            "strongest": strongest,
            "changes": records,
        }

        def update(
            record: dict[str, Any],
            *,
            evidence=evidence,
            suspected=suspected,
        ) -> dict[str, Any]:
            metadata = dict(
                record.get("metadata")
                or {}
            )

            metadata[
                "speaker_change_evidence"
            ] = evidence

            existing_pipeline = (
                metadata.get(
                    "automatic_pipeline"
                )
            )

            has_pipeline = isinstance(
                existing_pipeline,
                dict,
            )

            pipeline = dict(
                existing_pipeline
                if has_pipeline
                else {}
            )

            raw_reasons = pipeline.get(
                "review_reasons"
            )

            reasons = (
                [
                    reason
                    for reason in raw_reasons
                    if isinstance(
                        reason,
                        str,
                    )
                ]
                if isinstance(
                    raw_reasons,
                    list,
                )
                else []
            )

            # This analysis owns this reason.
            reasons = [
                reason
                for reason in reasons
                if reason
                != "suspected_speaker_change"
            ]

            if suspected:
                reasons.append(
                    "suspected_speaker_change"
                )

            # Existing automatic turns keep their
            # automatic-pipeline state. A manually
            # created turn only gets one when there
            # is an actual automatic review reason.
            if has_pipeline or suspected:
                pipeline[
                    "review_reasons"
                ] = reasons

                pipeline["status"] = (
                    "review"
                    if reasons
                    else "ready"
                )

                metadata[
                    "automatic_pipeline"
                ] = pipeline

            record["metadata"] = metadata

            return record

        storage.update_turn(
            turn_id,
            update,
        )

    return PersistSpeakerChangeEvidenceResult(
        analyzed_turns=len(
            changes_by_turn
        ),
        comparisons=len(changes),
        suspected_turns=(
            suspected_turns
        ),
    )


def prepare_source_speaker_change_evidence(
    storage: DatasetStorage,
    source_id: str,
    *,
    evidence_name: str = "qwen3",
    step: float = 0.5,
    capture_output: bool = False,
) -> PersistSpeakerChangeEvidenceResult:
    source = storage.get_source(
        source_id
    )

    if source is None:
        raise KeyError(
            f"Source does not exist: "
            f"{source_id}"
        )

    metadata = source.get(
        "metadata"
    )

    if not isinstance(metadata, dict):
        raise ValueError(
            f"Source has invalid metadata: "
            f"{source_id}"
        )

    continuous_asr = metadata.get(
        "continuous_asr"
    )

    if not isinstance(
        continuous_asr,
        dict,
    ):
        raise ValueError(
            "Source has no continuous ASR metadata"
        )

    evidence = continuous_asr.get(
        evidence_name
    )

    if not isinstance(evidence, dict):
        raise ValueError(
            "Continuous ASR evidence does not exist: "
            f"{source_id}/{evidence_name}"
        )

    words = evidence.get("words")

    if not isinstance(words, list):
        raise ValueError(
            "Continuous ASR evidence has "
            "invalid words"
        )

    turns = [
        turn
        for turn in storage.turns.load()
        if (
            turn.get("source_id")
            == source_id
            and (
                (
                    turn.get("review")
                    or {}
                ).get(
                    "status",
                    "pending",
                )
                != "reviewed"
            )
            and (
                (
                    turn.get("curation")
                    or {}
                ).get(
                    "status",
                    "pending",
                )
                == "pending"
            )
        )
    ]

    turns.sort(
        key=lambda turn: (
            float(
                turn["source_start"]
            ),
            float(
                turn["source_end"]
            ),
            str(turn["id"]),
        )
    )

    boundaries: list[
        SpeakerChangeBoundary
    ] = []

    for turn in turns:
        boundaries.extend(
            plan_turn_speaker_change_boundaries(
                turn,
                words=words,
                step=step,
            )
        )

    changes = (
        analyze_speaker_change_boundaries(
            storage,
            source_id,
            boundaries,
            capture_output=(
                capture_output
            ),
        )
    )

    return apply_source_speaker_change_evidence(
        storage,
        source_id,
        boundaries,
        changes,
        step=step,
        turn_ids={
            str(turn["id"])
            for turn in turns
        },
    )

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


def plan_turn_speaker_change_boundaries(
    turn: dict[str, Any],
    *,
    window: float = DEFAULT_SPEAKER_CHANGE_WINDOW,
    step: float = DEFAULT_SPEAKER_CHANGE_STEP,
) -> tuple[SpeakerChangeBoundary, ...]:
    if window <= 0:
        raise ValueError(
            "window must be positive"
        )

    if step <= 0:
        raise ValueError(
            "step must be positive"
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

    first = start + window
    last = end - window

    if first > last:
        return ()

    boundaries = []

    boundary = first

    while boundary <= last + 1e-9:
        boundary = min(
            boundary,
            last,
        )

        boundaries.append(
            SpeakerChangeBoundary(
                turn_id=turn_id,
                boundary=boundary,
                left_start=boundary - window,
                left_end=boundary,
                right_start=boundary,
                right_end=boundary + window,
            )
        )

        if boundary >= last:
            break

        boundary += step

    return tuple(boundaries)


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

    boundaries = []

    for turn in turns:
        boundaries.extend(
            plan_turn_speaker_change_boundaries(
                turn,
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

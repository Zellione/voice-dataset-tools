from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .storage import DatasetStorage


@dataclass(frozen=True)
class SpeakerSimilarity:
    left_turn_id: str
    right_turn_id: str
    embedding_name: str
    encoder: str
    dimension: int
    similarity: float


def _embedding_reference(
    turn: dict[str, Any],
    embedding_name: str,
) -> dict[str, Any]:
    embeddings = turn.get("embeddings")

    if not isinstance(embeddings, dict):
        raise ValueError(
            f"{turn.get('id')}: embeddings must be an object"
        )

    reference = embeddings.get(embedding_name)

    if not isinstance(reference, dict):
        raise ValueError(
            f"{turn.get('id')}: embedding does not exist: "
            f"{embedding_name}"
        )

    return reference


def _embedding_path(
    storage: DatasetStorage,
    reference: dict[str, Any],
) -> Path:
    raw_path = reference.get("path")

    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(
            "Embedding reference has invalid path"
        )

    relative_path = Path(raw_path)

    if relative_path.is_absolute():
        raise ValueError(
            "Embedding path must be dataset-relative"
        )

    dataset_root = storage.root.resolve()
    path = (dataset_root / relative_path).resolve()

    try:
        path.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(
            f"Embedding path escapes the dataset: {raw_path}"
        ) from exc

    if not path.is_file():
        raise ValueError(
            f"Embedding file does not exist: {path}"
        )

    return path


def load_turn_embedding(
    storage: DatasetStorage,
    turn_id: str,
    embedding_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    turn = storage.get_turn(turn_id)

    if turn is None:
        raise KeyError(
            f"Unknown turn: {turn_id}"
        )

    reference = _embedding_reference(
        turn,
        embedding_name,
    )

    encoder = reference.get("encoder")

    if not isinstance(encoder, str) or not encoder:
        raise ValueError(
            f"{turn_id}: embedding has invalid encoder"
        )

    dimension = reference.get("dimension")

    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension <= 0
    ):
        raise ValueError(
            f"{turn_id}: embedding has invalid dimension"
        )

    path = _embedding_path(
        storage,
        reference,
    )

    try:
        vector = np.load(
            path,
            allow_pickle=False,
        )
    except Exception as exc:
        raise ValueError(
            f"{turn_id}: cannot load embedding: {exc}"
        ) from exc

    if vector.ndim != 1:
        raise ValueError(
            f"{turn_id}: embedding must be a 1-D vector"
        )

    if vector.shape[0] != dimension:
        raise ValueError(
            f"{turn_id}: embedding dimension does not "
            "match metadata"
        )

    if not np.issubdtype(
        vector.dtype,
        np.number,
    ):
        raise ValueError(
            f"{turn_id}: embedding must be numeric"
        )

    if not np.all(np.isfinite(vector)):
        raise ValueError(
            f"{turn_id}: embedding contains "
            "non-finite values"
        )

    return vector, reference


def compare_turn_embeddings(
    storage: DatasetStorage,
    left_turn_id: str,
    right_turn_id: str,
    embedding_name: str,
) -> SpeakerSimilarity:
    left, left_reference = load_turn_embedding(
        storage,
        left_turn_id,
        embedding_name,
    )

    right, right_reference = load_turn_embedding(
        storage,
        right_turn_id,
        embedding_name,
    )

    left_encoder = left_reference["encoder"]
    right_encoder = right_reference["encoder"]

    if left_encoder != right_encoder:
        raise ValueError(
            "Cannot compare embeddings from "
            "different encoders"
        )

    left_metadata = left_reference.get("metadata")
    right_metadata = right_reference.get("metadata")

    if not isinstance(left_metadata, dict):
        raise ValueError(
            f"{left_turn_id}: embedding has invalid metadata"
        )

    if not isinstance(right_metadata, dict):
        raise ValueError(
            f"{right_turn_id}: embedding has invalid metadata"
        )

    left_model = left_metadata.get("model")
    right_model = right_metadata.get("model")

    if left_model != right_model:
        raise ValueError(
            "Cannot compare embeddings from "
            "different models"
        )

    left_representation = left_reference.get(
        "representation"
    )
    right_representation = right_reference.get(
        "representation"
    )

    if left_representation != right_representation:
        raise ValueError(
            "Cannot compare embeddings from "
            "different representations"
        )

    left_dimension = left_reference["dimension"]
    right_dimension = right_reference["dimension"]

    if left_dimension != right_dimension:
        raise ValueError(
            "Cannot compare embeddings with "
            "different dimensions"
        )

    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))

    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError(
            "Cannot compare zero-norm embedding vectors"
        )

    similarity = float(
        np.dot(left, right)
        / (left_norm * right_norm)
    )

    return SpeakerSimilarity(
        left_turn_id=left_turn_id,
        right_turn_id=right_turn_id,
        embedding_name=embedding_name,
        encoder=left_encoder,
        dimension=left_dimension,
        similarity=similarity,
    )

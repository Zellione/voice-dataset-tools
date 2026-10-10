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


@dataclass(frozen=True)
class VoiceTurnMatch:
    turn_id: str
    similarity: float
    source_id: str | None = None


@dataclass(frozen=True)
class VoiceMatch:
    voice_id: str
    matches: tuple[VoiceTurnMatch, ...]


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


def _load_embedding_from_turn(
    storage: DatasetStorage,
    turn: dict[str, Any],
    embedding_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    turn_id = str(turn.get("id"))

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
            f"match metadata"
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
            f"non-finite values"
        )

    return vector, reference


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

    return _load_embedding_from_turn(
        storage,
        turn,
        embedding_name,
    )



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


def rank_voice_matches(
    storage: DatasetStorage,
    turn_id: str,
    embedding_name: str,
) -> list[VoiceMatch]:
    turns = storage.turns.load()

    query_turn = next(
        (
            turn
            for turn in turns
            if turn.get("id") == turn_id
        ),
        None,
    )

    if query_turn is None:
        raise KeyError(
            f"Unknown turn: {turn_id}"
        )

    query_vector, query_reference = (
        _load_embedding_from_turn(
            storage,
            query_turn,
            embedding_name,
        )
    )

    query_encoder = query_reference["encoder"]
    query_dimension = query_reference["dimension"]
    query_metadata = query_reference.get(
        "metadata"
    )
    query_representation = query_reference.get(
        "representation"
    )

    if not isinstance(query_metadata, dict):
        raise ValueError(
            f"{turn_id}: embedding has invalid metadata"
        )

    query_model = query_metadata.get("model")

    query_norm = float(
        np.linalg.norm(query_vector)
    )

    if query_norm == 0.0:
        raise ValueError(
            "Cannot compare zero-norm embedding vectors"
        )

    voices = {
        str(voice["id"]): voice
        for voice in storage.voices.load()
    }

    matches_by_voice: dict[
        str,
        list[VoiceTurnMatch],
    ] = {}

    for reference_turn in turns:
        reference_turn_id = reference_turn.get(
            "id"
        )

        if reference_turn_id == turn_id:
            continue

        assignment = reference_turn.get(
            "assignment"
        )

        if not isinstance(assignment, dict):
            continue

        if assignment.get("status") != "assigned":
            continue

        if assignment.get("method") != "manual":
            continue

        voice_id = assignment.get("voice_id")

        if not isinstance(voice_id, str) or not voice_id:
            continue

        voice = voices.get(voice_id)

        if voice is None:
            raise ValueError(
                f"{reference_turn_id}: assigned voice "
                f"does not exist: {voice_id}"
            )

        if voice.get("ignored") is True:
            continue

        embeddings = reference_turn.get(
            "embeddings"
        )

        if not isinstance(embeddings, dict):
            continue

        if embedding_name not in embeddings:
            continue

        reference_vector, reference = (
            _load_embedding_from_turn(
                storage,
                reference_turn,
                embedding_name,
            )
        )

        if reference["encoder"] != query_encoder:
            raise ValueError(
                "Cannot compare embeddings from "
                "different encoders"
            )

        if reference["dimension"] != query_dimension:
            raise ValueError(
                "Cannot compare embeddings with "
                "different dimensions"
            )

        reference_metadata = reference.get(
            "metadata"
        )

        if not isinstance(
            reference_metadata,
            dict,
        ):
            raise ValueError(
                f"{reference_turn_id}: embedding has "
                f"invalid metadata"
            )

        if (
            reference_metadata.get("model")
            != query_model
        ):
            raise ValueError(
                "Cannot compare embeddings from "
                "different models"
            )

        if (
            reference.get("representation")
            != query_representation
        ):
            raise ValueError(
                "Cannot compare embeddings from "
                "different representations"
            )

        reference_norm = float(
            np.linalg.norm(reference_vector)
        )

        if reference_norm == 0.0:
            raise ValueError(
                "Cannot compare zero-norm "
                "embedding vectors"
            )

        similarity = float(
            np.dot(
                query_vector,
                reference_vector,
            )
            / (
                query_norm
                * reference_norm
            )
        )

        source_id = reference_turn.get(
            "source_id"
        )

        if not isinstance(source_id, str):
            source_id = None

        matches_by_voice.setdefault(
            voice_id,
            [],
        ).append(
            VoiceTurnMatch(
                turn_id=str(
                    reference_turn_id
                ),
                similarity=similarity,
                source_id=source_id,
            )
        )

    results = []

    for voice_id, matches in matches_by_voice.items():
        matches.sort(
            key=lambda match: (
                -match.similarity,
                match.turn_id,
            )
        )

        results.append(
            VoiceMatch(
                voice_id=voice_id,
                matches=tuple(matches),
            )
        )

    results.sort(
        key=lambda match: match.voice_id
    )

    return results

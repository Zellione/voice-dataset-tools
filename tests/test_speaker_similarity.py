from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voice_dataset.speaker_similarity import (
    compare_turn_embeddings,
)
from voice_dataset.storage import DatasetStorage


def make_storage(tmp_path: Path) -> DatasetStorage:
    return DatasetStorage(tmp_path / "dataset")


def add_turn_with_embedding(
    storage: DatasetStorage,
    *,
    turn_id: str,
    vector: np.ndarray,
    encoder: str = "test-encoder",
    embedding_name: str = "speaker",
    model: str | None = "test-model",
    representation: str = "speech",
) -> None:
    relative_path = (
        Path("turns")
        / turn_id
        / "embeddings"
        / f"{embedding_name}.npy"
    )

    path = storage.root / relative_path
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(path, vector)

    storage.turns.append(
        {
            "schema_version": 2,
            "record_type": "turn",
            "id": turn_id,
            "source_id": "source_001",
            "source_start": 0.0,
            "source_end": 1.0,
            "source_regions": [],
            "language": None,
            "transcript": None,
            "representations": {},
            "embeddings": {
                embedding_name: {
                    "encoder": encoder,
                    "representation": representation,
                    "path": relative_path.as_posix(),
                    "dimension": int(vector.shape[0]),
                    "metadata": {
                        "model": model,
                    },
                },
            },
            "assignment": {
                "status": "unknown",
                "voice_id": None,
                "method": None,
                "confidence": None,
            },
            "review": {
                "status": "pending",
            },
            "metadata": {},
        }
    )


def test_compare_identical_embeddings(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    vector = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float32,
    )

    add_turn_with_embedding(
        storage,
        turn_id="turn_000001",
        vector=vector,
    )
    add_turn_with_embedding(
        storage,
        turn_id="turn_000002",
        vector=vector,
    )

    result = compare_turn_embeddings(
        storage,
        "turn_000001",
        "turn_000002",
        "speaker",
    )

    assert result.similarity == pytest.approx(1.0)
    assert result.encoder == "test-encoder"
    assert result.dimension == 3


def test_compare_orthogonal_embeddings(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    add_turn_with_embedding(
        storage,
        turn_id="turn_000001",
        vector=np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
    )
    add_turn_with_embedding(
        storage,
        turn_id="turn_000002",
        vector=np.array(
            [0.0, 1.0],
            dtype=np.float32,
        ),
    )

    result = compare_turn_embeddings(
        storage,
        "turn_000001",
        "turn_000002",
        "speaker",
    )

    assert result.similarity == pytest.approx(0.0)


def test_compare_rejects_different_encoders(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    vector = np.array(
        [1.0, 2.0],
        dtype=np.float32,
    )

    add_turn_with_embedding(
        storage,
        turn_id="turn_000001",
        vector=vector,
        encoder="encoder-a",
    )
    add_turn_with_embedding(
        storage,
        turn_id="turn_000002",
        vector=vector,
        encoder="encoder-b",
    )

    with pytest.raises(
        ValueError,
        match="different encoders",
    ):
        compare_turn_embeddings(
            storage,
            "turn_000001",
            "turn_000002",
            "speaker",
        )


def test_compare_rejects_zero_vector(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    add_turn_with_embedding(
        storage,
        turn_id="turn_000001",
        vector=np.array(
            [0.0, 0.0],
            dtype=np.float32,
        ),
    )
    add_turn_with_embedding(
        storage,
        turn_id="turn_000002",
        vector=np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
    )

    with pytest.raises(
        ValueError,
        match="zero-norm",
    ):
        compare_turn_embeddings(
            storage,
            "turn_000001",
            "turn_000002",
            "speaker",
        )


def test_compare_rejects_different_models(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    vector = np.array(
        [1.0, 2.0],
        dtype=np.float32,
    )

    add_turn_with_embedding(
        storage,
        turn_id="turn_000001",
        vector=vector,
        model="model-a",
    )
    add_turn_with_embedding(
        storage,
        turn_id="turn_000002",
        vector=vector,
        model="model-b",
    )

    with pytest.raises(
        ValueError,
        match="different models",
    ):
        compare_turn_embeddings(
            storage,
            "turn_000001",
            "turn_000002",
            "speaker",
        )


def test_compare_rejects_different_representations(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    vector = np.array(
        [1.0, 2.0],
        dtype=np.float32,
    )

    add_turn_with_embedding(
        storage,
        turn_id="turn_000001",
        vector=vector,
        representation="raw",
    )
    add_turn_with_embedding(
        storage,
        turn_id="turn_000002",
        vector=vector,
        representation="speech",
    )

    with pytest.raises(
        ValueError,
        match="different representations",
    ):
        compare_turn_embeddings(
            storage,
            "turn_000001",
            "turn_000002",
            "speaker",
        )


def test_compare_rejects_different_dimensions(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    add_turn_with_embedding(
        storage,
        turn_id="turn_000001",
        vector=np.array(
            [1.0, 2.0],
            dtype=np.float32,
        ),
    )
    add_turn_with_embedding(
        storage,
        turn_id="turn_000002",
        vector=np.array(
            [1.0, 2.0, 3.0],
            dtype=np.float32,
        ),
    )

    with pytest.raises(
        ValueError,
        match="different dimensions",
    ):
        compare_turn_embeddings(
            storage,
            "turn_000001",
            "turn_000002",
            "speaker",
        )


def test_load_rejects_embedding_path_escape(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    vector = np.array(
        [1.0, 2.0],
        dtype=np.float32,
    )

    add_turn_with_embedding(
        storage,
        turn_id="turn_000001",
        vector=vector,
    )
    add_turn_with_embedding(
        storage,
        turn_id="turn_000002",
        vector=vector,
    )

    def escape_embedding_path(
        record: dict,
    ) -> dict:
        record["embeddings"]["speaker"]["path"] = (
            "../outside.npy"
        )
        return record

    storage.update_turn(
        "turn_000001",
        escape_embedding_path,
    )

    with pytest.raises(
        ValueError,
        match="escapes the dataset",
    ):
        compare_turn_embeddings(
            storage,
            "turn_000001",
            "turn_000002",
            "speaker",
        )

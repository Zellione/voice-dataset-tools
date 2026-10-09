from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voice_dataset.speaker_similarity import (
    compare_turn_embeddings,
    rank_voice_matches,
)
from voice_dataset.storage import DatasetStorage


def make_storage(tmp_path: Path) -> DatasetStorage:
    return DatasetStorage(tmp_path / "dataset")


def add_voice(
    storage: DatasetStorage,
    voice_id: str,
    *,
    ignored: bool = False,
) -> None:
    storage.voices.append(
        {
            "schema_version": 1,
            "record_type": "voice_profile",
            "id": voice_id,
            "character": None,
            "language": None,
            "aliases": [],
            "ignored": ignored,
            "notes": None,
            "metadata": {},
        }
    )


def add_turn_with_embedding(
    storage: DatasetStorage,
    *,
    turn_id: str,
    vector: np.ndarray,
    encoder: str = "test-encoder",
    embedding_name: str = "speaker",
    model: str | None = "test-model",
    representation: str = "speech",
    voice_id: str | None = None,
    assignment_method: str | None = None,
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
                "status": (
                    "assigned"
                    if voice_id is not None
                    else "unknown"
                ),
                "voice_id": voice_id,
                "method": (
                    assignment_method
                    if voice_id is not None
                    else None
                ),
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


def test_rank_voice_matches_groups_manual_references(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    add_voice(storage, "voice_001")
    add_voice(storage, "voice_002")

    add_turn_with_embedding(
        storage,
        turn_id="turn_query",
        vector=np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
    )

    add_turn_with_embedding(
        storage,
        turn_id="turn_voice_1_a",
        vector=np.array(
            [0.8, 0.6],
            dtype=np.float32,
        ),
        voice_id="voice_001",
        assignment_method="manual",
    )

    add_turn_with_embedding(
        storage,
        turn_id="turn_voice_1_b",
        vector=np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        voice_id="voice_001",
        assignment_method="manual",
    )

    add_turn_with_embedding(
        storage,
        turn_id="turn_voice_2",
        vector=np.array(
            [0.0, 1.0],
            dtype=np.float32,
        ),
        voice_id="voice_002",
        assignment_method="manual",
    )

    results = rank_voice_matches(
        storage,
        "turn_query",
        "speaker",
    )

    assert [result.voice_id for result in results] == [
        "voice_001",
        "voice_002",
    ]

    assert [
        match.turn_id
        for match in results[0].matches
    ] == [
        "turn_voice_1_b",
        "turn_voice_1_a",
    ]

    assert [
        match.similarity
        for match in results[0].matches
    ] == pytest.approx(
        [1.0, 0.8]
    )

    assert [
        match.turn_id
        for match in results[1].matches
    ] == [
        "turn_voice_2",
    ]

    assert results[1].matches[0].similarity == pytest.approx(
        0.0
    )

    assert [
        match.source_id
        for match in results[0].matches
    ] == [
        "source_001",
        "source_001",
    ]


def test_rank_voice_matches_skips_unknown_turns(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    add_voice(storage, "voice_001")

    vector = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    add_turn_with_embedding(
        storage,
        turn_id="turn_query",
        vector=vector,
    )

    add_turn_with_embedding(
        storage,
        turn_id="turn_unknown",
        vector=vector,
    )

    results = rank_voice_matches(
        storage,
        "turn_query",
        "speaker",
    )

    assert results == []


def test_rank_voice_matches_skips_non_manual_assignments(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    add_voice(storage, "voice_001")

    vector = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    add_turn_with_embedding(
        storage,
        turn_id="turn_query",
        vector=vector,
    )

    add_turn_with_embedding(
        storage,
        turn_id="turn_automatic",
        vector=vector,
        voice_id="voice_001",
        assignment_method="speaker_similarity",
    )

    results = rank_voice_matches(
        storage,
        "turn_query",
        "speaker",
    )

    assert results == []


def test_rank_voice_matches_skips_ignored_voices(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    add_voice(
        storage,
        "voice_001",
        ignored=True,
    )

    vector = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    add_turn_with_embedding(
        storage,
        turn_id="turn_query",
        vector=vector,
    )

    add_turn_with_embedding(
        storage,
        turn_id="turn_reference",
        vector=vector,
        voice_id="voice_001",
        assignment_method="manual",
    )

    results = rank_voice_matches(
        storage,
        "turn_query",
        "speaker",
    )

    assert results == []


def test_rank_voice_matches_skips_reference_without_embedding(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    add_voice(storage, "voice_001")

    vector = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    add_turn_with_embedding(
        storage,
        turn_id="turn_query",
        vector=vector,
    )

    add_turn_with_embedding(
        storage,
        turn_id="turn_reference",
        vector=vector,
        voice_id="voice_001",
        assignment_method="manual",
    )

    def remove_embedding(
        record: dict,
    ) -> dict:
        record["embeddings"] = {}
        return record

    storage.update_turn(
        "turn_reference",
        remove_embedding,
    )

    results = rank_voice_matches(
        storage,
        "turn_query",
        "speaker",
    )

    assert results == []


def test_rank_voice_matches_rejects_incompatible_embedding(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    add_voice(storage, "voice_001")

    add_turn_with_embedding(
        storage,
        turn_id="turn_query",
        vector=np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        encoder="encoder-a",
    )

    add_turn_with_embedding(
        storage,
        turn_id="turn_reference",
        vector=np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        encoder="encoder-b",
        voice_id="voice_001",
        assignment_method="manual",
    )

    with pytest.raises(
        ValueError,
        match="different encoders",
    ):
        rank_voice_matches(
            storage,
            "turn_query",
            "speaker",
        )

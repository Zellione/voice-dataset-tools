from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from voice_dataset.embeddings import (
    import_embeddings,
    load_embedding_output,
)
from voice_dataset.storage import DatasetStorage


def make_storage(tmp_path: Path) -> DatasetStorage:
    storage = DatasetStorage(tmp_path / "dataset")

    storage.turns.append(
        {
            "schema_version": 2,
            "record_type": "turn",
            "id": "turn_000001",
            "source_id": "source_001",
            "source_start": 0.0,
            "source_end": 1.0,
            "source_regions": [],
            "language": None,
            "transcript": None,
            "representations": {
                "raw": {
                    "path": "audio.wav",
                },
            },
            "embeddings": {},
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

    return storage


def write_embedding_output(
    tmp_path: Path,
    *,
    revision=...,
) -> Path:
    output_dir = tmp_path / "output"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    embedding_path = (
        output_dir / "turn_000001.npy"
    )

    np.save(
        embedding_path,
        np.array(
            [1.0, 2.0, 3.0],
            dtype=np.float32,
        ),
    )

    encoder = {
        "name": "test-encoder",
        "model": "test-model",
    }

    if revision is not ...:
        encoder["revision"] = revision

    manifest = {
        "format": "voice-dataset-embedding-output",
        "version": 1,
        "record_type": "turn",
        "encoder": encoder,
        "representation": "raw",
        "embeddings": [
            {
                "turn_id": "turn_000001",
                "path": "turn_000001.npy",
                "dimension": 3,
                "metadata": {},
            },
        ],
    }

    manifest_path = output_dir / "manifest.json"

    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    return manifest_path


def test_import_preserves_encoder_revision(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    manifest_path = write_embedding_output(
        tmp_path,
        revision="abc123",
    )

    output = load_embedding_output(
        manifest_path
    )

    result = import_embeddings(
        storage,
        output,
        "speaker",
    )

    assert result.imported == 1
    assert result.skipped == 0

    turn = storage.get_turn("turn_000001")
    assert turn is not None

    metadata = turn["embeddings"][
        "speaker"
    ]["metadata"]

    assert metadata["model"] == "test-model"
    assert metadata["revision"] == "abc123"
    assert isinstance(
        metadata["sha256"],
        str,
    )


def test_import_without_revision_is_backward_compatible(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    manifest_path = write_embedding_output(
        tmp_path
    )

    output = load_embedding_output(
        manifest_path
    )

    first = import_embeddings(
        storage,
        output,
        "speaker",
    )

    second = import_embeddings(
        storage,
        output,
        "speaker",
    )

    assert first.imported == 1
    assert first.skipped == 0
    assert second.imported == 0
    assert second.skipped == 1

    turn = storage.get_turn("turn_000001")
    assert turn is not None

    metadata = turn["embeddings"][
        "speaker"
    ]["metadata"]

    assert metadata["model"] == "test-model"
    assert "revision" not in metadata


def test_null_revision_is_not_persisted(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    manifest_path = write_embedding_output(
        tmp_path,
        revision=None,
    )

    output = load_embedding_output(
        manifest_path
    )

    import_embeddings(
        storage,
        output,
        "speaker",
    )

    turn = storage.get_turn("turn_000001")
    assert turn is not None

    metadata = turn["embeddings"][
        "speaker"
    ]["metadata"]

    assert "revision" not in metadata


def test_load_rejects_invalid_encoder_revision(
    tmp_path: Path,
) -> None:
    manifest_path = write_embedding_output(
        tmp_path,
        revision=123,
    )

    with pytest.raises(
        ValueError,
        match=(
            "encoder.revision must be "
            "a string or null"
        ),
    ):
        load_embedding_output(manifest_path)

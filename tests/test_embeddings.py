from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from voice_dataset.embeddings import (
    embed_turns,
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


def test_embed_turns_skips_worker_when_embedding_is_complete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage = make_storage(tmp_path)

    embedding_path = (
        storage.root
        / "turns"
        / "turn_000001"
        / "embeddings"
        / "ecapa_speaker.npy"
    )
    embedding_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        embedding_path,
        np.array(
            [1.0, 2.0, 3.0],
            dtype=np.float32,
        ),
    )

    import hashlib

    sha256 = hashlib.sha256(
        embedding_path.read_bytes()
    ).hexdigest()

    def update(turn):
        turn["representations"]["speaker"] = {
            "path": (
                "turns/turn_000001/"
                "speaker.wav"
            ),
        }
        turn["embeddings"]["ecapa_speaker"] = {
            "encoder": "speechbrain-ecapa",
            "representation": "speaker",
            "path": (
                "turns/turn_000001/embeddings/"
                "ecapa_speaker.npy"
            ),
            "dimension": 3,
            "metadata": {
                "model": "test-model",
                "sha256": sha256,
            },
        }
        return turn

    storage.update_turn(
        "turn_000001",
        update,
    )

    def fail_worker(*args, **kwargs):
        raise AssertionError(
            "embedding worker should not run"
        )

    monkeypatch.setattr(
        "voice_dataset.embeddings.run_worker",
        fail_worker,
    )

    result = embed_turns(
        storage,
        encoder="ecapa",
        representation="speaker",
        name="ecapa_speaker",
    )

    assert result.encoder == "ecapa"
    assert result.representation == "speaker"
    assert result.imported == 0
    assert result.skipped == 1


def test_embed_turns_rejects_invalid_existing_encoder(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    embedding_path = (
        storage.root
        / "turns"
        / "turn_000001"
        / "embeddings"
        / "ecapa_speaker.npy"
    )
    embedding_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        embedding_path,
        np.array(
            [1.0, 2.0, 3.0],
            dtype=np.float32,
        ),
    )

    import hashlib

    sha256 = hashlib.sha256(
        embedding_path.read_bytes()
    ).hexdigest()

    def update(turn):
        turn["representations"]["speaker"] = {
            "path": (
                "turns/turn_000001/"
                "speaker.wav"
            ),
        }
        turn["embeddings"]["ecapa_speaker"] = {
            "encoder": None,
            "representation": "speaker",
            "path": (
                "turns/turn_000001/embeddings/"
                "ecapa_speaker.npy"
            ),
            "dimension": 3,
            "metadata": {
                "model": "test-model",
                "sha256": sha256,
            },
        }
        return turn

    storage.update_turn(
        "turn_000001",
        update,
    )

    with pytest.raises(
        ValueError,
        match="invalid encoder",
    ):
        embed_turns(
            storage,
            encoder="ecapa",
            representation="speaker",
            name="ecapa_speaker",
        )


def test_embed_turns_rejects_corrupt_existing_embedding(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    embedding_path = (
        storage.root
        / "turns"
        / "turn_000001"
        / "embeddings"
        / "ecapa_speaker.npy"
    )
    embedding_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        embedding_path,
        np.array(
            [1.0, 2.0, 3.0],
            dtype=np.float32,
        ),
    )

    def update(turn):
        turn["representations"]["speaker"] = {
            "path": (
                "turns/turn_000001/"
                "speaker.wav"
            ),
        }
        turn["embeddings"]["ecapa_speaker"] = {
            "encoder": "ecapa",
            "representation": "speaker",
            "path": (
                "turns/turn_000001/embeddings/"
                "ecapa_speaker.npy"
            ),
            "dimension": 3,
            "metadata": {
                "model": "test-model",
                "sha256": "not-the-real-hash",
            },
        }
        return turn

    storage.update_turn(
        "turn_000001",
        update,
    )

    with pytest.raises(
        ValueError,
        match="does not match metadata",
    ):
        embed_turns(
            storage,
            encoder="ecapa",
            representation="speaker",
            name="ecapa_speaker",
        )


def test_embed_turns_runs_worker_when_embedding_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage = make_storage(tmp_path)

    def update(turn):
        turn["representations"]["speaker"] = {
            "path": (
                "turns/turn_000001/"
                "speaker.wav"
            ),
        }
        return turn

    storage.update_turn(
        "turn_000001",
        update,
    )

    def worker_was_called(*args, **kwargs):
        raise RuntimeError("worker was called")

    monkeypatch.setattr(
        "voice_dataset.embeddings.run_worker",
        worker_was_called,
    )

    with pytest.raises(
        RuntimeError,
        match="worker was called",
    ):
        embed_turns(
            storage,
            encoder="ecapa",
            representation="speaker",
            name="ecapa_speaker",
        )

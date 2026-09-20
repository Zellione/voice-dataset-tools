from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .schema import EmbeddingReference
from .storage import DatasetStorage


EMBEDDING_OUTPUT_FORMAT = (
    "voice-dataset-embedding-output"
)

EMBEDDING_OUTPUT_VERSION = 1

RecordType = Literal["region", "turn"]


@dataclass
class ValidatedEmbedding:
    record_id: str
    path: Path
    dimension: int
    metadata: dict[str, Any]


@dataclass
class EmbeddingOutput:
    record_type: RecordType
    encoder_name: str
    encoder_model: str | None
    encoder_revision: str | None
    representation: str
    embeddings: list[ValidatedEmbedding]


@dataclass
class EmbeddingImportResult:
    imported: int
    skipped: int


def _optional_string(
    value: Any,
    field: str,
) -> str | None:
    if value is None:
        return None

    if not isinstance(value, str):
        raise ValueError(
            f"{field} must be a string or null"
        )

    return value


def _positive_int(
    value: Any,
    field: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            f"{field} must be a positive integer"
        )

    return value


def _record_type(
    value: Any,
) -> RecordType:
    # Version 1 manifests created before record_type
    # existed always described candidate regions.
    if value is None:
        return "region"

    if value not in ("region", "turn"):
        raise ValueError(
            "record_type must be "
            "'region' or 'turn'"
        )

    return value


def _id_field(
    record_type: RecordType,
) -> str:
    if record_type == "region":
        return "region_id"

    return "turn_id"


def load_embedding_output(
    path: Path,
) -> EmbeddingOutput:
    path = path.resolve()

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            document = json.load(file)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid embedding JSON: {exc}"
        ) from exc

    if not isinstance(document, dict):
        raise ValueError(
            "Embedding output must be an object"
        )

    if (
        document.get("format")
        != EMBEDDING_OUTPUT_FORMAT
    ):
        raise ValueError(
            "Unsupported embedding output format"
        )

    if (
        document.get("version")
        != EMBEDDING_OUTPUT_VERSION
    ):
        raise ValueError(
            "Unsupported embedding output version"
        )

    record_type = _record_type(
        document.get("record_type")
    )

    id_field = _id_field(record_type)

    encoder = document.get("encoder")

    if not isinstance(encoder, dict):
        raise ValueError(
            "encoder must be an object"
        )

    encoder_name = encoder.get("name")

    if (
        not isinstance(encoder_name, str)
        or not encoder_name
    ):
        raise ValueError(
            "encoder.name must be a "
            "non-empty string"
        )

    encoder_model = _optional_string(
        encoder.get("model"),
        "encoder.model",
    )

    encoder_revision = _optional_string(
        encoder.get("revision"),
        "encoder.revision",
    )

    representation = document.get(
        "representation"
    )

    if (
        not isinstance(representation, str)
        or not representation
    ):
        raise ValueError(
            "representation must be a "
            "non-empty string"
        )

    raw_embeddings = document.get(
        "embeddings"
    )

    if not isinstance(raw_embeddings, list):
        raise ValueError(
            "embeddings must be a list"
        )

    manifest_dir = path.parent

    embeddings: list[
        ValidatedEmbedding
    ] = []

    seen_ids: set[str] = set()

    for index, item in enumerate(
        raw_embeddings
    ):
        if not isinstance(item, dict):
            raise ValueError(
                "embedding item "
                f"{index} must be an object"
            )

        record_id = item.get(id_field)

        if (
            not isinstance(record_id, str)
            or not record_id
        ):
            raise ValueError(
                "embedding item "
                f"{index} has invalid {id_field}"
            )

        if record_id in seen_ids:
            raise ValueError(
                f"Duplicate embedding {id_field}: "
                f"{record_id}"
            )

        seen_ids.add(record_id)

        raw_path = item.get("path")

        if (
            not isinstance(raw_path, str)
            or not raw_path
        ):
            raise ValueError(
                f"{record_id}: path must be a "
                "non-empty string"
            )

        embedding_path = Path(raw_path)

        if not embedding_path.is_absolute():
            embedding_path = (
                manifest_dir
                / embedding_path
            )

        embedding_path = (
            embedding_path.resolve()
        )

        if not embedding_path.is_file():
            raise ValueError(
                f"{record_id}: embedding file "
                f"does not exist: "
                f"{embedding_path}"
            )

        dimension = _positive_int(
            item.get("dimension"),
            f"{record_id}.dimension",
        )

        metadata = item.get(
            "metadata",
            {},
        )

        if not isinstance(metadata, dict):
            raise ValueError(
                f"{record_id}.metadata "
                "must be an object"
            )

        try:
            vector = np.load(
                embedding_path,
                allow_pickle=False,
            )
        except Exception as exc:
            raise ValueError(
                f"{record_id}: cannot load "
                f"embedding: {exc}"
            ) from exc

        if vector.ndim != 1:
            raise ValueError(
                f"{record_id}: embedding must "
                "be a 1-D vector"
            )

        if vector.shape[0] != dimension:
            raise ValueError(
                f"{record_id}: declared "
                f"dimension {dimension} does "
                f"not match vector dimension "
                f"{vector.shape[0]}"
            )

        if not np.issubdtype(
            vector.dtype,
            np.number,
        ):
            raise ValueError(
                f"{record_id}: embedding must "
                "contain numeric values"
            )

        if not np.all(
            np.isfinite(vector)
        ):
            raise ValueError(
                f"{record_id}: embedding contains "
                "non-finite values"
            )

        embeddings.append(
            ValidatedEmbedding(
                record_id=record_id,
                path=embedding_path,
                dimension=dimension,
                metadata=metadata,
            )
        )

    return EmbeddingOutput(
        record_type=record_type,
        encoder_name=encoder_name,
        encoder_model=encoder_model,
        encoder_revision=encoder_revision,
        representation=representation,
        embeddings=embeddings,
    )


def _sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(
            1024 * 1024
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _expected_reference(
    output: EmbeddingOutput,
    embedding: ValidatedEmbedding,
    relative_path: Path,
) -> EmbeddingReference:
    metadata = {
        **embedding.metadata,
        "model": output.encoder_model,
        "sha256": _sha256_file(
            embedding.path
        ),
    }

    if output.encoder_revision is not None:
        metadata["revision"] = (
            output.encoder_revision
        )

    return EmbeddingReference(
        encoder=output.encoder_name,
        representation=(
            output.representation
        ),
        path=str(relative_path),
        dimension=embedding.dimension,
        metadata=metadata,
    )


def _load_records(
    storage: DatasetStorage,
    record_type: RecordType,
) -> list[dict[str, Any]]:
    if record_type == "region":
        return storage.regions.load()

    return storage.turns.load()


def _get_record(
    storage: DatasetStorage,
    record_type: RecordType,
    record_id: str,
) -> dict[str, Any] | None:
    if record_type == "region":
        return storage.get_region(record_id)

    return storage.get_turn(record_id)


def _update_record(
    storage: DatasetStorage,
    record_type: RecordType,
    record_id: str,
    update_fn,
) -> dict[str, Any]:
    if record_type == "region":
        return storage.update_region(
            record_id,
            update_fn,
        )

    return storage.update_turn(
        record_id,
        update_fn,
    )


def import_embeddings(
    storage: DatasetStorage,
    output: EmbeddingOutput,
    name: str,
) -> EmbeddingImportResult:
    if not isinstance(name, str) or not name:
        raise ValueError(
            "Embedding name must be a "
            "non-empty string"
        )

    records = _load_records(
        storage,
        output.record_type,
    )

    records_by_id: dict[
        str,
        dict[str, Any],
    ] = {}

    for record in records:
        record_id = record.get("id")

        if (
            not isinstance(record_id, str)
            or not record_id
        ):
            raise ValueError(
                "Persisted "
                f"{output.record_type} "
                "has invalid id"
            )

        if record_id in records_by_id:
            raise ValueError(
                f"Duplicate "
                f"{output.record_type} id: "
                f"{record_id}"
            )

        records_by_id[record_id] = record

    # Preflight everything before copying or
    # updating dataset metadata.
    for embedding in output.embeddings:
        record = records_by_id.get(
            embedding.record_id
        )

        if record is None:
            raise ValueError(
                "Embedding references unknown "
                f"{output.record_type}: "
                f"{embedding.record_id}"
            )

        representations = record.get(
            "representations"
        )

        if not isinstance(
            representations,
            dict,
        ):
            raise ValueError(
                f"{embedding.record_id}: "
                "representations must be "
                "an object"
            )

        if (
            output.representation
            not in representations
        ):
            raise ValueError(
                f"{embedding.record_id}: "
                "representation does not exist: "
                f"{output.representation}"
            )

        existing_embeddings = record.get(
            "embeddings"
        )

        if not isinstance(
            existing_embeddings,
            dict,
        ):
            raise ValueError(
                f"{embedding.record_id}: "
                "embeddings must be an object"
            )

    imported = 0
    skipped = 0

    record_directory = (
        "regions"
        if output.record_type == "region"
        else "turns"
    )

    for embedding in output.embeddings:
        record = _get_record(
            storage,
            output.record_type,
            embedding.record_id,
        )

        if record is None:
            raise RuntimeError(
                f"{output.record_type.capitalize()} "
                "disappeared during embedding "
                "import: "
                f"{embedding.record_id}"
            )

        relative_path = (
            Path(record_directory)
            / embedding.record_id
            / "embeddings"
            / f"{name}.npy"
        )

        destination = (
            storage.root
            / relative_path
        )

        expected = _expected_reference(
            output,
            embedding,
            relative_path,
        ).to_dict()

        existing_embeddings = record[
            "embeddings"
        ]

        existing = existing_embeddings.get(
            name
        )

        if existing is not None:
            if not isinstance(
                existing,
                dict,
            ):
                raise ValueError(
                    "Persisted embedding "
                    "reference must be an object: "
                    f"{embedding.record_id}/{name}"
                )

            if existing != expected:
                raise ValueError(
                    "Embedding evidence conflict: "
                    f"{embedding.record_id}/{name}"
                )

            if not destination.is_file():
                raise ValueError(
                    "Embedding metadata exists "
                    "without file: "
                    f"{destination}"
                )

            if (
                _sha256_file(destination)
                != expected["metadata"]["sha256"]
            ):
                raise ValueError(
                    "Persisted embedding file "
                    "does not match metadata: "
                    f"{destination}"
                )

            skipped += 1
            continue

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise ValueError(
                "Embedding file exists without "
                "metadata: "
                f"{destination}"
            )

        fd, temporary_name = (
            tempfile.mkstemp(
                prefix=f".{name}.",
                suffix=".tmp.npy",
                dir=destination.parent,
            )
        )

        os.close(fd)

        temporary = Path(
            temporary_name
        )

        try:
            shutil.copyfile(
                embedding.path,
                temporary,
            )

            if (
                _sha256_file(temporary)
                != expected["metadata"]["sha256"]
            ):
                raise RuntimeError(
                    "Embedding changed while "
                    "being imported: "
                    f"{embedding.path}"
                )

            os.replace(
                temporary,
                destination,
            )

            def update(
                record: dict[str, Any],
            ) -> dict[str, Any]:
                current = record.get(
                    "embeddings"
                )

                if not isinstance(
                    current,
                    dict,
                ):
                    raise ValueError(
                        "embeddings must be "
                        "an object"
                    )

                if name in current:
                    raise ValueError(
                        "Embedding already exists: "
                        f"{embedding.record_id}/"
                        f"{name}"
                    )

                current[name] = expected

                return record

            _update_record(
                storage,
                output.record_type,
                embedding.record_id,
                update,
            )

        except Exception:
            temporary.unlink(
                missing_ok=True
            )

            destination.unlink(
                missing_ok=True
            )

            raise

        imported += 1

    return EmbeddingImportResult(
        imported=imported,
        skipped=skipped,
    )

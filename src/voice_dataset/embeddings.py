from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .schema import EmbeddingReference
from .storage import DatasetStorage


EMBEDDING_OUTPUT_FORMAT = (
    "voice-dataset-embedding-output"
)

EMBEDDING_OUTPUT_VERSION = 1


@dataclass
class ValidatedEmbedding:
    region_id: str
    path: Path
    dimension: int
    metadata: dict[str, Any]


@dataclass
class EmbeddingOutput:
    encoder_name: str
    encoder_model: str | None
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

    seen_regions: set[str] = set()

    for index, item in enumerate(
        raw_embeddings
    ):
        if not isinstance(item, dict):
            raise ValueError(
                "embedding item "
                f"{index} must be an object"
            )

        region_id = item.get("region_id")

        if (
            not isinstance(region_id, str)
            or not region_id
        ):
            raise ValueError(
                "embedding item "
                f"{index} has invalid region_id"
            )

        if region_id in seen_regions:
            raise ValueError(
                "Duplicate embedding region_id: "
                f"{region_id}"
            )

        seen_regions.add(region_id)

        raw_path = item.get("path")

        if (
            not isinstance(raw_path, str)
            or not raw_path
        ):
            raise ValueError(
                f"{region_id}: path must be a "
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
                f"{region_id}: embedding file "
                f"does not exist: "
                f"{embedding_path}"
            )

        dimension = _positive_int(
            item.get("dimension"),
            f"{region_id}.dimension",
        )

        metadata = item.get(
            "metadata",
            {},
        )

        if not isinstance(metadata, dict):
            raise ValueError(
                f"{region_id}.metadata "
                "must be an object"
            )

        try:
            vector = np.load(
                embedding_path,
                allow_pickle=False,
            )
        except Exception as exc:
            raise ValueError(
                f"{region_id}: cannot load "
                f"embedding: {exc}"
            ) from exc

        if vector.ndim != 1:
            raise ValueError(
                f"{region_id}: embedding must "
                "be a 1-D vector"
            )

        if vector.shape[0] != dimension:
            raise ValueError(
                f"{region_id}: declared "
                f"dimension {dimension} does "
                f"not match vector dimension "
                f"{vector.shape[0]}"
            )

        if not np.issubdtype(
            vector.dtype,
            np.number,
        ):
            raise ValueError(
                f"{region_id}: embedding must "
                "contain numeric values"
            )

        if not np.all(
            np.isfinite(vector)
        ):
            raise ValueError(
                f"{region_id}: embedding contains "
                "non-finite values"
            )

        embeddings.append(
            ValidatedEmbedding(
                region_id=region_id,
                path=embedding_path,
                dimension=dimension,
                metadata=metadata,
            )
        )

    return EmbeddingOutput(
        encoder_name=encoder_name,
        encoder_model=encoder_model,
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
    return EmbeddingReference(
        encoder=output.encoder_name,
        representation=(
            output.representation
        ),
        path=str(relative_path),
        dimension=embedding.dimension,
        metadata={
            **embedding.metadata,
            "model": output.encoder_model,
            "sha256": _sha256_file(
                embedding.path
            ),
        },
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

    regions = storage.regions.load()

    regions_by_id: dict[
        str,
        dict[str, Any],
    ] = {}

    for region in regions:
        region_id = region.get("id")

        if (
            not isinstance(region_id, str)
            or not region_id
        ):
            raise ValueError(
                "Persisted region has invalid id"
            )

        if region_id in regions_by_id:
            raise ValueError(
                f"Duplicate region id: "
                f"{region_id}"
            )

        regions_by_id[region_id] = region

    # Preflight everything before copying or
    # updating dataset metadata.
    for embedding in output.embeddings:
        region = regions_by_id.get(
            embedding.region_id
        )

        if region is None:
            raise ValueError(
                "Embedding references unknown "
                f"region: {embedding.region_id}"
            )

        representations = region.get(
            "representations"
        )

        if not isinstance(
            representations,
            dict,
        ):
            raise ValueError(
                f"{embedding.region_id}: "
                "representations must be "
                "an object"
            )

        if (
            output.representation
            not in representations
        ):
            raise ValueError(
                f"{embedding.region_id}: "
                "representation does not exist: "
                f"{output.representation}"
            )

        existing_embeddings = region.get(
            "embeddings"
        )

        if not isinstance(
            existing_embeddings,
            dict,
        ):
            raise ValueError(
                f"{embedding.region_id}: "
                "embeddings must be an object"
            )

    imported = 0
    skipped = 0

    for embedding in output.embeddings:
        region = storage.get_region(
            embedding.region_id
        )

        if region is None:
            raise RuntimeError(
                "Region disappeared during "
                "embedding import: "
                f"{embedding.region_id}"
            )

        relative_path = (
            Path("regions")
            / embedding.region_id
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

        existing_embeddings = region[
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
                    f"{embedding.region_id}/{name}"
                )

            if existing != expected:
                raise ValueError(
                    "Embedding evidence conflict: "
                    f"{embedding.region_id}/{name}"
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
                        f"{embedding.region_id}/"
                        f"{name}"
                    )

                current[name] = expected

                return record

            storage.update_region(
                embedding.region_id,
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

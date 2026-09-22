from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .storage import DatasetStorage


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _parse_sortformer_segments(
    path: Path,
) -> list[dict[str, Any]]:
    document = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(document, list):
        raise ValueError(
            "SortFormer segments must be a list"
        )

    segments: list[dict[str, Any]] = []

    for index, item in enumerate(document):
        if not isinstance(item, str):
            raise ValueError(
                f"SortFormer segment {index} "
                "must be a string"
            )

        parts = item.split()

        if len(parts) != 3:
            raise ValueError(
                f"Invalid SortFormer segment {index}: "
                f"{item!r}"
            )

        raw_start, raw_end, speaker = parts

        try:
            start = float(raw_start)
            end = float(raw_end)
        except ValueError as exc:
            raise ValueError(
                f"Invalid SortFormer segment {index}: "
                f"{item!r}"
            ) from exc

        if start < 0 or end <= start:
            raise ValueError(
                f"Invalid SortFormer segment range "
                f"{index}: {start}-{end}"
            )

        if not speaker:
            raise ValueError(
                f"SortFormer segment {index} "
                "has no speaker"
            )

        segments.append(
            {
                "start": start,
                "end": end,
                "speaker": speaker,
            }
        )

    return segments


def import_sortformer_evidence(
    storage: DatasetStorage,
    source_id: str,
    name: str,
    *,
    model: str,
    representation: str,
    segments_path: Path,
    activity_path: Path,
    frame_duration: float,
) -> dict[str, Any]:
    if not name:
        raise ValueError(
            "Diarization evidence name must not be empty"
        )

    if not model:
        raise ValueError(
            "Diarization model must not be empty"
        )

    if not representation:
        raise ValueError(
            "Diarization representation must not be empty"
        )

    frame_duration = float(frame_duration)

    if frame_duration <= 0:
        raise ValueError(
            "Diarization frame duration must be positive"
        )

    source = storage.get_source(source_id)

    if source is None:
        raise KeyError(
            f"Source does not exist: {source_id}"
        )

    representations = source.get(
        "representations",
        {},
    )

    if not isinstance(representations, dict):
        raise ValueError(
            f"Source has invalid representations: "
            f"{source_id}"
        )

    if representation not in representations:
        raise ValueError(
            "Source representation does not exist: "
            f"{source_id}/{representation}"
        )

    segments_path = segments_path.resolve()
    activity_path = activity_path.resolve()

    if not segments_path.is_file():
        raise ValueError(
            "SortFormer segments file does not exist: "
            f"{segments_path}"
        )

    if not activity_path.is_file():
        raise ValueError(
            "SortFormer activity file does not exist: "
            f"{activity_path}"
        )

    segments = _parse_sortformer_segments(
        segments_path
    )

    activity = np.load(
        activity_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    if activity.ndim != 3:
        raise ValueError(
            "SortFormer activity must have shape "
            "(batch, frames, speakers)"
        )

    if activity.shape[0] != 1:
        raise ValueError(
            "SortFormer activity batch dimension "
            "must be 1"
        )

    if activity.shape[1] <= 0:
        raise ValueError(
            "SortFormer activity must contain frames"
        )

    if activity.shape[2] <= 0:
        raise ValueError(
            "SortFormer activity must contain speakers"
        )

    relative_path = (
        Path("evidence")
        / "diarization"
        / name
        / "speaker_activity.npy"
    )

    destination = storage.root / relative_path

    activity_sha256 = _sha256_file(
        activity_path
    )

    evidence = {
        "model": model,
        "representation": representation,
        "segments": segments,
        "activity": {
            "path": str(relative_path),
            "shape": [
                int(value)
                for value in activity.shape
            ],
            "dtype": str(activity.dtype),
            "frame_duration": frame_duration,
            "sha256": activity_sha256,
        },
    }

    metadata = source.get("metadata", {})

    if not isinstance(metadata, dict):
        raise ValueError(
            f"Source has invalid metadata: "
            f"{source_id}"
        )

    diarization = metadata.get(
        "diarization",
        {},
    )

    if not isinstance(diarization, dict):
        raise ValueError(
            f"Source has invalid diarization metadata: "
            f"{source_id}"
        )

    existing = diarization.get(name)

    if existing is not None:
        if existing != evidence:
            raise ValueError(
                "Diarization evidence already exists "
                "with different data: "
                f"{source_id}/{name}"
            )

        if not destination.is_file():
            raise ValueError(
                "Diarization metadata exists "
                "without activity file: "
                f"{destination}"
            )

        if (
            _sha256_file(destination)
            != activity_sha256
        ):
            raise ValueError(
                "Persisted diarization activity "
                "does not match metadata: "
                f"{destination}"
            )

        return source

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        raise ValueError(
            "Diarization activity file exists "
            "without metadata: "
            f"{destination}"
        )

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{name}.",
        suffix=".tmp.npy",
        dir=destination.parent,
    )
    os.close(fd)

    temporary = Path(temporary_name)

    try:
        shutil.copyfile(
            activity_path,
            temporary,
        )

        if (
            _sha256_file(temporary)
            != activity_sha256
        ):
            raise RuntimeError(
                "SortFormer activity changed while "
                "being imported: "
                f"{activity_path}"
            )

        os.replace(
            temporary,
            destination,
        )

        def update(
            record: dict[str, Any],
        ) -> dict[str, Any]:
            current_metadata = record.get(
                "metadata",
                {},
            )

            if not isinstance(
                current_metadata,
                dict,
            ):
                raise ValueError(
                    "Source metadata must be an object"
                )

            current_diarization = (
                current_metadata.get(
                    "diarization",
                    {},
                )
            )

            if not isinstance(
                current_diarization,
                dict,
            ):
                raise ValueError(
                    "Source diarization metadata "
                    "must be an object"
                )

            if name in current_diarization:
                raise ValueError(
                    "Diarization evidence already exists: "
                    f"{source_id}/{name}"
                )

            current_diarization[name] = evidence
            current_metadata["diarization"] = (
                current_diarization
            )
            record["metadata"] = current_metadata

            return record

        updated = storage.update_source(
            source_id,
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

    return updated

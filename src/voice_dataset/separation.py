from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .sources import (
    probe_audio_representation,
    register_derived_representation,
    resolve_source_representation,
)
from .storage import DatasetStorage
from .workers import run_worker, worker


BANDIT_PROCESSOR = "bandit-v2"
BANDIT_REVISION = (
    "e6bdc5bf56abc10d72f09fb6214c224e0da5864b"
)
BANDIT_CHECKPOINT_SHA256 = (
    "abcfccf65446752a057f4a302c941479a54b7560ebf8d7b"
    "ca039d2ea98e64cfc"
)


@dataclass(frozen=True)
class SeparationResult:
    source_id: str
    representation_name: str
    path: Path
    representation: dict[str, Any]


def separate_source(
    storage: DatasetStorage,
    source_id: str,
    *,
    input_representation: str,
    output_representation: str = "speech",
    chunk: float = 30.0,
    overlap: float = 2.0,
) -> SeparationResult:
    if chunk <= 0:
        raise ValueError(
            "chunk must be greater than zero"
        )

    if overlap < 0:
        raise ValueError(
            "overlap must not be negative"
        )

    if overlap >= chunk:
        raise ValueError(
            "overlap must be smaller than chunk"
        )

    _, input_path = resolve_source_representation(
        storage,
        source_id,
        input_representation,
    )

    output_path = (
        storage.root
        / "audio"
        / source_id
        / output_representation
        / "speech.wav"
    ).resolve()

    parameters = {
        "chunk": float(chunk),
        "overlap": float(overlap),
    }

    metadata = {
        "checkpoint_sha256": (
            BANDIT_CHECKPOINT_SHA256
        ),
        "parameters": parameters,
    }

    existing_source = storage.get_source(source_id)

    if existing_source is None:
        raise KeyError(
            f"Source does not exist: {source_id}"
        )

    existing = existing_source.get(
        "representations",
        {},
    ).get(output_representation)

    if existing is not None:
        _, existing_path = (
            resolve_source_representation(
                storage,
                source_id,
                output_representation,
            )
        )

        expected = {
            "kind": "separated_speech",
            "processor": BANDIT_PROCESSOR,
            "processor_version": BANDIT_REVISION,
            "purposes": [
                "asr",
                "review",
                "tts_candidate",
            ],
            "metadata": metadata,
        }

        for key, value in expected.items():
            if existing.get(key) != value:
                raise ValueError(
                    "Source representation already "
                    "exists with different provenance: "
                    f"{output_representation}"
                )

        return SeparationResult(
            source_id=source_id,
            representation_name=(
                output_representation
            ),
            path=existing_path,
            representation=existing,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        raise ValueError(
            "Derived representation file already "
            "exists without a registered "
            "representation: "
            f"{output_path}"
        )

    with tempfile.TemporaryDirectory(
        prefix="voice-dataset-bandit-"
    ) as temporary_directory:
        worker_output = (
            Path(temporary_directory)
            / "output"
        )

        run_worker(
            worker("bandit"),
            [
                input_path,
                worker_output,
                "--chunk",
                str(chunk),
                "--overlap",
                str(overlap),
            ],
        )

        speech_path = (
            worker_output
            / "speech_estimate.wav"
        )

        probe_audio_representation(
            speech_path
        )

        temporary_target = (
            output_path.parent
            / ".speech.wav.tmp"
        )

        if temporary_target.exists():
            temporary_target.unlink()

        shutil.copyfile(
            speech_path,
            temporary_target,
        )

        probe_audio_representation(
            temporary_target
        )

        temporary_target.replace(
            output_path
        )

    try:
        source = register_derived_representation(
            storage,
            source_id,
            output_representation,
            path=output_path,
            kind="separated_speech",
            parent_representation_name=(
                input_representation
            ),
            processor=BANDIT_PROCESSOR,
            processor_version=BANDIT_REVISION,
            purposes=[
                "asr",
                "review",
                "tts_candidate",
            ],
            metadata=metadata,
        )
    except Exception:
        output_path.unlink(missing_ok=True)
        raise

    representation = source[
        "representations"
    ][output_representation]

    return SeparationResult(
        source_id=source_id,
        representation_name=output_representation,
        path=output_path,
        representation=representation,
    )

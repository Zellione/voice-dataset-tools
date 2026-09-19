from pathlib import Path
from typing import Any

import sounddevice as sd
import soundfile as sf


def representation_path(
    dataset: Path,
    representation: dict[str, Any],
) -> Path:
    relative_path = representation.get("path")

    if not relative_path:
        raise ValueError(
            "Audio representation has no path"
        )

    path = (dataset / relative_path).resolve()
    dataset_root = dataset.resolve()

    try:
        path.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(
            "Audio representation path is outside "
            f"the dataset: {relative_path}"
        ) from exc

    if not path.is_file():
        raise FileNotFoundError(
            f"Audio representation does not exist: {path}"
        )

    return path


def preferred_review_representation(
    turn: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    representations = turn.get(
        "representations",
        {},
    )

    for name, representation in representations.items():
        purposes = representation.get(
            "purposes",
            [],
        )

        if "review" in purposes:
            return name, representation

    raise ValueError(
        f"Turn {turn.get('id', '<unknown>')} has no "
        "audio representation for review"
    )


def play_file(path: Path) -> None:
    audio, sample_rate = sf.read(
        path,
        dtype="float32",
    )

    sd.stop()
    sd.play(
        audio,
        sample_rate,
    )


def wait() -> None:
    sd.wait()


def stop() -> None:
    sd.stop()


def play_representation(
    dataset: Path,
    representation: dict[str, Any],
    *,
    blocking: bool = False,
) -> Path:
    path = representation_path(
        dataset,
        representation,
    )

    play_file(path)

    if blocking:
        wait()

    return path


def play_preferred_review_audio(
    dataset: Path,
    turn: dict[str, Any],
    *,
    blocking: bool = False,
) -> tuple[str, Path]:
    name, representation = (
        preferred_review_representation(turn)
    )

    path = play_representation(
        dataset,
        representation,
        blocking=blocking,
    )

    return name, path

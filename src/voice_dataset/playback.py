from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import sounddevice as sd
import soundfile as sf

from .media import extract_media_audio_region
from .timeline import representation_time_to_media_time


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


def external_representation_path(
    representation: dict[str, Any],
) -> Path:
    raw_path = representation.get("path")

    if not raw_path:
        raise ValueError(
            "Audio representation has no path"
        )

    path = Path(raw_path).expanduser()

    if not path.is_absolute():
        raise ValueError(
            "External audio representation path "
            f"must be absolute: {raw_path}"
        )

    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Audio representation does not exist: {path}"
        )

    return path


def source_media_path(
    source: dict[str, Any],
) -> Path:
    raw_path = source.get("media_path")

    if not raw_path:
        raise ValueError(
            "Source has no media_path"
        )

    path = Path(raw_path).expanduser()

    if not path.is_absolute():
        raise ValueError(
            "Source media_path must be absolute: "
            f"{raw_path}"
        )

    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Source media does not exist: {path}"
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


def preferred_context_representation(
    source: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    representations = source.get(
        "representations",
        {},
    )

    for name, representation in representations.items():
        purposes = representation.get(
            "purposes",
            [],
        )

        if "context" in purposes:
            return name, representation

    raise ValueError(
        f"Source {source.get('id', '<unknown>')} has no "
        "audio representation for context"
    )


def has_original_media_mapping(
    representation: dict[str, Any],
) -> bool:
    return (
        representation.get("media_start") is not None
        and representation.get("stream_index") is not None
        and representation.get("channel_mode") is not None
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


def play_file_region(
    path: Path,
    start: float,
    end: float,
) -> tuple[float, float]:
    if start < 0:
        raise ValueError(
            "Audio region start must not be negative"
        )

    if end <= start:
        raise ValueError(
            "Audio region end must be after start"
        )

    with sf.SoundFile(path) as audio_file:
        sample_rate = audio_file.samplerate

        if sample_rate <= 0:
            raise ValueError(
                f"Invalid sample rate: {sample_rate}"
            )

        duration = (
            len(audio_file)
            / sample_rate
        )

        actual_start = min(
            start,
            duration,
        )
        actual_end = min(
            end,
            duration,
        )

        if actual_end <= actual_start:
            raise ValueError(
                "Audio region is outside the file"
            )

        start_frame = round(
            actual_start * sample_rate
        )
        end_frame = round(
            actual_end * sample_rate
        )

        audio_file.seek(start_frame)

        audio = audio_file.read(
            end_frame - start_frame,
            dtype="float32",
        )

    sd.stop()
    sd.play(
        audio,
        sample_rate,
    )

    return actual_start, actual_end


def wait() -> None:
    sd.wait()


def stop() -> None:
    sd.stop()


def is_playing() -> bool:
    try:
        stream = sd.get_stream()
    except RuntimeError:
        return False

    return bool(stream.active)


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


def play_original_media_context(
    source: dict[str, Any],
    representation: dict[str, Any],
    source_start: float,
    source_end: float,
    *,
    padding: float,
    blocking: bool,
) -> tuple[Path, float, float]:
    media_path = source_media_path(source)

    media_start = representation_time_to_media_time(
        representation,
        source_start,
    )
    media_end = representation_time_to_media_time(
        representation,
        source_end,
    )

    requested_start = max(
        0.0,
        media_start - padding,
    )
    requested_end = media_end + padding

    stream_index = representation.get(
        "stream_index"
    )
    channel_mode = representation.get(
        "channel_mode"
    )

    if not isinstance(stream_index, int):
        raise ValueError(
            "Context representation has invalid "
            "stream_index"
        )

    if channel_mode not in {
        "mono",
        "center",
    }:
        raise ValueError(
            "Context representation has invalid "
            "channel_mode"
        )

    with TemporaryDirectory(
        prefix="voice-dataset-context-"
    ) as temp_dir:
        context_path = (
            Path(temp_dir) / "context.wav"
        )

        extract_media_audio_region(
            source=media_path,
            destination=context_path,
            start=requested_start,
            end=requested_end,
            stream_index=stream_index,
            channel_mode=channel_mode,
        )

        play_file(context_path)

        if blocking:
            wait()

    return (
        media_path,
        requested_start,
        requested_end,
    )


def play_turn_context(
    source: dict[str, Any],
    turn: dict[str, Any],
    *,
    padding: float = 2.0,
    blocking: bool = False,
) -> tuple[str, Path, float, float]:
    if padding < 0:
        raise ValueError(
            "Context padding must not be negative"
        )

    source_id = source.get("id")
    turn_source_id = turn.get("source_id")

    if source_id != turn_source_id:
        raise ValueError(
            "Turn source does not match "
            f"source record: {turn_source_id!r} "
            f"!= {source_id!r}"
        )

    source_start = turn.get("source_start")
    source_end = turn.get("source_end")

    if not isinstance(
        source_start,
        (int, float),
    ):
        raise ValueError(
            "Turn has invalid source_start"
        )

    if not isinstance(
        source_end,
        (int, float),
    ):
        raise ValueError(
            "Turn has invalid source_end"
        )

    source_start = float(source_start)
    source_end = float(source_end)

    if source_end <= source_start:
        raise ValueError(
            "Turn source_end must be after "
            "source_start"
        )

    name, representation = (
        preferred_context_representation(source)
    )

    if has_original_media_mapping(
        representation
    ):
        (
            media_path,
            actual_start,
            actual_end,
        ) = play_original_media_context(
            source,
            representation,
            source_start,
            source_end,
            padding=padding,
            blocking=blocking,
        )

        return (
            "original-media",
            media_path,
            actual_start,
            actual_end,
        )

    path = external_representation_path(
        representation
    )

    requested_start = max(
        0.0,
        source_start - padding,
    )
    requested_end = (
        source_end + padding
    )

    actual_start, actual_end = (
        play_file_region(
            path,
            requested_start,
            requested_end,
        )
    )

    if blocking:
        wait()

    return (
        name,
        path,
        actual_start,
        actual_end,
    )

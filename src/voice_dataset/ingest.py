from __future__ import annotations

from pathlib import Path

from .media import (
    choose_channel_mode,
    normalize_media,
    probe_audio_streams,
    select_audio_stream,
)
from .schema import AudioRepresentation, SourceRecord
from .sources import probe_audio_representation
from .storage import DatasetStorage


def ingest(
    source: Path,
    output: Path,
    source_id: str,
    audio_stream: int | None = None,
    language: str | None = None,
    channel: str = "auto",
    start: float | None = None,
    duration: float | None = None,
) -> SourceRecord:
    source = source.resolve()
    output = output.resolve()

    if not source.is_file():
        raise FileNotFoundError(source)

    if not source_id.strip():
        raise ValueError("source_id must not be empty")

    streams = probe_audio_streams(source)

    if not streams:
        raise RuntimeError(
            f"No audio streams found in {source}"
        )

    audio_stream, selected_stream = select_audio_stream(
        streams,
        audio_stream=audio_stream,
        language=language,
    )

    selected_stream = streams[audio_stream]

    channel_mode = choose_channel_mode(
        selected_stream,
        requested=channel,
    )

    print(
        f"Selected audio stream {audio_stream}: "
        f"{selected_stream.codec}, "
        f"{selected_stream.channels}ch, "
        f"{selected_stream.channel_layout or 'unknown layout'}, "
        f"language={selected_stream.language or 'unknown'}"
    )

    print(f"Channel mode: {channel_mode}")

    storage = DatasetStorage(output)

    if storage.get_source(source_id) is not None:
        raise ValueError(
            f"Source already exists: {source_id}"
        )

    representation_name = channel_mode

    audio_path = (
        output
        / "audio"
        / source_id
        / f"{representation_name}.wav"
    )

    if audio_path.exists():
        raise ValueError(
            "Source audio already exists without a registered "
            f"source: {audio_path}"
        )

    audio_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Normalizing media: {source.name}",
        flush=True,
    )

    normalize_media(
        source=source,
        destination=audio_path,
        stream_index=selected_stream.index,
        channel_mode=channel_mode,
        start=start,
        duration=duration,
    )

    try:
        audio_info = probe_audio_representation(
            audio_path
        )

        representation = AudioRepresentation(
            path=str(
                audio_path.relative_to(output)
            ),
            kind=representation_name,
            sample_rate=audio_info["sample_rate"],
            channels=audio_info["channels"],
            duration=audio_info["duration"],
            media_start=start or 0.0,
            stream_index=selected_stream.index,
            channel_mode=channel_mode,
            purposes=[
                "speaker_embedding",
                "boundary_analysis",
                "context",
            ],
            metadata={
                "audio_stream": audio_stream,
                "codec": selected_stream.codec,
                "source_channels": selected_stream.channels,
                "source_channel_layout": (
                    selected_stream.channel_layout
                ),
                "source_language": (
                    selected_stream.language
                ),
            },
        )

        record = SourceRecord(
            id=source_id,
            media_path=str(source),
            representations={
                representation_name: representation,
            },
        )

        storage.add_source(record)

    except Exception:
        audio_path.unlink(missing_ok=True)
        raise

    return record

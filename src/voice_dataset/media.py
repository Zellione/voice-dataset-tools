import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


WORK_SAMPLE_RATE = 48000


@dataclass
class AudioStream:
    index: int
    codec: str | None
    sample_rate: int | None
    channels: int | None
    channel_layout: str | None
    language: str | None
    title: str | None


def probe_audio_streams(source: Path) -> list[AudioStream]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index,codec_name,sample_rate,channels,channel_layout:stream_tags=language,title",
        "-of",
        "json",
        str(source),
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffprobe was not found in PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffprobe failed to inspect {source}"
        ) from exc

    data = json.loads(result.stdout)

    streams = []

    for stream in data.get("streams", []):
        tags = stream.get("tags", {})

        streams.append(
            AudioStream(
                index=stream["index"],
                codec=stream.get("codec_name"),
                sample_rate=(
                    int(stream["sample_rate"])
                    if stream.get("sample_rate")
                    else None
                ),
                channels=stream.get("channels"),
                channel_layout=stream.get("channel_layout"),
                language=tags.get("language"),
                title=tags.get("title"),
            )
        )

    return streams

def choose_channel_mode(
    stream: AudioStream,
    requested: str = "auto",
) -> str:
    if requested != "auto":
        return requested

    layout = (stream.channel_layout or "").lower()

    if "5.1" in layout or "7.1" in layout:
        return "center"

    return "mono"

def normalize_media(
    source: Path,
    destination: Path,
    stream_index: int,
    channel_mode: str = "mono",
    start: float | None = None,
    duration: float | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    
    if start is not None:
        command.extend([
            "-ss",
            str(start),
        ])
    
    command.extend([
        "-i",
        str(source),
    ])
    
    if duration is not None:
        command.extend([
            "-t",
            str(duration),
        ])
    
    command.extend([
        "-map",
        f"0:{stream_index}",
        "-vn",
    ])

    if channel_mode == "center":
        command.extend([
            "-af",
            "pan=mono|c0=FC",
        ])
    elif channel_mode == "mono":
        command.extend([
            "-ac",
            "1",
        ])
    else:
        raise ValueError(
            f"Unsupported channel mode: {channel_mode}"
        )

    command.extend([
        "-ar",
        str(WORK_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(destination),
    ])

    try:
        subprocess.run(
            command,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg was not found in PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg failed to decode {source}"
        ) from exc


def extract_audio_region(
    source: Path,
    destination: Path,
    start: float,
    end: float,
) -> None:
    if start < 0:
        raise ValueError(
            "Region start must not be negative"
        )

    if end <= start:
        raise ValueError(
            "Region end must be greater than start"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(start),
        "-i",
        str(source),
        "-t",
        str(end - start),
        "-vn",
        "-acodec",
        "pcm_s16le",
        str(destination),
    ]

    try:
        subprocess.run(
            command,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg was not found in PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg failed to extract region "
            f"{start:.3f}-{end:.3f} "
            f"from {source}"
        ) from exc

def extract_media_audio_region(
    source: Path,
    destination: Path,
    start: float,
    end: float,
    *,
    stream_index: int,
    channel_mode: str = "mono",
) -> None:
    if start < 0:
        raise ValueError(
            "Region start must not be negative"
        )

    if end <= start:
        raise ValueError(
            "Region end must be greater than start"
        )

    normalize_media(
        source=source,
        destination=destination,
        stream_index=stream_index,
        channel_mode=channel_mode,
        start=start,
        duration=end - start,
    )

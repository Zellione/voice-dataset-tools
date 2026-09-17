import hashlib
import json
import tempfile
from pathlib import Path
from .media import (
    choose_channel_mode,
    normalize_media,
    probe_audio_streams,
)

import soundfile as sf

from .asr import Transcriber
from .boundaries import detect_speech_regions, reconcile_boundaries
from .media import normalize_media


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def load_metadata(path: Path) -> list[dict]:
    if not path.exists():
        return []

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def timestamp(value: float) -> float:
    return round(value, 6)


def ingest(
    source: Path,
    output: Path,
    speaker: str,
    character: str,
    language: str | None = None,
    audio_stream: int = 0,
    channel: str = "auto",
    start: float | None = None,
    duration: float | None = None,
):
    source = source.resolve()

    if not source.is_file():
        raise FileNotFoundError(source)

    streams = probe_audio_streams(source)
    
    if not streams:
        raise RuntimeError(
            f"No audio streams found in {source}"
        )

    if audio_stream < 0 or audio_stream >= len(streams):
        raise ValueError(
            f"Audio stream {audio_stream} does not exist. "
            f"Source contains {len(streams)} audio stream(s)."
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
    
    print(
        f"Channel mode: {channel_mode}"
    )

    output.mkdir(parents=True, exist_ok=True)

    clips_dir = output / "clips"
    clips_dir.mkdir(exist_ok=True)

    metadata_path = output / "metadata.jsonl"

    # Hash the original source, not the normalized temporary WAV.
    source_id = hash_file(source)
    existing_records = load_metadata(metadata_path)

    already_imported = any(
        record.get("source_id") == source_id
        and record.get("speaker_id") == speaker
        and record.get("extraction", {}).get("audio_stream") == audio_stream
        and record.get("extraction", {}).get("requested_start") == start
        and record.get("extraction", {}).get("requested_duration") == duration
        and record.get("extraction", {}).get("channel_mode") == channel_mode
        for record in existing_records
    )

    if already_imported:
        print(
            f"Source already imported for speaker {speaker}: "
            f"{source}"
        )
        return

    next_index = (
        max(
            (
                int(record["id"].rsplit("_", 1)[1])
                for record in existing_records
                if record.get("speaker_id") == speaker
                and record.get("id", "").startswith(f"{speaker}_")
            ),
            default=0,
        )
        + 1
    )

    new_records = []

    # Everything below operates on normalized working audio.
    # The temporary directory disappears automatically after ingest.
    with tempfile.TemporaryDirectory(
        prefix="voice-dataset-"
    ) as temp_dir:
        work_audio = Path(temp_dir) / "source.wav"

        print(
            f"Normalizing media: {source.name}",
            flush=True,
        )

        normalize_media(
            source=source,
            destination=work_audio,
            audio_stream=audio_stream,
            channel_mode=channel_mode,
            start=start,
            duration=duration,
        )

        audio, sample_rate = sf.read(work_audio)

        if audio.ndim == 1:
            channels = 1
        else:
            channels = audio.shape[1]

        audio_duration = len(audio) / sample_rate

        transcriber = Transcriber()

        detected_language, segments = transcriber.transcribe(
            str(work_audio),
            language=language,
        )

        speech_regions = detect_speech_regions(
            str(work_audio)
        )

        clip_bounds = reconcile_boundaries(
            segments=segments,
            speech_regions=speech_regions,
            audio_duration=audio_duration,
        )

        source_offset = start or 0.0
        for offset, (segment, bounds) in enumerate(
            zip(segments, clip_bounds)
        ):
            index = next_index + offset

            clip_id = f"{speaker}_{index:05d}"
            filename = f"{clip_id}.wav"
            clip_path = clips_dir / filename

            start_sample = round(
                bounds.start * sample_rate
            )
            end_sample = round(
                bounds.end * sample_rate
            )

            clip = audio[start_sample:end_sample]

            sf.write(
                clip_path,
                clip,
                sample_rate,
                subtype="PCM_16",
            )

            record = {
                "schema_version": 1,

                "id": clip_id,
                "speaker_id": speaker,
                "character": character,
                "language": detected_language,

                "transcript": segment.text,

                "source_id": source_id,
                "source": str(source),

                "extraction": {
                    "audio_stream": audio_stream,
                    "stream_index": selected_stream.index,
                    "codec": selected_stream.codec,
                    "source_channels": selected_stream.channels,
                    "source_channel_layout": selected_stream.channel_layout,
                    "source_language": selected_stream.language,
                    "channel_mode": channel_mode,
                    "requested_start": start,
                    "requested_duration": duration,
                },

                "source_start": timestamp(
                    source_offset + bounds.start
                ),
                "source_end": timestamp(
                    source_offset + bounds.end
                ),
                
                "asr_start": timestamp(
                    source_offset + segment.start
                ),
                "asr_end": timestamp(
                    source_offset + segment.end
                ),

                "duration": timestamp(
                    bounds.end - bounds.start
                ),
                "sample_rate": sample_rate,
                "channels": channels,

                "audio": f"clips/{filename}",

                "review": {
                    "status": "pending",
                },
            }

            new_records.append(record)

            print(
                f"{filename}: "
                f"ASR [{segment.start:.2f}-{segment.end:.2f}] "
                f"CLIP [{bounds.start:.2f}-{bounds.end:.2f}] "
                f"{segment.text}"
            )

    # Only append metadata after media processing completed successfully.
    with metadata_path.open(
        "a",
        encoding="utf-8",
    ) as metadata:
        for record in new_records:
            metadata.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print(
        f"Imported {len(new_records)} clips "
        f"from {source.name} into {output}"
    )

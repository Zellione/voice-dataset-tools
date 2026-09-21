from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .asr import Transcriber
from .representations import materialize_regions
from .sources import resolve_source_representation
from .storage import DatasetStorage
from .transcripts import attach_transcript


WHISPER_MODEL = "large-v3"


@dataclass(frozen=True)
class TranscribeRegionsResult:
    transcribed: int
    skipped: int


def transcribe_regions(
    storage: DatasetStorage,
    source_id: str,
    *,
    representation_name: str,
    transcript_name: str,
    language: str | None = None,
) -> TranscribeRegionsResult:
    source_representation, source_path = (
        resolve_source_representation(
            storage,
            source_id,
            representation_name,
        )
    )

    kind = source_representation.get("kind")

    if not isinstance(kind, str) or not kind:
        raise ValueError(
            f"Invalid source representation kind: "
            f"{source_id}/{representation_name}"
        )

    materialize_regions(
        storage=storage,
        source_id=source_id,
        source=source_path,
        representation_name=representation_name,
        kind=kind,
        purposes=["asr"],
    )

    regions = [
        region
        for region in storage.regions.load()
        if region.get("source_id") == source_id
    ]

    pending: list[tuple[str, Path]] = []
    skipped = 0

    for region in regions:
        region_id = region.get("id")

        if not isinstance(region_id, str) or not region_id:
            raise ValueError(
                "Candidate region has invalid id"
            )

        transcripts = region.get("transcripts")

        if not isinstance(transcripts, dict):
            raise ValueError(
                f"Invalid transcripts for {region_id}"
            )

        if transcript_name in transcripts:
            skipped += 1
            continue

        representations = region.get(
            "representations"
        )

        if not isinstance(representations, dict):
            raise ValueError(
                f"Invalid representations for {region_id}"
            )

        representation = representations.get(
            representation_name
        )

        if not isinstance(representation, dict):
            raise ValueError(
                f"Missing representation: "
                f"{region_id}/{representation_name}"
            )

        path_value = representation.get("path")

        if not isinstance(path_value, str) or not path_value:
            raise ValueError(
                f"Invalid representation path: "
                f"{region_id}/{representation_name}"
            )

        path = Path(path_value)

        if not path.is_absolute():
            path = storage.root / path

        if not path.is_file():
            raise ValueError(
                f"Representation audio does not exist: {path}"
            )

        pending.append(
            (region_id, path)
        )

    if not pending:
        return TranscribeRegionsResult(
            transcribed=0,
            skipped=skipped,
        )

    transcriber = Transcriber(
        model_name=WHISPER_MODEL
    )

    transcribed = 0

    for region_id, path in pending:
        detected_language, segments = (
            transcriber.transcribe(
                str(path),
                language=language,
            )
        )

        text = " ".join(
            segment.text
            for segment in segments
            if segment.text
        ).strip()

        attach_transcript(
            storage,
            region_id,
            transcript_name,
            text=text or None,
            language=detected_language,
            model=WHISPER_MODEL,
            representation=representation_name,
            metadata={
                "language_override": language,
                "segments": [
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                    }
                    for segment in segments
                ],
            },
        )

        transcribed += 1

        print(
            f"{region_id}: "
            f"[{detected_language}] "
            f"{text or '<no speech>'}"
        )

    return TranscribeRegionsResult(
        transcribed=transcribed,
        skipped=skipped,
    )

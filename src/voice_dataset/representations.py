from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .media import (
    extract_audio_region,
    probe_audio_streams,
)
from .schema import AudioRepresentation
from .storage import DatasetStorage


TURN_BOUNDARY_PADDING_SECONDS = 0.250
BOUNDARY_PADDED_TURN_PURPOSES = {
    "review",
    "tts_candidate",
}

BOUNDARY_SIGNAL_WINDOW_SECONDS = 0.030
BOUNDARY_SIGNAL_MINIMUM_RMS = 0.003
BOUNDARY_SIGNAL_MINIMUM_CONTINUITY = 0.25


def _boundary_requires_padding(
    source: Path,
    boundary: float,
    *,
    side: str,
    window: float = BOUNDARY_SIGNAL_WINDOW_SECONDS,
    minimum_rms: float = BOUNDARY_SIGNAL_MINIMUM_RMS,
    minimum_continuity: float = (
        BOUNDARY_SIGNAL_MINIMUM_CONTINUITY
    ),
) -> bool:
    if side not in {"start", "end"}:
        raise ValueError(
            f"Unsupported boundary side: {side}"
        )

    if window <= 0:
        raise ValueError(
            "window must be positive"
        )

    if minimum_rms < 0:
        raise ValueError(
            "minimum_rms must not be negative"
        )

    if not (
        0.0
        <= minimum_continuity
        <= 1.0
    ):
        raise ValueError(
            "minimum_continuity must be between "
            "zero and one"
        )

    source = source.resolve()

    if not source.is_file():
        raise FileNotFoundError(source)

    with sf.SoundFile(source) as audio:
        sample_rate = int(audio.samplerate)
        total_frames = int(audio.frames)

        if sample_rate <= 0:
            raise ValueError(
                "Audio has invalid sample rate"
            )

        boundary_frame = int(
            round(
                float(boundary)
                * sample_rate
            )
        )

        boundary_frame = max(
            0,
            min(
                total_frames,
                boundary_frame,
            ),
        )

        window_frames = max(
            1,
            int(
                round(
                    window
                    * sample_rate
                )
            ),
        )

        left_start = max(
            0,
            boundary_frame - window_frames,
        )
        left_count = (
            boundary_frame - left_start
        )

        right_end = min(
            total_frames,
            boundary_frame + window_frames,
        )
        right_count = (
            right_end - boundary_frame
        )

        if (
            left_count <= 0
            or right_count <= 0
        ):
            return False

        audio.seek(left_start)

        left = audio.read(
            left_count,
            dtype="float32",
            always_2d=True,
        )

        audio.seek(boundary_frame)

        right = audio.read(
            right_count,
            dtype="float32",
            always_2d=True,
        )

    if (
        left.size == 0
        or right.size == 0
    ):
        return False

    # Compute RMS across all samples and channels
    # directly. Averaging channels first could cancel
    # valid signal in anti-phase material.
    left_rms = float(
        np.sqrt(
            np.mean(
                np.square(
                    left,
                    dtype=np.float32,
                )
            )
        )
    )

    right_rms = float(
        np.sqrt(
            np.mean(
                np.square(
                    right,
                    dtype=np.float32,
                )
            )
        )
    )

    peak_rms = max(
        left_rms,
        right_rms,
    )

    if peak_rms < minimum_rms:
        return False

    continuity = (
        min(
            left_rms,
            right_rms,
        )
        / peak_rms
    )

    return (
        continuity
        >= minimum_continuity
    )


@dataclass
class MaterializeRegionsResult:
    created: int
    skipped: int


@dataclass
class MaterializeTurnsResult:
    created: int
    skipped: int


def probe_representation_source(
    source: Path,
) -> tuple[int, int]:
    streams = probe_audio_streams(
        source
    )

    if len(streams) != 1:
        raise ValueError(
            "Representation source must contain "
            "exactly one audio stream; "
            f"found {len(streams)}: {source}"
        )

    stream = streams[0]

    if stream.sample_rate is None:
        raise ValueError(
            "Representation source has no "
            f"sample rate: {source}"
        )

    if stream.channels is None:
        raise ValueError(
            "Representation source has no "
            f"channel count: {source}"
        )

    return (
        stream.sample_rate,
        stream.channels,
    )


def materialize_region_audio(
    storage: DatasetStorage,
    region_id: str,
    source: Path,
    representation_name: str,
    kind: str,
    purposes: list[str],
    sample_rate: int | None = None,
    channels: int | None = None,
    replace_existing: bool = False,
) -> AudioRepresentation:
    source = source.resolve()

    if not source.is_file():
        raise ValueError(
            f"Audio source does not exist: {source}"
        )

    if (
        sample_rate is None
        or channels is None
    ):
        (
            detected_sample_rate,
            detected_channels,
        ) = probe_representation_source(
            source
        )

        if sample_rate is None:
            sample_rate = (
                detected_sample_rate
            )

        if channels is None:
            channels = detected_channels

    region = storage.get_region(
        region_id
    )

    if region is None:
        raise KeyError(
            f"Region does not exist: {region_id}"
        )

    representations = region.get(
        "representations",
        {},
    )

    if not isinstance(
        representations,
        dict,
    ):
        raise ValueError(
            f"Invalid representations "
            f"for {region_id}"
        )

    existing = representations.get(
        representation_name
    )

    if (
        existing is not None
        and not replace_existing
    ):
        raise ValueError(
            f"Representation already exists: "
            f"{region_id}/{representation_name}"
        )

    start = region.get(
        "source_start"
    )

    end = region.get(
        "source_end"
    )

    if not isinstance(
        start,
        (int, float),
    ):
        raise ValueError(
            f"Invalid source_start for {region_id}"
        )

    if not isinstance(
        end,
        (int, float),
    ):
        raise ValueError(
            f"Invalid source_end for {region_id}"
        )

    canonical_start = float(start)
    canonical_end = float(end)

    if (
        canonical_start < 0
        or canonical_end <= canonical_start
    ):
        raise ValueError(
            f"Invalid source range for {region_id}: "
            f"{canonical_start}-{canonical_end}"
        )

    relative_path = (
        Path("regions")
        / region_id
        / f"{representation_name}.wav"
    )

    destination = (
        storage.root
        / relative_path
    )

    temporary = destination.with_name(
        f".{destination.name}.tmp.wav"
    )

    backup = destination.with_name(
        f".{destination.name}.backup"
    )

    if (
        destination.exists()
        and not replace_existing
    ):
        raise ValueError(
            f"Representation file already exists: "
            f"{destination}"
        )

    if (
        replace_existing
        and existing is None
    ):
        raise ValueError(
            "Cannot replace missing representation: "
            f"{region_id}/{representation_name}"
        )

    stale_embedding_names: list[str] = []
    invalidated_embedding_paths: list[Path] = []
    stale_transcript_names: list[str] = []

    if replace_existing:
        transcripts = region.get(
            "transcripts"
        )

        if not isinstance(
            transcripts,
            dict,
        ):
            raise ValueError(
                f"Invalid transcripts for {region_id}"
            )

        for (
            transcript_name,
            transcript,
        ) in transcripts.items():
            if not isinstance(
                transcript,
                dict,
            ):
                raise ValueError(
                    f"Invalid transcript "
                    f"{region_id}/{transcript_name}"
                )

            if (
                transcript.get("representation")
                == representation_name
            ):
                stale_transcript_names.append(
                    transcript_name
                )

        embeddings = region.get(
            "embeddings"
        )

        if not isinstance(
            embeddings,
            dict,
        ):
            raise ValueError(
                f"Invalid embeddings for {region_id}"
            )

        for (
            embedding_name,
            embedding,
        ) in embeddings.items():
            if not isinstance(
                embedding,
                dict,
            ):
                raise ValueError(
                    f"Invalid embedding "
                    f"{region_id}/{embedding_name}"
                )

            if (
                embedding.get("representation")
                != representation_name
            ):
                continue

            embedding_path = embedding.get(
                "path"
            )

            if not isinstance(
                embedding_path,
                str,
            ) or not embedding_path:
                raise ValueError(
                    f"Invalid embedding path "
                    f"{region_id}/{embedding_name}"
                )

            stale_embedding_names.append(
                embedding_name
            )

            invalidated_embedding_paths.append(
                storage.root / embedding_path
            )

    temporary.unlink(
        missing_ok=True
    )

    if backup.exists():
        raise ValueError(
            "Representation backup already exists: "
            f"{backup}"
        )

    backup_created = False
    destination_replaced = False

    try:
        extract_audio_region(
            source=source,
            destination=temporary,
            start=canonical_start,
            end=canonical_end,
        )

        if replace_existing:
            if not destination.is_file():
                raise ValueError(
                    "Representation metadata exists "
                    "but file is missing: "
                    f"{destination}"
                )

            os.replace(
                destination,
                backup,
            )
            backup_created = True

        os.replace(
            temporary,
            destination,
        )
        destination_replaced = True

        representation = AudioRepresentation(
            path=relative_path.as_posix(),
            kind=kind,
            sample_rate=sample_rate,
            channels=channels,
            purposes=list(purposes),
            metadata={
                "source": str(source),
                "canonical_start": canonical_start,
                "canonical_end": canonical_end,
            },
        )

        def update(
            record: dict[str, Any],
        ) -> dict[str, Any]:
            current = record.get(
                "representations"
            )

            if not isinstance(
                current,
                dict,
            ):
                raise ValueError(
                    f"Invalid representations "
                    f"for {region_id}"
                )

            if (
                representation_name in current
                and not replace_existing
            ):
                raise ValueError(
                    f"Representation already exists: "
                    f"{region_id}/"
                    f"{representation_name}"
                )

            current[
                representation_name
            ] = representation.to_dict()

            if replace_existing:
                transcripts = record.get(
                    "transcripts"
                )

                if not isinstance(
                    transcripts,
                    dict,
                ):
                    raise ValueError(
                        f"Invalid transcripts "
                        f"for {region_id}"
                    )

                for transcript_name in (
                    stale_transcript_names
                ):
                    transcripts.pop(
                        transcript_name,
                        None,
                    )

                embeddings = record.get(
                    "embeddings"
                )

                if not isinstance(
                    embeddings,
                    dict,
                ):
                    raise ValueError(
                        f"Invalid embeddings "
                        f"for {region_id}"
                    )

                for embedding_name in (
                    stale_embedding_names
                ):
                    embeddings.pop(
                        embedding_name,
                        None,
                    )

            return record

        storage.update_region(
            region_id,
            update,
        )

    except Exception:
        temporary.unlink(
            missing_ok=True
        )

        if destination_replaced:
            destination.unlink(
                missing_ok=True
            )

        if backup_created:
            os.replace(
                backup,
                destination,
            )

        raise

    backup.unlink(
        missing_ok=True
    )

    for embedding_path in invalidated_embedding_paths:
        try:
            embedding_path.unlink(
                missing_ok=True
            )
        except OSError:
            pass

    return representation


def materialize_regions(
    storage: DatasetStorage,
    source_id: str,
    source: Path,
    representation_name: str,
    kind: str,
    purposes: list[str],
) -> MaterializeRegionsResult:
    source = source.resolve()

    if not source.is_file():
        raise ValueError(
            f"Audio source does not exist: {source}"
        )

    (
        sample_rate,
        channels,
    ) = probe_representation_source(
        source
    )

    regions = storage.regions.load()

    created = 0
    skipped = 0

    for region in regions:
        if region.get("source_id") != source_id:
            continue

        region_id = region.get("id")

        if not isinstance(
            region_id,
            str,
        ) or not region_id:
            raise ValueError(
                "Region has invalid id"
            )

        representations = region.get(
            "representations"
        )

        if not isinstance(
            representations,
            dict,
        ):
            raise ValueError(
                f"Invalid representations "
                f"for {region_id}"
            )

        existing = representations.get(
            representation_name
        )

        expected_relative_path = (
            Path("regions")
            / region_id
            / f"{representation_name}.wav"
        )

        expected_destination = (
            storage.root
            / expected_relative_path
        )

        if existing is not None:
            if not isinstance(
                existing,
                dict,
            ):
                raise ValueError(
                    f"Invalid representation "
                    f"{region_id}/"
                    f"{representation_name}"
                )

            existing_path = existing.get(
                "path"
            )

            if (
                existing_path
                != expected_relative_path.as_posix()
            ):
                raise ValueError(
                    f"Unexpected representation path "
                    f"for {region_id}/"
                    f"{representation_name}: "
                    f"{existing_path}"
                )

            if not expected_destination.is_file():
                raise ValueError(
                    f"Representation metadata exists "
                    f"but file is missing: "
                    f"{expected_destination}"
                )

            metadata = existing.get(
                "metadata"
            )

            stale = True

            if isinstance(metadata, dict):
                existing_start = metadata.get(
                    "canonical_start"
                )
                existing_end = metadata.get(
                    "canonical_end"
                )

                if (
                    isinstance(
                        existing_start,
                        (int, float),
                    )
                    and isinstance(
                        existing_end,
                        (int, float),
                    )
                    and float(existing_start)
                    == float(region["source_start"])
                    and float(existing_end)
                    == float(region["source_end"])
                ):
                    stale = False

                # Legacy region representations did
                # not persist their geometry. Preserve
                # them rather than forcing a rebuild
                # of existing datasets.
                elif (
                    "canonical_start"
                    not in metadata
                    and "canonical_end"
                    not in metadata
                ):
                    stale = False

            if not stale:
                skipped += 1
                continue

            materialize_region_audio(
                storage=storage,
                region_id=region_id,
                source=source,
                representation_name=(
                    representation_name
                ),
                kind=kind,
                purposes=purposes,
                sample_rate=sample_rate,
                channels=channels,
                replace_existing=True,
            )

            created += 1
            continue

        if expected_destination.exists():
            raise ValueError(
                f"Representation file exists "
                f"without metadata: "
                f"{expected_destination}"
            )

        materialize_region_audio(
            storage=storage,
            region_id=region_id,
            source=source,
            representation_name=(
                representation_name
            ),
            kind=kind,
            purposes=purposes,
            sample_rate=sample_rate,
            channels=channels,
        )

        created += 1

    return MaterializeRegionsResult(
        created=created,
        skipped=skipped,
    )


def materialize_turn_audio(
    storage: DatasetStorage,
    turn_id: str,
    source: Path,
    representation_name: str,
    kind: str,
    purposes: list[str],
    sample_rate: int | None = None,
    channels: int | None = None,
    clip_start: float | None = None,
    clip_end: float | None = None,
    replace_existing: bool = False,
) -> AudioRepresentation:
    source = source.resolve()

    if not source.is_file():
        raise ValueError(
            f"Audio source does not exist: {source}"
        )

    if (
        sample_rate is None
        or channels is None
    ):
        (
            detected_sample_rate,
            detected_channels,
        ) = probe_representation_source(
            source
        )

        if sample_rate is None:
            sample_rate = detected_sample_rate

        if channels is None:
            channels = detected_channels

    turn = storage.get_turn(turn_id)

    if turn is None:
        raise KeyError(
            f"Turn does not exist: {turn_id}"
        )

    representations = turn.get(
        "representations",
        {},
    )

    if not isinstance(
        representations,
        dict,
    ):
        raise ValueError(
            f"Invalid representations for {turn_id}"
        )

    existing = representations.get(
        representation_name
    )

    if (
        existing is not None
        and not replace_existing
    ):
        raise ValueError(
            f"Representation already exists: "
            f"{turn_id}/{representation_name}"
        )

    start = turn.get("source_start")
    end = turn.get("source_end")

    if not isinstance(
        start,
        (int, float),
    ):
        raise ValueError(
            f"Invalid source_start for {turn_id}"
        )

    if not isinstance(
        end,
        (int, float),
    ):
        raise ValueError(
            f"Invalid source_end for {turn_id}"
        )

    canonical_start = float(start)
    canonical_end = float(end)

    if (
        canonical_start < 0
        or canonical_end <= canonical_start
    ):
        raise ValueError(
            f"Invalid source range for {turn_id}: "
            f"{canonical_start}-{canonical_end}"
        )

    actual_start = (
        canonical_start
        if clip_start is None
        else float(clip_start)
    )
    actual_end = (
        canonical_end
        if clip_end is None
        else float(clip_end)
    )

    if (
        actual_start < 0
        or actual_start > canonical_start
        or actual_end < canonical_end
        or actual_end <= actual_start
    ):
        raise ValueError(
            f"Invalid clip range for {turn_id}: "
            f"{actual_start}-{actual_end}; "
            f"canonical range is "
            f"{canonical_start}-{canonical_end}"
        )

    relative_path = (
        Path("turns")
        / turn_id
        / f"{representation_name}.wav"
    )

    destination = (
        storage.root
        / relative_path
    )

    temporary = destination.with_name(
        f".{destination.name}.tmp.wav"
    )

    backup = destination.with_name(
        f".{destination.name}.backup"
    )

    if (
        destination.exists()
        and not replace_existing
    ):
        raise ValueError(
            f"Representation file already exists: "
            f"{destination}"
        )

    if (
        replace_existing
        and existing is None
    ):
        raise ValueError(
            "Cannot replace missing representation: "
            f"{turn_id}/{representation_name}"
        )

    stale_embedding_names: list[str] = []
    invalidated_embedding_paths: list[Path] = []

    if replace_existing:
        embeddings = turn.get(
            "embeddings"
        )

        if not isinstance(
            embeddings,
            dict,
        ):
            raise ValueError(
                f"Invalid embeddings for {turn_id}"
            )

        for (
            embedding_name,
            embedding,
        ) in embeddings.items():
            if not isinstance(
                embedding,
                dict,
            ):
                raise ValueError(
                    f"Invalid embedding "
                    f"{turn_id}/{embedding_name}"
                )

            if (
                embedding.get("representation")
                != representation_name
            ):
                continue

            path = embedding.get("path")

            if not isinstance(
                path,
                str,
            ) or not path:
                raise ValueError(
                    f"Invalid embedding path "
                    f"{turn_id}/{embedding_name}"
                )

            stale_embedding_names.append(
                embedding_name
            )
            invalidated_embedding_paths.append(
                storage.root / path
            )

    representation = AudioRepresentation(
        path=relative_path.as_posix(),
        kind=kind,
        sample_rate=sample_rate,
        channels=channels,
        purposes=list(purposes),
        metadata={
            "source": str(source),
            "canonical_start": canonical_start,
            "canonical_end": canonical_end,
            "clip_start": actual_start,
            "clip_end": actual_end,
            "padding_before": (
                canonical_start - actual_start
            ),
            "padding_after": (
                actual_end - canonical_end
            ),
        },
    )

    temporary.unlink(
        missing_ok=True
    )

    if backup.exists():
        raise ValueError(
            "Representation backup already exists: "
            f"{backup}"
        )

    backup_created = False
    destination_replaced = False

    try:
        extract_audio_region(
            source=source,
            destination=temporary,
            start=actual_start,
            end=actual_end,
        )

        if replace_existing:
            if not destination.is_file():
                raise ValueError(
                    "Representation metadata exists "
                    "but file is missing: "
                    f"{destination}"
                )

            os.replace(
                destination,
                backup,
            )
            backup_created = True

        os.replace(
            temporary,
            destination,
        )
        destination_replaced = True

        def update(
            record: dict[str, Any],
        ) -> dict[str, Any]:
            current = record.get(
                "representations"
            )

            if not isinstance(
                current,
                dict,
            ):
                raise ValueError(
                    f"Invalid representations "
                    f"for {turn_id}"
                )

            if (
                representation_name in current
                and not replace_existing
            ):
                raise ValueError(
                    f"Representation already exists: "
                    f"{turn_id}/"
                    f"{representation_name}"
                )

            current[
                representation_name
            ] = representation.to_dict()

            if replace_existing:
                embeddings = record.get(
                    "embeddings"
                )

                if not isinstance(
                    embeddings,
                    dict,
                ):
                    raise ValueError(
                        f"Invalid embeddings "
                        f"for {turn_id}"
                    )

                for embedding_name in (
                    stale_embedding_names
                ):
                    embeddings.pop(
                        embedding_name,
                        None,
                    )

            return record

        storage.update_turn(
            turn_id,
            update,
        )

    except Exception:
        temporary.unlink(
            missing_ok=True
        )

        if destination_replaced:
            destination.unlink(
                missing_ok=True
            )

        if backup_created:
            os.replace(
                backup,
                destination,
            )

        raise

    backup.unlink(
        missing_ok=True
    )

    for path in invalidated_embedding_paths:
        try:
            path.unlink(
                missing_ok=True
            )
        except OSError:
            # Metadata no longer references the stale
            # embedding. A leftover file is harmless
            # and can be cleaned up separately.
            pass

    return representation


def materialize_turns(
    storage: DatasetStorage,
    source_id: str,
    source: Path,
    representation_name: str,
    kind: str,
    purposes: list[str],
) -> MaterializeTurnsResult:
    source = source.resolve()

    if not source.is_file():
        raise ValueError(
            f"Audio source does not exist: {source}"
        )

    (
        sample_rate,
        channels,
    ) = probe_representation_source(
        source
    )

    turns = storage.turns.load()

    source_turns = [
        turn
        for turn in turns
        if turn.get("source_id") == source_id
    ]

    source_turns.sort(
        key=lambda turn: (
            float(turn.get("source_start", 0.0)),
            float(turn.get("source_end", 0.0)),
            str(turn.get("id", "")),
        )
    )

    purpose_uses_boundary_padding = bool(
        BOUNDARY_PADDED_TURN_PURPOSES.intersection(
            purposes
        )
    )

    created = 0
    skipped = 0

    for index, turn in enumerate(source_turns):
        turn_id = turn.get("id")

        if not isinstance(
            turn_id,
            str,
        ) or not turn_id:
            raise ValueError(
                "Turn has invalid id"
            )

        start = turn.get("source_start")
        end = turn.get("source_end")

        if not isinstance(
            start,
            (int, float),
        ):
            raise ValueError(
                f"Invalid source_start for {turn_id}"
            )

        if not isinstance(
            end,
            (int, float),
        ):
            raise ValueError(
                f"Invalid source_end for {turn_id}"
            )

        canonical_start = float(start)
        canonical_end = float(end)

        if (
            canonical_start < 0
            or canonical_end <= canonical_start
        ):
            raise ValueError(
                f"Invalid source range for {turn_id}: "
                f"{canonical_start}-{canonical_end}"
            )

        clip_start = canonical_start
        clip_end = canonical_end

        metadata = turn.get("metadata")
        boundary_curation = (
            metadata.get("boundary_curation")
            if isinstance(metadata, dict)
            else None
        )

        manually_curated_boundary = (
            isinstance(boundary_curation, dict)
            and boundary_curation.get("method")
            == "manual"
        )

        if (
            purpose_uses_boundary_padding
            and not manually_curated_boundary
        ):
            pad_start = _boundary_requires_padding(
                source,
                canonical_start,
                side="start",
            )

            pad_end = _boundary_requires_padding(
                source,
                canonical_end,
                side="end",
            )

            if pad_start:
                clip_start = max(
                    0.0,
                    canonical_start
                    - TURN_BOUNDARY_PADDING_SECONDS,
                )

                if index > 0:
                    previous_end = float(
                        source_turns[index - 1][
                            "source_end"
                        ]
                    )

                    if previous_end <= canonical_start:
                        clip_start = max(
                            clip_start,
                            previous_end,
                        )

            if pad_end:
                clip_end = (
                    canonical_end
                    + TURN_BOUNDARY_PADDING_SECONDS
                )

                if index + 1 < len(source_turns):
                    next_start = float(
                        source_turns[index + 1][
                            "source_start"
                        ]
                    )

                    if next_start >= canonical_end:
                        clip_end = min(
                            clip_end,
                            next_start,
                        )

        representations = turn.get(
            "representations"
        )

        if not isinstance(
            representations,
            dict,
        ):
            raise ValueError(
                f"Invalid representations "
                f"for {turn_id}"
            )

        existing = representations.get(
            representation_name
        )

        expected_relative_path = (
            Path("turns")
            / turn_id
            / f"{representation_name}.wav"
        )

        expected_destination = (
            storage.root
            / expected_relative_path
        )

        replace_existing = False

        if existing is not None:
            if not isinstance(
                existing,
                dict,
            ):
                raise ValueError(
                    f"Invalid representation "
                    f"{turn_id}/"
                    f"{representation_name}"
                )

            existing_path = existing.get(
                "path"
            )

            if (
                existing_path
                != expected_relative_path.as_posix()
            ):
                raise ValueError(
                    f"Unexpected representation path "
                    f"for {turn_id}/"
                    f"{representation_name}: "
                    f"{existing_path}"
                )

            if not expected_destination.is_file():
                raise ValueError(
                    f"Representation metadata exists "
                    f"but file is missing: "
                    f"{expected_destination}"
                )

            metadata = existing.get(
                "metadata"
            )

            stale = True

            if isinstance(metadata, dict):
                existing_start = metadata.get(
                    "canonical_start"
                )
                existing_end = metadata.get(
                    "canonical_end"
                )

                existing_clip_start = metadata.get(
                    "clip_start"
                )
                existing_clip_end = metadata.get(
                    "clip_end"
                )

                if (
                    isinstance(
                        existing_start,
                        (int, float),
                    )
                    and isinstance(
                        existing_end,
                        (int, float),
                    )
                    and isinstance(
                        existing_clip_start,
                        (int, float),
                    )
                    and isinstance(
                        existing_clip_end,
                        (int, float),
                    )
                    and float(existing_start)
                    == canonical_start
                    and float(existing_end)
                    == canonical_end
                    and float(existing_clip_start)
                    == clip_start
                    and float(existing_clip_end)
                    == clip_end
                ):
                    stale = False

            if not stale:
                skipped += 1
                continue

            replace_existing = stale

        if (
            expected_destination.exists()
            and not replace_existing
        ):
            raise ValueError(
                f"Representation file exists "
                f"without metadata: "
                f"{expected_destination}"
            )

        materialize_turn_audio(
            storage=storage,
            turn_id=turn_id,
            source=source,
            representation_name=(
                representation_name
            ),
            kind=kind,
            purposes=purposes,
            sample_rate=sample_rate,
            channels=channels,
            clip_start=clip_start,
            clip_end=clip_end,
            replace_existing=replace_existing,
        )

        created += 1

    return MaterializeTurnsResult(
        created=created,
        skipped=skipped,
    )

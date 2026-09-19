from pathlib import Path

import typer

from .cuda import preload_cuda_libraries
from .media import probe_audio_streams
from .representations import (
    materialize_regions,
    materialize_turns,
)
from .reconciliation import (
    create_turn_from_regions,
    edit_turn,
    effective_region_reconciliation,
    reject_region,
    split_turn,
)
from .storage import DatasetStorage
from .sources import set_representation_provenance
from .playback import (
    play_preferred_review_audio,
    play_representation,
    play_turn_context,
    stop,
)
from .review import (
    mark_turn_pending,
    mark_turn_reviewed,
)
from .reviewer import (
    format_turn,
    raw_representation,
    reviewer_help,
    sorted_turns,
)
from .voices import (
    assign_turn,
    create_voice,
    ignore_turn,
    mark_turn_unknown,
    set_voice_ignored,
)
from .detectors import (
    import_detector_regions,
    load_detector_output,
)
from .embeddings import (
    import_embeddings,
    load_embedding_output,
)
from .transcripts import (
    import_transcripts,
    load_transcript_output,
)
from .boundary_evidence import (
    refresh_source_boundary_evidence,
)

preload_cuda_libraries()

from .ingest import ingest as ingest_source


app = typer.Typer(
    help="Build and manage speech datasets for voice model training.",
    no_args_is_help=True,
)

source_app = typer.Typer(
    help="Manage dataset sources and their representations.",
    no_args_is_help=True,
)

voice_app = typer.Typer(
    help="Manage persistent voice profiles.",
    no_args_is_help=True,
)

region_app = typer.Typer(
    help="Manage detector candidate regions.",
    no_args_is_help=True,
)

turn_app = typer.Typer(
    help="Manage speech turns and voice assignments.",
    no_args_is_help=True,
)

app.add_typer(
    source_app,
    name="source",
)

app.add_typer(
    voice_app,
    name="voice",
)

app.add_typer(
    region_app,
    name="region",
)

app.add_typer(
    turn_app,
    name="turn",
)


def storage_for(dataset: Path) -> DatasetStorage:
    return DatasetStorage(
        dataset.resolve()
    )


@app.callback()
def main():
    """Build and manage speech datasets for voice model training."""
    pass


@source_app.command(
    "set-provenance"
)
def source_set_provenance(
    source_id: str,
    representation: str,
    dataset: Path = typer.Option(
        ...,
        help="Dataset root directory.",
    ),
    media_start: float = typer.Option(
        ...,
        help=(
            "Timestamp in the original media that "
            "corresponds to 0.0 seconds in the "
            "representation."
        ),
    ),
    stream_index: int = typer.Option(
        ...,
        help=(
            "Global media container stream index "
            "reported by ffprobe."
        ),
    ),
    channel_mode: str = typer.Option(
        ...,
        help="Source channel extraction mode.",
    ),
):
    """Set original-media provenance for a source representation."""

    storage = storage_for(dataset)

    try:
        updated = set_representation_provenance(
            storage,
            source_id,
            representation,
            media_start=media_start,
            stream_index=stream_index,
            channel_mode=channel_mode,
        )
    except (KeyError, ValueError) as exc:
        raise typer.BadParameter(
            str(exc)
        ) from exc

    provenance = updated[
        "representations"
    ][representation]

    typer.echo(
        f"Updated {source_id}:{representation}"
    )
    typer.echo(
        f"  media_start:  "
        f"{provenance['media_start']}"
    )
    typer.echo(
        f"  stream_index: "
        f"{provenance['stream_index']}"
    )
    typer.echo(
        f"  channel_mode: "
        f"{provenance['channel_mode']}"
    )


@app.command()
def probe(
    source: Path,
):
    """Inspect audio streams in a media source."""

    source = source.resolve()

    if not source.is_file():
        raise typer.BadParameter(
            f"File does not exist: {source}"
        )

    streams = probe_audio_streams(source)

    if not streams:
        typer.echo("No audio streams found.")
        raise typer.Exit()

    typer.echo(f"Source: {source}")
    typer.echo()

    for stream in streams:
        typer.echo(
            f"Audio stream #{stream.index}"
        )
        typer.echo(
            f"  Codec:          "
            f"{stream.codec or '-'}"
        )
        typer.echo(
            f"  Sample rate:    "
            f"{stream.sample_rate or '-'}"
        )
        typer.echo(
            f"  Channels:       "
            f"{stream.channels or '-'}"
        )
        typer.echo(
            f"  Channel layout: "
            f"{stream.channel_layout or '-'}"
        )
        typer.echo(
            f"  Language:       "
            f"{stream.language or '-'}"
        )
        typer.echo(
            f"  Title:          "
            f"{stream.title or '-'}"
        )
        typer.echo()


@app.command()
def ingest(
    source: Path,
    speaker: str = typer.Option(
        ...,
        help="Unique speaker identifier.",
    ),
    character: str = typer.Option(
        ...,
        help="Character represented by the speaker.",
    ),
    language: str | None = typer.Option(
        None,
        help="ASR language override, e.g. en or de.",
    ),
    audio_stream: int = typer.Option(
        0,
        help=(
            "Audio stream number within the audio streams "
            "(0 = first audio stream)."
        ),
    ),
    channel: str = typer.Option(
        "auto",
        help=(
            "Channel extraction mode: "
            "auto, mono, or center."
        ),
    ),
    start: float | None = typer.Option(
        None,
        min=0.0,
        help="Start position in seconds.",
    ),
    duration: float | None = typer.Option(
        None,
        min=0.001,
        help="Maximum duration to ingest in seconds.",
    ),
    output: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset output directory.",
    ),
):
    """Ingest source media into a speech dataset."""

    if channel not in {
        "auto",
        "mono",
        "center",
    }:
        raise typer.BadParameter(
            "--channel must be "
            "auto, mono, or center"
        )

    ingest_source(
        source=source,
        output=output,
        speaker=speaker,
        character=character,
        language=language,
        audio_stream=audio_stream,
        channel=channel,
        start=start,
        duration=duration,
    )


@voice_app.command("create")
def voice_create(
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
    character: str | None = typer.Option(
        None,
        help="Character represented by this voice.",
    ),
    language: str | None = typer.Option(
        None,
        help="Voice language, e.g. en or de.",
    ),
    alias: list[str] | None = typer.Option(
        None,
        "--alias",
        help="Alias. May be specified multiple times.",
    ),
    notes: str | None = typer.Option(
        None,
        help="Optional notes about this voice.",
    ),
    ignore: bool = typer.Option(
        False,
        help="Create this as an ignored voice profile.",
    ),
):
    """Create a persistent voice profile."""

    storage = storage_for(dataset)

    voice = create_voice(
        storage,
        character=character,
        language=language,
        aliases=alias,
        ignored=ignore,
        notes=notes,
    )

    typer.echo(
        f"Created {voice['id']}"
    )

    typer.echo(
        f"  character: "
        f"{voice['character'] or '-'}"
    )

    typer.echo(
        f"  language:  "
        f"{voice['language'] or '-'}"
    )

    typer.echo(
        f"  ignored:   "
        f"{voice['ignored']}"
    )


@voice_app.command("list")
def voice_list(
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """List persistent voice profiles."""

    storage = storage_for(dataset)
    voices = storage.voices.load()

    if not voices:
        typer.echo(
            "No voice profiles."
        )
        return

    for voice in voices:
        character = (
            voice.get("character")
            or "-"
        )

        language = (
            voice.get("language")
            or "-"
        )

        ignored = (
            " ignored"
            if voice.get("ignored")
            else ""
        )

        typer.echo(
            f"{voice['id']}  "
            f"character={character}  "
            f"language={language}"
            f"{ignored}"
        )


@voice_app.command("ignore")
def voice_ignore(
    voice_id: str,
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """Ignore a known voice in future processing."""

    storage = storage_for(dataset)

    try:
        voice = set_voice_ignored(
            storage,
            voice_id,
            True,
        )
    except KeyError as exc:
        typer.echo(
            str(exc),
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(
        f"{voice['id']} is now ignored."
    )


@voice_app.command("restore")
def voice_restore(
    voice_id: str,
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """Restore an ignored voice profile."""

    storage = storage_for(dataset)

    try:
        voice = set_voice_ignored(
            storage,
            voice_id,
            False,
        )
    except KeyError as exc:
        typer.echo(
            str(exc),
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(
        f"{voice['id']} is no longer ignored."
    )


@region_app.command("import-detector")
def region_import_detector(
    detector_output: Path,
    source_id: str = typer.Option(
        ...,
        help=(
            "Dataset source identifier associated "
            "with these detector regions."
        ),
    ),
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """Import candidate regions from detector output."""

    detector_output = (
        detector_output.resolve()
    )

    if not detector_output.is_file():
        raise typer.BadParameter(
            "Detector output does not exist: "
            f"{detector_output}"
        )

    storage = storage_for(dataset)

    try:
        output = load_detector_output(
            detector_output
        )

        result = import_detector_regions(
            storage,
            output,
            source_id=source_id,
        )
    except (
        ValueError,
        OSError,
    ) as exc:
        typer.echo(
            f"Import failed: {exc}",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(
        f"Imported {len(result.imported)} "
        f"candidate regions."
    )

    typer.echo(
        f"Skipped {result.skipped} "
        f"existing candidate regions."
    )

    typer.echo(
        f"  detector: "
        f"{output.detector_name}"
    )

    typer.echo(
        f"  source:   "
        f"{source_id}"
    )


@region_app.command("import-transcripts")
def region_import_transcripts(
    transcript_output: Path,
    name: str = typer.Option(
        ...,
        help=(
            "Transcript hypothesis name, "
            "e.g. whisper_raw or whisper_speech."
        ),
    ),
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """Import transcript hypotheses for candidate regions."""

    transcript_output = (
        transcript_output.resolve()
    )

    if not transcript_output.is_file():
        raise typer.BadParameter(
            "Transcript output does not exist: "
            f"{transcript_output}"
        )

    storage = storage_for(dataset)

    try:
        output = load_transcript_output(
            transcript_output
        )

        result = import_transcripts(
            storage,
            output,
            name=name,
        )

    except (
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as exc:
        typer.echo(
            f"Import failed: {exc}",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(
        f"Imported {result.imported} "
        f"transcript hypotheses."
    )

    typer.echo(
        f"Skipped {result.skipped} "
        f"existing transcript hypotheses."
    )

    typer.echo(
        f"  transcriber:    "
        f"{output.transcriber_name}"
    )

    typer.echo(
        f"  model:          "
        f"{output.transcriber_model or '-'}"
    )

    typer.echo(
        f"  representation: "
        f"{output.representation}"
    )

    typer.echo(
        f"  name:           "
        f"{name}"
    )


@region_app.command("import-embeddings")
def region_import_embeddings(
    embedding_output: Path,
    name: str = typer.Option(
        ...,
        help=(
            "Embedding evidence name, "
            "e.g. ecapa_raw or wespeaker_raw."
        ),
    ),
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """Import speaker embeddings for candidate regions."""

    embedding_output = (
        embedding_output.resolve()
    )

    if not embedding_output.is_file():
        raise typer.BadParameter(
            "Embedding output does not exist: "
            f"{embedding_output}"
        )

    storage = storage_for(dataset)

    try:
        output = load_embedding_output(
            embedding_output
        )

        if output.record_type != "region":
            raise ValueError(
                "Embedding output contains "
                f"{output.record_type} records; "
                "region records are required"
            )

        result = import_embeddings(
            storage,
            output,
            name=name,
        )

    except (
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as exc:
        typer.echo(
            f"Import failed: {exc}",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(
        f"Imported {result.imported} "
        f"speaker embeddings."
    )

    typer.echo(
        f"Skipped {result.skipped} "
        f"existing speaker embeddings."
    )

    typer.echo(
        f"  encoder:        "
        f"{output.encoder_name}"
    )

    typer.echo(
        f"  model:          "
        f"{output.encoder_model or '-'}"
    )

    typer.echo(
        f"  representation: "
        f"{output.representation}"
    )

    typer.echo(
        f"  name:           "
        f"{name}"
    )


@turn_app.command(
    "refresh-boundary-evidence"
)
def turn_refresh_boundary_evidence(
    source_id: str,
    dataset: Path = typer.Option(
        ...,
        help="Dataset root directory.",
    ),
):
    """Refresh boundary evidence for all turns of a source."""

    storage = storage_for(dataset)

    try:
        turns = refresh_source_boundary_evidence(
            storage,
            source_id,
        )
    except (KeyError, ValueError) as exc:
        raise typer.BadParameter(
            str(exc)
        ) from exc

    near_start = sum(
        bool(
            turn.get("metadata", {})
            .get("boundary_evidence", {})
            .get("near_source_start")
        )
        for turn in turns
    )

    near_end = sum(
        bool(
            turn.get("metadata", {})
            .get("boundary_evidence", {})
            .get("near_source_end")
        )
        for turn in turns
    )

    typer.echo(
        f"Updated {len(turns)} turns "
        f"for {source_id}"
    )
    typer.echo(
        f"  near source start: {near_start}"
    )
    typer.echo(
        f"  near source end:   {near_end}"
    )


@turn_app.command("import-embeddings")
def turn_import_embeddings(
    embedding_output: Path,
    name: str = typer.Option(
        ...,
        help=(
            "Embedding evidence name, "
            "e.g. ecapa_raw or wespeaker_raw."
        ),
    ),
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """Import speaker embeddings for reconciled turns."""

    embedding_output = (
        embedding_output.resolve()
    )

    if not embedding_output.is_file():
        raise typer.BadParameter(
            "Embedding output does not exist: "
            f"{embedding_output}"
        )

    storage = storage_for(dataset)

    try:
        output = load_embedding_output(
            embedding_output
        )

        if output.record_type != "turn":
            raise ValueError(
                "Embedding output contains "
                f"{output.record_type} records; "
                "turn records are required"
            )

        result = import_embeddings(
            storage,
            output,
            name=name,
        )

    except (
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as exc:
        typer.echo(
            f"Import failed: {exc}",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(
        f"Imported {result.imported} "
        f"speaker embeddings."
    )

    typer.echo(
        f"Skipped {result.skipped} "
        f"existing speaker embeddings."
    )

    typer.echo(
        f"  encoder:        "
        f"{output.encoder_name}"
    )

    typer.echo(
        f"  model:          "
        f"{output.encoder_model or '-'}"
    )

    typer.echo(
        f"  representation: "
        f"{output.representation}"
    )

    typer.echo(
        f"  name:           "
        f"{name}"
    )


@region_app.command("materialize-audio")
def region_materialize_audio(
    source: Path,
    source_id: str = typer.Option(
        ...,
        help=(
            "Dataset source identifier whose "
            "candidate regions should be materialized."
        ),
    ),
    name: str = typer.Option(
        ...,
        help="Representation name, e.g. raw or speech.",
    ),
    kind: str = typer.Option(
        ...,
        help="Representation kind, e.g. center or separated.",
    ),
    purpose: list[str] | None = typer.Option(
        None,
        "--purpose",
        help=(
            "Intended use of the representation. "
            "May be specified multiple times."
        ),
    ),
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """Materialize audio for candidate regions."""

    source = source.resolve()

    if not source.is_file():
        raise typer.BadParameter(
            f"Audio source does not exist: {source}"
        )

    storage = storage_for(dataset)

    try:
        result = materialize_regions(
            storage=storage,
            source_id=source_id,
            source=source,
            representation_name=name,
            kind=kind,
            purposes=purpose or [],
        )
    except (
        ValueError,
        KeyError,
        OSError,
    ) as exc:
        typer.echo(
            f"Materialization failed: {exc}",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(
        f"Created {result.created} "
        f"audio representations."
    )

    typer.echo(
        f"Skipped {result.skipped} "
        f"existing audio representations."
    )


@region_app.command("reject")
def region_reject(
    region_id: str = typer.Argument(
        ...,
        help="Candidate region ID to reject.",
    ),
    reason: str = typer.Option(
        ...,
        help=(
            "Rejection reason: non_speech, "
            "unusable, duplicate, or other."
        ),
    ),
    notes: str | None = typer.Option(
        None,
        help="Optional notes about the rejection.",
    ),
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """Reject a candidate region."""

    storage = storage_for(dataset)

    try:
        region = reject_region(
            storage,
            region_id,
            reason=reason,
            notes=notes,
        )

    except (
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as exc:
        typer.echo(
            f"Region rejection failed: {exc}",
            err=True,
        )
        raise typer.Exit(1)

    reconciliation = (
        effective_region_reconciliation(
            storage,
            region,
        )
    )

    typer.echo(f"Rejected {region_id}")
    typer.echo(
        f"  reason: {reconciliation['reason']}"
    )
    typer.echo(
        f"  notes:  "
        f"{reconciliation['notes'] or '-'}"
    )


@region_app.command("list")
def region_list(
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """List detector candidate regions."""

    storage = storage_for(dataset)
    regions = storage.regions.load()

    if not regions:
        typer.echo(
            "No candidate regions."
        )
        return

    for region in regions:
        label = (
            region.get("detector_label")
            or "-"
        )

        try:
            reconciliation = (
                effective_region_reconciliation(
                    storage,
                    region,
                )
            )
        except (
            ValueError,
            RuntimeError,
        ) as exc:
            typer.echo(
                "Failed to determine reconciliation "
                f"for {region.get('id', '<unknown>')}: "
                f"{exc}",
                err=True,
            )
            raise typer.Exit(1)

        status = reconciliation["status"]

        if status == "reconciled":
            reconciliation_text = (
                "reconciled -> "
                f"{reconciliation['turn_id']}"
            )

        elif status == "rejected":
            reconciliation_text = (
                "rejected "
                f"({reconciliation['reason']})"
            )

        else:
            reconciliation_text = "pending"

        typer.echo(
            f"{region['id']}  "
            f"{region['source_start']:.3f}-"
            f"{region['source_end']:.3f}  "
            f"{region['detector']}  "
            f"{label}  "
            f"{reconciliation_text}"
        )


@turn_app.command("materialize-audio")
def turn_materialize_audio(
    source_id: str = typer.Option(
        ...,
        help="Source ID whose turns should be materialized.",
    ),
    source: Path = typer.Option(
        ...,
        help="Timeline-aligned audio source.",
    ),
    name: str = typer.Option(
        ...,
        help="Representation name, e.g. raw or speech.",
    ),
    kind: str = typer.Option(
        ...,
        help="Representation kind, e.g. center or separated-speech.",
    ),
    purpose: list[str] = typer.Option(
        ...,
        help=(
            "Purpose of this representation. "
            "May be specified multiple times."
        ),
    ),
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """Materialize audio representations for speech turns."""

    storage = storage_for(dataset)

    try:
        result = materialize_turns(
            storage=storage,
            source_id=source_id,
            source=source,
            representation_name=name,
            kind=kind,
            purposes=purpose,
        )
    except (
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as exc:
        typer.echo(
            f"Turn materialization failed: {exc}",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(
        f"Created {result.created} turn "
        f"audio representations."
    )
    typer.echo(
        f"Skipped {result.skipped} existing "
        f"turn audio representations."
    )


@turn_app.command("create-from-regions")
def turn_create_from_regions(
    region_ids: list[str] = typer.Argument(
        ...,
        help=(
            "Candidate region IDs in source "
            "timeline order."
        ),
    ),
    transcript: str | None = typer.Option(
        None,
        help="Optional reconciled transcript.",
    ),
    language: str | None = typer.Option(
        None,
        help="Optional language, e.g. en or de.",
    ),
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """Create a speech turn from candidate regions."""

    storage = storage_for(dataset)

    try:
        before_ids = {
            turn.get("id")
            for turn in storage.turns.load()
        }

        turn = create_turn_from_regions(
            storage,
            region_ids,
            transcript=transcript,
            language=language,
        )

    except (
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as exc:
        typer.echo(
            f"Reconciliation failed: {exc}",
            err=True,
        )
        raise typer.Exit(1)

    created = (
        turn["id"] not in before_ids
    )

    action = (
        "Created"
        if created
        else "Existing"
    )

    typer.echo(
        f"{action} {turn['id']}"
    )

    typer.echo(
        "  regions:    "
        + ", ".join(
            turn["source_regions"]
        )
    )

    typer.echo(
        f"  source:     {turn['source_id']}"
    )

    typer.echo(
        f"  range:      "
        f"{turn['source_start']:.6f}-"
        f"{turn['source_end']:.6f}"
    )

    typer.echo(
        f"  language:   "
        f"{turn.get('language') or '-'}"
    )

    typer.echo(
        f"  transcript: "
        f"{turn.get('transcript') or '-'}"
    )


@turn_app.command("edit")
def turn_edit(
    turn_id: str,
    transcript: str | None = typer.Option(
        None,
        help="Set the reconciled transcript.",
    ),
    language: str | None = typer.Option(
        None,
        help="Set the language, e.g. en or de.",
    ),
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """Edit transcript and language metadata for a speech turn."""

    if transcript is None and language is None:
        typer.echo(
            "Nothing to edit: specify --transcript "
            "and/or --language.",
            err=True,
        )
        raise typer.Exit(1)

    storage = storage_for(dataset)

    try:
        turn = edit_turn(
            storage,
            turn_id,
            transcript=transcript,
            language=language,
        )
    except (
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as exc:
        typer.echo(
            f"Turn edit failed: {exc}",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(
        f"Updated {turn_id}"
    )
    typer.echo(
        f"  language:   "
        f"{turn.get('language') or '-'}"
    )
    typer.echo(
        f"  transcript: "
        f"{turn.get('transcript') or '-'}"
    )


@turn_app.command("split")

def turn_split(
    turn_id: str,
    after_region: str = typer.Option(
        ...,
        "--after",
        help=(
            "Split after this candidate region. "
            "The original turn ID is retained "
            "for the left side."
        ),
    ),
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """Split one reconciled speech turn into two turns."""

    storage = storage_for(dataset)

    try:
        left, right = split_turn(
            storage,
            turn_id,
            after_region_id=after_region,
        )

    except (
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as exc:
        typer.echo(
            f"Turn split failed: {exc}",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(
        f"Split {turn_id}"
    )

    typer.echo(
        f"  left:   {left['id']}  "
        + ", ".join(left["source_regions"])
    )

    typer.echo(
        f"          "
        f"{left['source_start']:.6f}-"
        f"{left['source_end']:.6f}"
    )

    typer.echo(
        f"  right:  {right['id']}  "
        + ", ".join(right["source_regions"])
    )

    typer.echo(
        f"          "
        f"{right['source_start']:.6f}-"
        f"{right['source_end']:.6f}"
    )


@turn_app.command("list")
def turn_list(
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """List speech turns."""

    storage = storage_for(dataset)
    turns = storage.turns.load()

    if not turns:
        typer.echo("No turns.")
        return

    for turn in turns:
        assignment = turn.get(
            "assignment",
            {},
        )

        status = assignment.get(
            "status",
            "unknown",
        )

        voice_id = assignment.get(
            "voice_id"
        )

        assignment_text = (
            voice_id
            if status == "assigned"
            and voice_id
            else status
        )

        language = (
            turn.get("language")
            or "-"
        )

        transcript = (
            turn.get("transcript")
            or ""
        )

        typer.echo(
            f"{turn['id']}  "
            f"{turn['source_start']:.3f}-"
            f"{turn['source_end']:.3f}  "
            f"{language}  "
            f"{assignment_text}  "
            f"{transcript}"
        )


@turn_app.command("assign")
def turn_assign(
    turn_id: str,
    voice_id: str,
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """Assign a turn to a persistent voice profile."""

    storage = storage_for(dataset)

    try:
        turn = assign_turn(
            storage,
            turn_id,
            voice_id,
            method="manual",
        )
    except KeyError as exc:
        typer.echo(
            str(exc),
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(
        f"{turn_id} -> "
        f"{turn['assignment']['voice_id']}"
    )


@turn_app.command("unknown")
def turn_unknown(
    turn_id: str,
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """Reset a turn to unknown voice."""

    storage = storage_for(dataset)

    try:
        mark_turn_unknown(
            storage,
            turn_id,
        )
    except KeyError as exc:
        typer.echo(
            str(exc),
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(
        f"{turn_id} -> unknown"
    )


@turn_app.command("ignore")
def turn_ignore(
    turn_id: str,
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """Ignore one specific turn."""

    storage = storage_for(dataset)

    try:
        ignore_turn(
            storage,
            turn_id,
            method="manual",
        )
    except KeyError as exc:
        typer.echo(
            str(exc),
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(
        f"{turn_id} -> ignore"
    )


if __name__ == "__main__":
    app()


@turn_app.command("review")
def turn_review(
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
    source_id: str | None = typer.Option(
        None,
        help="Review only turns from this source.",
    ),
    context_padding: float = typer.Option(
        2.0,
        min=0.0,
        help="Context padding in seconds.",
    ),
):
    """Interactively review reconciled speech turns."""

    storage = storage_for(dataset)
    dataset = storage.root

    turns = sorted_turns(
        storage,
        source_id=source_id,
    )

    if not turns:
        typer.echo("No speech turns to review.")
        return

    index = 0

    typer.echo(reviewer_help())

    while True:
        turn_id = turns[index]["id"]

        current = storage.get_turn(turn_id)

        if current is None:
            typer.echo(
                f"Turn disappeared: {turn_id}",
                err=True,
            )
            raise typer.Exit(1)

        typer.echo()
        typer.echo(
            format_turn(
                current,
                position=index + 1,
                total=len(turns),
            )
        )
        typer.echo()

        try:
            command = input(
                "[p/r/c/s/t/l/v/u/i/a/x/n/b/h/q] > "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            stop()
            typer.echo()
            return

        try:
            if command == "p":
                name, path = (
                    play_preferred_review_audio(
                        dataset,
                        current,
                    )
                )

                typer.echo(
                    f"Playing {name}: {path}"
                )

            elif command == "r":
                representation = (
                    raw_representation(current)
                )

                path = play_representation(
                    dataset,
                    representation,
                )

                typer.echo(
                    f"Playing raw: {path}"
                )

            elif command == "c":
                source = storage.get_source(
                    current["source_id"]
                )

                if source is None:
                    raise KeyError(
                        "Source does not exist: "
                        f"{current['source_id']}"
                    )

                (
                    name,
                    path,
                    start,
                    end,
                ) = play_turn_context(
                    source,
                    current,
                    padding=context_padding,
                )

                typer.echo(
                    f"Playing context {name}: "
                    f"{start:.3f}-{end:.3f} "
                    f"from {path}"
                )

            elif command == "s":
                stop()
                typer.echo("Playback stopped.")

            elif command == "t":
                value = input(
                    "Transcript "
                    "(blank = cancel): "
                )

                if value.strip():
                    edit_turn(
                        storage,
                        turn_id,
                        transcript=value,
                    )

            elif command == "l":
                value = input(
                    "Language "
                    "(blank = cancel): "
                )

                if value.strip():
                    edit_turn(
                        storage,
                        turn_id,
                        language=value,
                    )

            elif command == "v":
                voices = storage.voices.load()

                if not voices:
                    typer.echo(
                        "No voice profiles."
                    )
                    continue

                typer.echo("Voices:")

                for voice in voices:
                    ignored = (
                        " [ignored]"
                        if voice.get("ignored")
                        else ""
                    )

                    typer.echo(
                        f"  {voice['id']}  "
                        f"{voice.get('character') or '-'}"
                        f"{ignored}"
                    )

                voice_id = input(
                    "Voice ID "
                    "(blank = cancel): "
                ).strip()

                if voice_id:
                    assign_turn(
                        storage,
                        turn_id,
                        voice_id,
                    )

            elif command == "u":
                mark_turn_unknown(
                    storage,
                    turn_id,
                )

            elif command == "i":
                ignore_turn(
                    storage,
                    turn_id,
                )

            elif command == "a":
                mark_turn_reviewed(
                    storage,
                    turn_id,
                )

            elif command == "x":
                mark_turn_pending(
                    storage,
                    turn_id,
                )

            elif command == "n":
                stop()

                if index < len(turns) - 1:
                    index += 1
                else:
                    typer.echo(
                        "Already at final turn."
                    )

            elif command == "b":
                stop()

                if index > 0:
                    index -= 1
                else:
                    typer.echo(
                        "Already at first turn."
                    )

            elif command == "h":
                typer.echo(
                    reviewer_help()
                )

            elif command == "q":
                stop()
                return

            elif not command:
                continue

            else:
                typer.echo(
                    f"Unknown command: {command}"
                )

        except (
            ValueError,
            KeyError,
            RuntimeError,
            OSError,
        ) as exc:
            typer.echo(
                f"Action failed: {exc}",
                err=True,
            )

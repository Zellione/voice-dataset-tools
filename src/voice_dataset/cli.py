from pathlib import Path

import typer

from .cuda import preload_cuda_libraries
from .media import probe_audio_streams
from .representations import materialize_regions
from .storage import DatasetStorage
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

preload_cuda_libraries()

from .ingest import ingest as ingest_source


app = typer.Typer(
    help="Build and manage speech datasets for voice model training.",
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

        typer.echo(
            f"{region['id']}  "
            f"{region['source_start']:.3f}-"
            f"{region['source_end']:.3f}  "
            f"{region['detector']}  "
            f"{label}"
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

        transcript = (
            turn.get("transcript")
            or ""
        )

        typer.echo(
            f"{turn['id']}  "
            f"{turn['source_start']:.3f}-"
            f"{turn['source_end']:.3f}  "
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

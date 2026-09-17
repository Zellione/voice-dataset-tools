from pathlib import Path
from .media import probe_audio_streams
import typer

from .cuda import preload_cuda_libraries

preload_cuda_libraries()

from .ingest import ingest as ingest_source

app = typer.Typer(
    help="Build and manage speech datasets for voice model training.",
    no_args_is_help=True,
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
        typer.echo(f"Audio stream #{stream.index}")
        typer.echo(f"  Codec:          {stream.codec or '-'}")
        typer.echo(f"  Sample rate:    {stream.sample_rate or '-'}")
        typer.echo(f"  Channels:       {stream.channels or '-'}")
        typer.echo(f"  Channel layout: {stream.channel_layout or '-'}")
        typer.echo(f"  Language:       {stream.language or '-'}")
        typer.echo(f"  Title:          {stream.title or '-'}")
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
        help="Audio stream number within the audio streams (0 = first audio stream).",
    ),
    channel: str = typer.Option(
        "auto",
        help="Channel extraction mode: auto, mono, or center.",
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

    if channel not in {"auto", "mono", "center"}:
        raise typer.BadParameter(
            "--channel must be auto, mono, or center"
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


if __name__ == "__main__":
    app()

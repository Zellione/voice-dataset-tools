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
    merge_turns,
    reject_region,
    split_turn,
)
from .automatic_reconciliation import (
    analyze_continuous_reconciliation,
    apply_continuous_merge_candidate,
)
from .storage import DatasetStorage
from .sources import set_representation_provenance
from .separation import separate_source
from .playback import (
    play_preferred_review_audio,
    play_representation,
    play_turn_context,
    stop,
)
from .review import (
    accept_alignment_recovery,
    accept_edge_recovery,
    mark_turn_boundary_clipped,
    mark_turn_boundary_complete,
    mark_turn_pending,
    mark_turn_reviewed,
)
from .reviewer import (
    format_turn,
    format_voice_matches,
    raw_representation,
    reviewer_help,
    sorted_turns,
)
from .reviewer_session import ReviewerSession
from .reviewer_tui import run_reviewer_tui
from .voices import (
    assign_turn,
    create_voice,
    mark_turn_unknown,
    set_voice_ignored,
)
from .turn_curation import (
    mark_turn_rejected,
    migrate_legacy_ignored_turns,
)
from .detectors import (
    detect_source_regions,
    import_detector_regions,
    load_detector_output,
)
from .embeddings import (
    embed_turns,
    import_embeddings,
    load_embedding_output,
)
from .transcripts import (
    import_transcripts,
    load_transcript_output,
)
from .region_asr import transcribe_regions
from .boundary_evidence import (
    refresh_source_boundary_evidence,
)
from .workers import WorkerError
from .process import process_source

preload_cuda_libraries()

from .ingest import ingest as ingest_source
from .speaker_similarity import rank_voice_matches
from .continuous_asr import transcribe_source_qwen3
from .utterance_pipeline import (
    apply_source_utterance_turns,
    build_source_utterances,
    prepare_source_speaker_evidence,
)


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


@source_app.command("separate")
def source_separate(
    source_id: str,
    representation: str = typer.Option(
        "center",
        help="Input source representation.",
    ),
    output_representation: str = typer.Option(
        "speech",
        help="Output speech representation name.",
    ),
    dataset: Path = typer.Option(
        ...,
        help="Dataset root directory.",
    ),
    chunk: float = typer.Option(
        30.0,
        min=0.001,
        help="BandIt chunk size in seconds.",
    ),
    overlap: float = typer.Option(
        2.0,
        min=0.0,
        help="BandIt chunk overlap in seconds.",
    ),
):
    """Separate speech with BandIt v2."""

    storage = storage_for(dataset)

    try:
        result = separate_source(
            storage,
            source_id,
            input_representation=representation,
            output_representation=(
                output_representation
            ),
            chunk=chunk,
            overlap=overlap,
        )
    except (
        KeyError,
        ValueError,
        WorkerError,
    ) as exc:
        raise typer.BadParameter(
            str(exc)
        ) from exc

    typer.echo(
        "Speech representation ready:"
    )
    typer.echo(
        f"  source:         {result.source_id}"
    )
    typer.echo(
        "  representation: "
        f"{result.representation_name}"
    )
    typer.echo(
        f"  path:           {result.path}"
    )


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
        typer.echo(
            f"  Default:        {'yes' if stream.is_default else 'no'}"
        )
        typer.echo()


@app.command()
def ingest(
    source: Path,
    source_id: str = typer.Option(
        ...,
        help="Stable identifier for this source.",
    ),
    audio_stream: int | None = typer.Option(
        None,
        help=(
            "Audio stream number within the audio streams. "
            "Automatically selected when omitted."
        ),
    ),
    language: str | None = typer.Option(
        None,
        help=(
            "Preferred audio stream language. "
            "Used for automatic stream selection."
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

    try:
        record = ingest_source(
            source=source,
            output=output,
            source_id=source_id,
            audio_stream=audio_stream,
            language=language,
            channel=channel,
            start=start,
            duration=duration,
        )
    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise typer.BadParameter(str(exc)) from exc

    representation_name = next(
        iter(record.representations)
    )
    representation = record.representations[
        representation_name
    ]

    typer.echo()
    typer.echo(f"Ingested source {record.id}")
    typer.echo(
        f"  representation: {representation_name}"
    )
    typer.echo(
        f"  audio:          {representation.path}"
    )
    typer.echo(
        f"  media start:    "
        f"{representation.media_start:.3f}s"
    )


@app.command()
def process(
    source: Path,
    source_id: str = typer.Option(
        ...,
        help="Stable identifier for this source.",
    ),
    audio_stream: int | None = typer.Option(
        None,
        help=(
            "Audio stream number within the audio streams. "
            "Automatically selected when omitted."
        ),
    ),
    audio_language: str | None = typer.Option(
        None,
        help=(
            "Preferred audio stream language. "
            "Used for automatic stream selection."
        ),
    ),
    language: str | None = typer.Option(
        None,
        help=(
            "Speech recognition language code, "
            "e.g. en or de."
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
        help="Maximum duration to process in seconds.",
    ),
    output: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset output directory.",
    ),
):
    """Prepare source media for interactive review."""

    if channel not in {
        "auto",
        "mono",
        "center",
    }:
        raise typer.BadParameter(
            "--channel must be "
            "auto, mono, or center"
        )

    try:
        result = process_source(
            source=source,
            output=output,
            source_id=source_id,
            audio_stream=audio_stream,
            audio_language=audio_language,
            language=language,
            channel=channel,
            start=start,
            duration=duration,
        )
    except (
        FileNotFoundError,
        KeyError,
        RuntimeError,
        ValueError,
        WorkerError,
    ) as exc:
        raise typer.BadParameter(
            str(exc)
        ) from exc

    typer.echo()
    typer.echo(
        f"Source ready for review: {result.source_id}"
    )
    typer.echo(
        f"  detector ran:  "
        f"{'yes' if result.detector_ran else 'no'}"
    )
    typer.echo(
        f"  Qwen ran:      "
        f"{'yes' if result.qwen_ran else 'no'}"
    )
    typer.echo(
        f"  turns created: {result.turns_created}"
    )
    typer.echo(
        f"  turns skipped: {result.turns_skipped}"
    )
    typer.echo(
        f"  turns review:  {result.turns_review}"
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


@region_app.command("detect")
def region_detect(
    source_id: str = typer.Argument(
        ...,
        help="Dataset source identifier.",
    ),
    representation: str = typer.Option(
        ...,
        help=(
            "Source representation to analyze, "
            "e.g. center."
        ),
    ),
    num_speakers: int | None = typer.Option(
        None,
        help="Exact number of speakers.",
    ),
    min_speakers: int | None = typer.Option(
        None,
        help="Minimum number of speakers.",
    ),
    max_speakers: int | None = typer.Option(
        None,
        help="Maximum number of speakers.",
    ),
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """Detect candidate speech regions."""

    storage = storage_for(dataset)

    try:
        result = detect_source_regions(
            storage,
            source_id,
            representation,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    except (
        ValueError,
        KeyError,
        OSError,
        WorkerError,
    ) as exc:
        typer.echo(
            f"Detection failed: {exc}",
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
        f"  detector:       "
        f"{result.output.detector_name}"
    )

    typer.echo(
        f"  model:          "
        f"{result.output.detector_model or '-'}"
    )

    typer.echo(
        f"  revision:       "
        f"{result.output.detector_revision or '-'}"
    )

    typer.echo(
        f"  representation: "
        f"{representation}"
    )

    typer.echo(
        f"  source:         "
        f"{source_id}"
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


@region_app.command("transcribe")
def region_transcribe(
    source_id: str,
    representation: str = typer.Option(
        "speech",
        help="Source representation to transcribe.",
    ),
    name: str = typer.Option(
        "whisper",
        help="Transcript hypothesis name.",
    ),
    language: str | None = typer.Option(
        None,
        help=(
            "Optional Whisper language override, "
            "e.g. en or de. Omit for automatic detection."
        ),
    ),
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """Transcribe candidate regions with Whisper."""

    storage = storage_for(dataset)

    try:
        result = transcribe_regions(
            storage,
            source_id,
            representation_name=representation,
            transcript_name=name,
            language=language,
        )
    except (
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as exc:
        typer.echo(
            f"Transcription failed: {exc}",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(
        f"Transcribed {result.transcribed} "
        f"candidate regions."
    )
    typer.echo(
        f"Skipped {result.skipped} "
        f"existing transcript hypotheses."
    )
    typer.echo(
        f"  model:          large-v3"
    )
    typer.echo(
        f"  representation: {representation}"
    )
    typer.echo(
        f"  name:           {name}"
    )
    typer.echo(
        f"  language:       "
        f"{language or 'auto'}"
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


@turn_app.command("embed")
def turn_embed(
    encoder: str = typer.Option(
        ...,
        help=(
            "Speaker encoder: ecapa or wespeaker."
        ),
    ),
    representation: str = typer.Option(
        "center",
        help=(
            "Turn audio representation to embed."
        ),
    ),
    name: str | None = typer.Option(
        None,
        help=(
            "Embedding evidence name. "
            "Defaults to <encoder>_<representation>."
        ),
    ),
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """Generate and import speaker embeddings for turns."""

    evidence_name = (
        name
        or f"{encoder}_{representation}"
    )

    storage = storage_for(dataset)

    try:
        result = embed_turns(
            storage,
            encoder=encoder,
            representation=representation,
            name=evidence_name,
        )
    except (
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as exc:
        typer.echo(
            f"Embedding failed: {exc}",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(
        f"Imported {result.imported} "
        "speaker embeddings."
    )
    typer.echo(
        f"Skipped {result.skipped} "
        "existing speaker embeddings."
    )
    typer.echo(
        f"  encoder:        {result.encoder}"
    )
    typer.echo(
        f"  representation: "
        f"{result.representation}"
    )
    typer.echo(
        f"  name:           {evidence_name}"
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


@region_app.command("transcribe-continuous")
def region_transcribe_continuous(
    source_id: str,
    representation: str = typer.Option(
        "speech",
        help="Source representation to transcribe.",
    ),
    evidence: str = typer.Option(
        "qwen3",
        help="Continuous ASR evidence name.",
    ),
    language: str | None = typer.Option(
        None,
        help="Optional ASR language hint.",
    ),
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """Transcribe a source continuously with Qwen3 ASR."""

    storage = storage_for(dataset)

    try:
        result = transcribe_source_qwen3(
            storage,
            source_id,
            representation_name=representation,
            evidence_name=evidence,
            language=language,
        )
    except (
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as exc:
        typer.echo(
            f"Continuous ASR failed: {exc}",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(
        f"Continuous ASR: {len(result.words)} words, "
        f"{len(result.utterances)} utterances."
    )

    if result.language is not None:
        typer.echo(f"  language: {result.language}")

    typer.echo(f"  evidence: {evidence}")
    typer.echo(f"  representation: {representation}")


@region_app.command("analyze-continuous")
def region_analyze_continuous(
    source_id: str,
    evidence: str = typer.Option(
        "qwen3",
        help="Continuous ASR evidence name.",
    ),
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """Analyze continuous ASR against detector regions."""

    storage = storage_for(dataset)

    try:
        results = analyze_continuous_reconciliation(
            storage,
            source_id,
            evidence_name=evidence,
        )
    except (
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as exc:
        typer.echo(
            f"Continuous reconciliation analysis failed: {exc}",
            err=True,
        )
        raise typer.Exit(1)

    for result in results:
        typer.echo(
            f"{result.utterance_index + 1:02d}  "
            f"{result.start:.3f}-"
            f"{result.end:.3f}  "
            f"{result.status}"
        )

        typer.echo(
            "  regions:  "
            + (
                ", ".join(result.region_ids)
                if result.region_ids
                else "-"
            )
        )

        typer.echo(
            "  speakers: "
            + (
                ", ".join(result.speaker_labels)
                if result.speaker_labels
                else "-"
            )
        )

        for boundary in result.boundaries:
            same_speaker = (
                boundary.left_speaker
                == boundary.right_speaker
            )

            typer.echo(
                "  boundary: "
                f"{boundary.left_region_id} -> "
                f"{boundary.right_region_id}  "
                f"region_gap="
                f"{boundary.region_gap:.3f}s  "
                f"word_gap="
                f"{boundary.word_gap:.3f}s  "
                f"same_speaker="
                f"{'yes' if same_speaker else 'no'}"
            )

            typer.echo(
                "            "
                f"{boundary.left_word!r} -> "
                f"{boundary.right_word!r}"
            )

        typer.echo(
            "  reasons:  "
            + (
                ", ".join(result.reasons)
                if result.reasons
                else "-"
            )
        )

        typer.echo(
            f"  text:     {result.text}"
        )


@region_app.command("apply-continuous-merge")
def region_apply_continuous_merge(
    source_id: str,
    utterance: int = typer.Option(
        ...,
        min=1,
        help="1-based continuous ASR utterance number.",
    ),
    evidence: str = typer.Option(
        "qwen3",
        help="Continuous ASR evidence name.",
    ),
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """Apply one continuous-ASR merge candidate."""

    storage = storage_for(dataset)

    try:
        turn = apply_continuous_merge_candidate(
            storage,
            source_id,
            utterance - 1,
            evidence_name=evidence,
        )
    except (
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as exc:
        typer.echo(
            f"Continuous merge failed: {exc}",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(
        f"Merged utterance {utterance:02d} "
        f"-> {turn['id']}"
    )

    typer.echo(
        "  regions: "
        + ", ".join(turn["source_regions"])
    )

    typer.echo(
        f"  range:   "
        f"{turn['source_start']:.6f}-"
        f"{turn['source_end']:.6f}"
    )

    typer.echo(
        f"  language: "
        f"{turn.get('language') or '-'}"
    )

    typer.echo(
        f"  text:     "
        f"{turn.get('transcript') or '-'}"
    )


@region_app.command("reconcile")
def region_reconcile(
    source_id: str,
    language: str | None = typer.Option(
        None,
        help="Language stored on generated turns.",
    ),
    asr: str = typer.Option(
        "qwen3",
        help="Continuous ASR evidence used for utterances.",
    ),
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """Create turns from continuous ASR and utterance evidence."""

    storage = storage_for(dataset)

    try:
        pipeline = build_source_utterances(
            storage,
            source_id,
            asr_evidence_name=asr,
        )

        result = apply_source_utterance_turns(
            storage,
            source_id,
            pipeline,
            language=language,
            asr_evidence_name=asr,
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

    typer.echo(
        f"Created {result.created} canonical turns."
    )
    typer.echo(
        f"Skipped {result.skipped} existing turns."
    )
    typer.echo(
        f"Flagged {result.review} turns for review."
    )
    typer.echo(
        f"  ASR: {asr}"
    )


@region_app.command("prepare-speakers")
def region_prepare_speakers(
    source_id: str = typer.Argument(
        ...,
        help="Source whose turns should receive speaker evidence.",
    ),
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """Prepare speaker audio and embeddings for source turns."""

    storage = storage_for(dataset)

    try:
        result = prepare_source_speaker_evidence(
            storage,
            source_id,
        )
    except (
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as exc:
        typer.echo(
            f"Speaker preparation failed: {exc}",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(
        "Speaker representations: "
        f"{result.representation.created} created, "
        f"{result.representation.skipped} skipped."
    )
    typer.echo(
        "ECAPA embeddings: "
        f"{result.ecapa.imported} imported, "
        f"{result.ecapa.skipped} skipped."
    )
    typer.echo(
        "WeSpeaker embeddings: "
        f"{result.wespeaker.imported} imported, "
        f"{result.wespeaker.skipped} skipped."
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


@turn_app.command("merge")
def turn_merge(
    turn_ids: list[str] = typer.Argument(
        ...,
        help=(
            "Turn IDs to merge, in source "
            "timeline order."
        ),
    ),
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """Merge reconciled speech turns into one turn."""

    storage = storage_for(dataset)

    try:
        merged = merge_turns(
            storage,
            turn_ids,
        )

    except (
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as exc:
        typer.echo(
            f"Turn merge failed: {exc}",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(
        "Merged "
        + ", ".join(turn_ids)
        + f" -> {merged['id']}"
    )

    typer.echo(
        "  regions: "
        + ", ".join(
            merged["source_regions"]
        )
    )

    typer.echo(
        f"  range:   "
        f"{merged['source_start']:.6f}-"
        f"{merged['source_end']:.6f}"
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


@turn_app.command("reject")
def turn_reject(
    turn_id: str,
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """Reject one specific turn."""

    storage = storage_for(dataset)

    try:
        mark_turn_rejected(
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
        f"{turn_id} -> rejected"
    )


if __name__ == "__main__":
    app()


@turn_app.command("migrate-legacy-ignore")
def turn_migrate_legacy_ignore(
    dataset: Path = typer.Option(
        Path("datasets/output"),
        help="Dataset directory.",
    ),
):
    """Migrate legacy ignored turns to rejected curation."""

    storage = storage_for(dataset)

    migrated_count = migrate_legacy_ignored_turns(
        storage
    )

    typer.echo(
        f"Migrated {migrated_count} legacy "
        "ignored turn"
        + ("" if migrated_count == 1 else "s")
        + "."
    )


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
    speaker_embedding: list[str] | None = typer.Option(
        None,
        "--speaker-embedding",
        help=(
            "Turn embedding name used to show "
            "speaker similarity evidence. "
            "May be specified multiple times."
        ),
    ),
    auto_review_only: bool = typer.Option(
        False,
        "--auto-review-only",
        help=(
            "Review only turns flagged for review "
            "by the automatic utterance pipeline."
        ),
    ),
    legacy: bool = typer.Option(
        False,
        "--legacy",
        help="Use the legacy line-based reviewer.",
    ),
):
    """Interactively review reconciled speech turns."""

    storage = storage_for(dataset)
    dataset = storage.root

    if not legacy:
        embedding_names = tuple(
            speaker_embedding
            or (
                "ecapa_speaker",
                "wespeaker_speaker",
            )
        )

        session = ReviewerSession(
            storage,
            source_id=source_id,
            auto_review_only=auto_review_only,
            embedding_names=embedding_names,
        )

        if session.total == 0:
            typer.echo("No speech turns to review.")
            return

        run_reviewer_tui(
            session,
            embedding_names=embedding_names,
            context_padding=context_padding,
        )
        return

    turns = sorted_turns(
        storage,
        source_id=source_id,
    )

    if auto_review_only:
        turns = [
            turn
            for turn in turns
            if (
                turn.get("metadata", {})
                .get("automatic_pipeline", {})
                .get("status")
                == "review"
            )
        ]

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
        assignment = current.get("assignment") or {}

        if (
            speaker_embedding
            and assignment.get(
                "status",
                "unknown",
            ) == "unknown"
        ):
            embeddings = current.get("embeddings") or {}

            voices = {
                voice["id"]: voice
                for voice in storage.voices.load()
            }

            for embedding_name in speaker_embedding:
                if embedding_name not in embeddings:
                    continue

                matches = rank_voice_matches(
                    storage,
                    turn_id,
                    embedding_name,
                )

                typer.echo()
                typer.echo(
                    "speaker evidence "
                    f"[{embedding_name}]"
                )
                typer.echo(
                    format_voice_matches(
                        matches,
                        voices,
                    )
                )
        typer.echo()

        try:
            command = input(
                "[p/r/c/g/f/s/t/l/v/u/i/a/x/k/d/n/b/h/q] > "
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
                        blocking=True,
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
                    blocking=True,
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
                    blocking=True,
                )

                typer.echo(
                    f"Playing context {name}: "
                    f"{start:.3f}-{end:.3f} "
                    f"from {path}"
                )

            elif command == "g":
                metadata = (
                    current.get("metadata")
                    or {}
                )
                alignment = (
                    metadata.get(
                        "alignment_evidence"
                    )
                    or {}
                )
                recovery = (
                    alignment.get("recovery")
                    or {}
                )

                recovery_start = recovery.get(
                    "source_start"
                )
                recovery_end = recovery.get(
                    "source_end"
                )

                if not isinstance(
                    recovery_start,
                    (int, float),
                ) or not isinstance(
                    recovery_end,
                    (int, float),
                ):
                    raise ValueError(
                        "Turn has no alignment "
                        "recovery suggestion"
                    )

                source = storage.get_source(
                    current["source_id"]
                )

                if source is None:
                    raise KeyError(
                        "Source does not exist: "
                        f"{current['source_id']}"
                    )

                recovery_turn = {
                    **current,
                    "source_start": float(
                        recovery_start
                    ),
                    "source_end": float(
                        recovery_end
                    ),
                }

                (
                    name,
                    path,
                    start,
                    end,
                ) = play_turn_context(
                    source,
                    recovery_turn,
                    padding=context_padding,
                    blocking=True,
                )

                typer.echo(
                    f"Playing recovery {name}: "
                    f"{start:.3f}-{end:.3f} "
                    f"from {path}"
                )
            elif command == "f":
                metadata = (
                    current.get("metadata")
                    or {}
                )
                alignment = (
                    metadata.get(
                        "alignment_evidence"
                    )
                    or {}
                )
                recovery = (
                    alignment.get("recovery")
                    or {}
                )

                if recovery.get("status") != "suggested":
                    raise ValueError(
                        "Turn has no suggested "
                        "alignment recovery"
                    )

                recovery_start = recovery.get(
                    "source_start"
                )
                recovery_end = recovery.get(
                    "source_end"
                )
                recovery_regions = (
                    recovery.get("region_ids")
                    or []
                )

                typer.echo(
                    "Accept alignment recovery:"
                )
                typer.echo(
                    "  range: "
                    f"{float(recovery_start):.3f}-"
                    f"{float(recovery_end):.3f}"
                )
                typer.echo(
                    "  regions: "
                    + ", ".join(recovery_regions)
                )
                typer.echo(
                    "  turn representations and "
                    "embeddings will be invalidated"
                )

                confirmation = input(
                    "Apply recovery? [y/N] "
                ).strip().lower()

                if confirmation == "y":
                    accept_alignment_recovery(
                        storage,
                        turn_id,
                    )

                    typer.echo(
                        "Alignment recovery accepted."
                    )
            elif command == "e":
                metadata = (
                    current.get("metadata")
                    or {}
                )
                edge = (
                    metadata.get("edge_evidence")
                    or {}
                )

                if edge.get("status") != "suggested":
                    raise ValueError(
                        "Turn has no suggested "
                        "edge recovery"
                    )

                edge_kind = edge.get("edge")
                candidate_text = edge.get(
                    "candidate_text"
                )
                whisper_text = edge.get(
                    "whisper_text"
                )

                typer.echo(
                    "Accept edge recovery:"
                )

                if edge_kind == "end":
                    edge_end = edge.get(
                        "source_end"
                    )

                    if not isinstance(
                        edge_end,
                        (int, float),
                    ):
                        raise ValueError(
                            "Edge recovery has "
                            "invalid source_end"
                        )

                    typer.echo(
                        "  end: "
                        f"{float(current['source_end']):.3f}"
                        " -> "
                        f"{float(edge_end):.3f}"
                    )
                    typer.echo(
                        "  transcript:"
                    )
                    typer.echo(
                        f"    {candidate_text}"
                    )
                    typer.echo(
                        "  ->"
                    )
                    typer.echo(
                        f"    {whisper_text}"
                    )

                elif edge_kind == "start":
                    edge_start = edge.get(
                        "source_start"
                    )

                    if not isinstance(
                        edge_start,
                        (int, float),
                    ):
                        raise ValueError(
                            "Edge recovery has "
                            "invalid source_start"
                        )

                    typer.echo(
                        "  start: "
                        f"{float(current['source_start']):.3f}"
                        " -> "
                        f"{float(edge_start):.3f}"
                    )
                    typer.echo(
                        "  transcript unchanged:"
                    )
                    typer.echo(
                        f"    {current.get('transcript')}"
                    )
                    typer.echo(
                        "  boundary evidence:"
                    )
                    typer.echo(
                        f"    {whisper_text}"
                    )

                else:
                    raise ValueError(
                        "Edge recovery has "
                        "invalid edge"
                    )

                typer.echo(
                    "  turn representations and "
                    "embeddings will be invalidated"
                )

                confirmation = input(
                    "Apply recovery? [y/N] "
                ).strip().lower()

                if confirmation == "y":
                    accept_edge_recovery(
                        storage,
                        turn_id,
                    )

                    typer.echo(
                        "Edge recovery accepted."
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
                mark_turn_rejected(
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

            elif command == "k":
                mark_turn_boundary_complete(
                    storage,
                    turn_id,
                )

            elif command == "d":
                mark_turn_boundary_clipped(
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

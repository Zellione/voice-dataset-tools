from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from pyannote.audio import Pipeline
from pyannote.audio.utils.reproducibility import (
    ReproducibilityWarning,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_ID = "pyannote/speaker-diarization-community-1"
MODEL_REVISION = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"

MODEL_DIR = (
    PROJECT_ROOT
    / "tools"
    / "pyannote"
    / "models"
    / "speaker-diarization-community-1"
)

DEVICE = torch.device("cuda")


def annotation_to_regions(
    annotation: Any,
) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []

    for turn, speaker in annotation:
        regions.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "label": str(speaker),
            }
        )

    return regions


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run pinned Pyannote Community-1 "
            "speaker diarization and emit "
            "voice-dataset detector regions."
        )
    )

    parser.add_argument(
        "audio",
        type=Path,
        help="Audio file to diarize.",
    )

    parser.add_argument(
        "output",
        type=Path,
        help="Detector output JSON file.",
    )

    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Exact number of speakers.",
    )

    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Minimum number of speakers.",
    )

    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum number of speakers.",
    )

    args = parser.parse_args()

    audio = args.audio.resolve()
    output = args.output.resolve()

    if not audio.is_file():
        raise SystemExit(
            f"Audio file does not exist: {audio}"
        )

    if not MODEL_DIR.is_dir():
        raise SystemExit(
            f"Community-1 model does not exist: "
            f"{MODEL_DIR}\n"
            "Run ./scripts/bootstrap first."
        )

    if args.num_speakers is not None and (
        args.min_speakers is not None
        or args.max_speakers is not None
    ):
        raise SystemExit(
            "--num-speakers cannot be combined with "
            "--min-speakers or --max-speakers."
        )

    if (
        args.min_speakers is not None
        and args.max_speakers is not None
        and args.min_speakers > args.max_speakers
    ):
        raise SystemExit(
            "--min-speakers cannot be greater than "
            "--max-speakers."
        )

    info = sf.info(audio)

    print(
        f"Loading Community-1 from {MODEL_DIR}..."
    )

    pipeline = Pipeline.from_pretrained(
        MODEL_DIR
    )

    pipeline.to(DEVICE)

    parameters: dict[str, int] = {}

    if args.num_speakers is not None:
        parameters["num_speakers"] = (
            args.num_speakers
        )

    if args.min_speakers is not None:
        parameters["min_speakers"] = (
            args.min_speakers
        )

    if args.max_speakers is not None:
        parameters["max_speakers"] = (
            args.max_speakers
        )

    print(f"Device: {DEVICE}")
    print(f"Audio: {audio}")
    print("Running diarization...")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=ReproducibilityWarning,
        )

        warnings.filterwarnings(
            "ignore",
            message=(
                r"std\(\): degrees of freedom "
                r"is <= 0\..*"
            ),
            category=UserWarning,
            module=(
                r"pyannote\.audio\.models\."
                r"blocks\.pooling"
            ),
        )

        result = pipeline(
            audio,
            **parameters,
        )

    regions = annotation_to_regions(
        result.speaker_diarization
    )

    payload = {
        "format": (
            "voice-dataset-detector-output"
        ),
        "version": 1,
        "detector": {
            "name": "pyannote-community-1",
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "parameters": parameters,
        },
        "source": {
            "path": str(audio),
            "sample_rate": int(
                info.samplerate
            ),
            "channels": int(
                info.channels
            ),
            "duration": float(
                info.duration
            ),
        },
        "regions": regions,
    }

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = output.with_name(
        f".{output.name}.tmp"
    )

    try:
        with temporary.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                ensure_ascii=False,
            )

            handle.write("\n")

        temporary.replace(output)

    finally:
        temporary.unlink(
            missing_ok=True,
        )

    print(
        f"Wrote {len(regions)} detector regions "
        f"to {output}"
    )


if __name__ == "__main__":
    main()

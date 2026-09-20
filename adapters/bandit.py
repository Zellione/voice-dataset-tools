from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BANDIT_SOURCE_DIR = (
    PROJECT_ROOT
    / "tools"
    / "bandit"
    / "source"
)

BANDIT_INFERENCE = BANDIT_SOURCE_DIR / "simple_inference.py"

CHECKPOINT = (
    PROJECT_ROOT
    / "tools"
    / "bandit"
    / "models"
    / "checkpoint-multi.ckpt"
)

BANDIT_REVISION = "e6bdc5bf56abc10d72f09fb6214c224e0da5864b"
CHECKPOINT_SHA256 = (
    "abcfccf65446752a057f4a302c941479a54b7560ebf8d7bca039d2ea98e64cfc"
)


def load_inference_module() -> ModuleType:
    if not BANDIT_INFERENCE.is_file():
        raise SystemExit(
            f"BandIt source does not exist: {BANDIT_INFERENCE}\n"
            "Run ./scripts/bootstrap first."
        )

    sys.path.insert(0, str(BANDIT_SOURCE_DIR))

    spec = importlib.util.spec_from_file_location(
        "voice_dataset_bandit_inference",
        BANDIT_INFERENCE,
    )

    if spec is None or spec.loader is None:
        raise SystemExit(
            f"Could not load BandIt inference module: {BANDIT_INFERENCE}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pinned BandIt v2 source separation."
    )
    parser.add_argument(
        "audio",
        type=Path,
        help="Audio file to separate.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output directory.",
    )
    parser.add_argument(
        "--chunk",
        type=float,
        default=30.0,
        help="Chunk size in seconds.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=2.0,
        help="Chunk overlap in seconds.",
    )
    args = parser.parse_args()

    audio = args.audio.resolve()
    output = args.output.resolve()

    if not audio.is_file():
        raise SystemExit(f"Audio file does not exist: {audio}")

    if not CHECKPOINT.is_file():
        raise SystemExit(
            f"BandIt checkpoint does not exist: {CHECKPOINT}\n"
            "Run ./scripts/bootstrap first."
        )

    if args.chunk <= 0:
        raise SystemExit("--chunk must be greater than zero.")

    if args.overlap < 0:
        raise SystemExit("--overlap cannot be negative.")

    if args.overlap >= args.chunk:
        raise SystemExit("--overlap must be smaller than --chunk.")

    if output.exists():
        raise SystemExit(
            f"Output already exists: {output}"
        )

    inference = load_inference_module()

    print(f"BandIt revision: {BANDIT_REVISION}")
    print(f"Checkpoint SHA-256: {CHECKPOINT_SHA256}")

    inference.run_inference(
        checkpoint_path=str(CHECKPOINT),
        audio_path=str(audio),
        output_dir=str(output),
        stems=["speech", "music", "sfx"],
        fs=48000,
        device="cuda",
        use_half=True,
        chunk_seconds=args.chunk,
        overlap_seconds=args.overlap,
    )


if __name__ == "__main__":
    main()

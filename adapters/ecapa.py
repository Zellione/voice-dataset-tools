import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = (
    PROJECT_ROOT
    / "tools"
    / "ecapa"
    / "models"
    / "spkrec-ecapa-voxceleb"
)

DEVICE = "cuda:0"
TARGET_SAMPLE_RATE = 16000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export SpeechBrain ECAPA speaker "
            "embeddings from a voice dataset."
        )
    )

    parser.add_argument(
        "dataset",
        type=Path,
        help="Voice dataset directory.",
    )

    parser.add_argument(
        "output",
        type=Path,
        help="Output directory.",
    )

    parser.add_argument(
        "--record-type",
        choices=("region", "turn"),
        default="region",
        help=(
            "Dataset record type to embed "
            "(default: region)."
        ),
    )

    parser.add_argument(
        "--representation",
        default="raw",
        help=(
            "Audio representation to embed "
            "(default: raw)."
        ),
    )

    return parser.parse_args()


def load_records(
    dataset: Path,
    record_type: str,
) -> list[dict]:
    if record_type == "region":
        filename = "regions.jsonl"
    elif record_type == "turn":
        filename = "turns.jsonl"
    else:
        raise ValueError(
            f"Unsupported record type: {record_type}"
        )

    path = dataset / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"{filename} does not exist: {path}"
        )

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: "
                    f"invalid JSON: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{line_number}: "
                    "record must be an object"
                )

            record_id = record.get("id")

            if (
                not isinstance(record_id, str)
                or not record_id
            ):
                raise ValueError(
                    f"{path}:{line_number}: "
                    "record id must be a "
                    "non-empty string"
                )

            records.append(record)

    return records


def resolve_audio(
    dataset: Path,
    record: dict,
    representation: str,
) -> Path:
    record_id = record["id"]

    representations = record.get(
        "representations"
    )

    if not isinstance(
        representations,
        dict,
    ):
        raise ValueError(
            f"{record_id}: representations "
            "must be an object"
        )

    audio = representations.get(
        representation
    )

    if not isinstance(audio, dict):
        raise ValueError(
            f"{record_id}: representation "
            f"does not exist: {representation}"
        )

    relative_path = audio.get("path")

    if (
        not isinstance(relative_path, str)
        or not relative_path
    ):
        raise ValueError(
            f"{record_id}: representation "
            "path must be a non-empty string"
        )

    path = (
        dataset / relative_path
    ).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"{record_id}: audio does not exist: "
            f"{path}"
        )

    return path


def load_audio(
    path: Path,
) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = (
        torchaudio.load(path)
    )

    if waveform.shape[0] > 1:
        waveform = waveform.mean(
            dim=0,
            keepdim=True,
        )

    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = (
            torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=TARGET_SAMPLE_RATE,
            )
        )

    return waveform, sample_rate


def encode(
    classifier: EncoderClassifier,
    waveform: torch.Tensor,
) -> np.ndarray:
    waveform = waveform.to(DEVICE)

    with torch.inference_mode():
        embedding = classifier.encode_batch(
            waveform
        )

    embedding = embedding.reshape(-1)
    embedding = embedding.float()
    embedding = F.normalize(
        embedding,
        dim=0,
    )

    vector = (
        embedding
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    if vector.ndim != 1:
        raise RuntimeError(
            "ECAPA produced a non-vector "
            f"embedding: {vector.shape}"
        )

    if not np.isfinite(vector).all():
        raise RuntimeError(
            "ECAPA produced non-finite values"
        )

    return vector


def main() -> None:
    args = parse_args()

    dataset = args.dataset.resolve()
    output = args.output.resolve()

    if not dataset.is_dir():
        raise FileNotFoundError(
            f"Dataset does not exist: {dataset}"
        )

    if output.exists():
        raise FileExistsError(
            "Output already exists; refusing "
            f"to overwrite it: {output}"
        )

    records = load_records(
        dataset,
        args.record_type,
    )

    if not records:
        raise ValueError(
            f"Dataset contains no "
            f"{args.record_type} records"
        )

    # Validate every input before loading the
    # model or creating output files.
    inputs = []
    seen_ids = set()

    for record in records:
        record_id = record["id"]

        if record_id in seen_ids:
            raise ValueError(
                f"Duplicate {args.record_type} "
                f"id: {record_id}"
            )

        seen_ids.add(record_id)

        audio_path = resolve_audio(
            dataset,
            record,
            args.representation,
        )

        inputs.append(
            (
                record_id,
                audio_path,
            )
        )

    print("Loading ECAPA-TDNN...")

    classifier = (
        EncoderClassifier.from_hparams(
            source=str(MODEL_DIR),
            overrides={
                "pretrained_path": str(MODEL_DIR),
            },
            run_opts={
                "device": DEVICE,
            },
        )
    )

    temporary = output.with_name(
        f".{output.name}.tmp"
    )

    if temporary.exists():
        shutil.rmtree(temporary)

    temporary.mkdir(
        parents=True,
    )

    manifest_embeddings = []

    id_field = (
        "region_id"
        if args.record_type == "region"
        else "turn_id"
    )

    try:
        print(
            f"Generating {len(inputs)} "
            f"{args.representation} "
            f"{args.record_type} embeddings:"
        )

        for index, (
            record_id,
            audio_path,
        ) in enumerate(
            inputs,
            start=1,
        ):
            waveform, source_sample_rate = (
                load_audio(audio_path)
            )

            duration = (
                waveform.shape[-1]
                / TARGET_SAMPLE_RATE
            )

            vector = encode(
                classifier,
                waveform,
            )

            filename = (
                f"{record_id}.npy"
            )

            np.save(
                temporary / filename,
                vector,
                allow_pickle=False,
            )

            manifest_embeddings.append(
                {
                    id_field: record_id,
                    "path": filename,
                    "dimension": int(
                        vector.shape[0]
                    ),
                    "metadata": {
                        "source_sample_rate": (
                            source_sample_rate
                        ),
                        "encoder_sample_rate": (
                            TARGET_SAMPLE_RATE
                        ),
                        "duration_seconds": (
                            duration
                        ),
                        "l2_normalized": True,
                    },
                }
            )

            print(
                f"[{index:02d}/{len(inputs):02d}] "
                f"{record_id}  "
                f"{duration:.3f}s  "
                f"dim={vector.shape[0]}"
            )

        manifest = {
            "format": (
                "voice-dataset-embedding-output"
            ),
            "version": 1,
            "record_type": args.record_type,
            "encoder": {
                "name": "speechbrain-ecapa",
                "model": MODEL_ID,
                "revision": MODEL_REVISION,
            },
            "representation": (
                args.representation
            ),
            "embeddings": (
                manifest_embeddings
            ),
        }

        with (
            temporary / "embeddings.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                manifest,
                file,
                indent=2,
            )
            file.write("\n")

        temporary.rename(output)

    except Exception:
        shutil.rmtree(
            temporary,
            ignore_errors=True,
        )
        raise

    print()
    print(
        f"Wrote {len(manifest_embeddings)} "
        f"embeddings to {output}"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
ALIGNER_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
LANGUAGE_ALIASES = {
    "en": "English",
    "de": "German",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    input_group = parser.add_mutually_exclusive_group(
        required=True,
    )

    input_group.add_argument(
        "--input",
        type=Path,
    )

    input_group.add_argument(
        "--input-manifest",
        type=Path,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--language",
        default=None,
    )

    parser.add_argument(
        "--mode",
        choices=("transcribe", "align"),
        default="transcribe",
    )

    parser.add_argument(
        "--text",
        default=None,
    )

    parser.add_argument(
        "--max-inference-batch-size",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    if (
        args.max_inference_batch_size
        <= 0
    ):
        parser.error(
            "--max-inference-batch-size "
            "must be positive"
        )

    if args.mode == "align" and not args.text:
        parser.error(
            "--text is required when --mode=align"
        )

    if (
        args.mode == "align"
        and args.input is None
    ):
        parser.error(
            "--mode=align requires --input"
        )

    if (
        args.mode == "align"
        and args.input_manifest is not None
    ):
        parser.error(
            "--input-manifest is only valid "
            "for transcription"
        )

    return args


def _alignment_words(
    alignment,
) -> list[dict]:
    words = []

    for item in alignment.items:
        words.append(
            {
                "text": item.text,
                "start": float(item.start_time),
                "end": float(item.end_time),
            }
        )

    return words


def main() -> None:
    args = parse_args()

    if (
        args.input is not None
        and not args.input.is_file()
    ):
        raise ValueError(
            f"Input audio does not exist: {args.input}"
        )

    if (
        args.input_manifest is not None
        and not args.input_manifest.is_file()
    ):
        raise ValueError(
            "Input manifest does not exist: "
            f"{args.input_manifest}"
        )

    from transformers.utils import logging

    logging.set_verbosity_error()
    logging.disable_progress_bar()

    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(
        MODEL_ID,
        forced_aligner=ALIGNER_ID,
        forced_aligner_kwargs={
            "device_map": "cuda:0",
            "dtype": "bfloat16",
        },
        device_map="cuda:0",
        dtype="bfloat16",
        max_inference_batch_size=(
            args.max_inference_batch_size
        ),
    )

    language = args.language

    if language is not None:
        language = LANGUAGE_ALIASES.get(
            language.casefold(),
            language,
        )

    if args.mode == "align":
        alignments = model.forced_aligner.align(
            audio=str(args.input),
            text=args.text,
            language=language,
        )

        if len(alignments) != 1:
            raise RuntimeError(
                "Qwen returned unexpected forced alignment "
                f"batch size: {len(alignments)}"
            )

        alignment = alignments[0]

        document = {
            "format": "voice-dataset-forced-alignment-output",
            "version": 1,
            "aligner": {
                "name": "qwen3-forced-aligner",
                "model": ALIGNER_ID,
            },
            "language": language,
            "text": args.text,
            "words": _alignment_words(
                alignment
            ),
        }

    else:
        if args.input_manifest is None:
            results = model.transcribe(
                audio=str(args.input),
                language=language,
                return_time_stamps=True,
            )

            result = results[0]

            alignment = result.time_stamps

            if alignment is None:
                raise RuntimeError(
                    "Qwen returned no forced alignment"
                )

            document = {
                "format": (
                    "voice-dataset-continuous-asr-output"
                ),
                "version": 1,
                "transcriber": {
                    "name": "qwen3-asr",
                    "model": MODEL_ID,
                },
                "aligner": {
                    "name": "qwen3-forced-aligner",
                    "model": ALIGNER_ID,
                },
                "language": result.language,
                "text": result.text,
                "words": _alignment_words(
                    alignment
                ),
            }

        else:
            manifest = json.loads(
                args.input_manifest.read_text(
                    encoding="utf-8"
                )
            )

            raw_chunks = manifest.get(
                "chunks"
            )

            if not isinstance(
                raw_chunks,
                list,
            ):
                raise ValueError(
                    "Input manifest has invalid chunks"
                )

            audio = []
            languages = []

            for item in raw_chunks:
                if not isinstance(item, dict):
                    raise ValueError(
                        "Input manifest chunk "
                        "must be an object"
                    )

                path = Path(
                    item["path"]
                )

                if not path.is_file():
                    raise ValueError(
                        "Chunk audio does not exist: "
                        f"{path}"
                    )

                audio.append(str(path))
                languages.append(language)

            results = model.transcribe(
                audio=audio,
                language=(
                    languages
                    if language is not None
                    else None
                ),
                return_time_stamps=True,
            )

            if len(results) != len(raw_chunks):
                raise RuntimeError(
                    "Qwen returned unexpected "
                    "batch size"
                )

            chunks = []

            for item, result in zip(
                raw_chunks,
                results,
            ):
                alignment = (
                    result.time_stamps
                )

                if alignment is None:
                    raise RuntimeError(
                        "Qwen returned no forced "
                        "alignment for chunk "
                        f"{item['index']}"
                    )

                chunks.append(
                    {
                        "index": int(
                            item["index"]
                        ),
                        "start": float(
                            item["start"]
                        ),
                        "end": float(
                            item["end"]
                        ),
                        "boundary": str(
                            item["boundary"]
                        ),
                        "language": (
                            result.language
                        ),
                        "text": result.text,
                        "words": _alignment_words(
                            alignment
                        ),
                    }
                )

            document = {
                "format": (
                    "voice-dataset-continuous-asr-output"
                ),
                "version": 2,
                "transcriber": {
                    "name": "qwen3-asr",
                    "model": MODEL_ID,
                },
                "aligner": {
                    "name": "qwen3-forced-aligner",
                    "model": ALIGNER_ID,
                },
                "chunks": chunks,
            }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

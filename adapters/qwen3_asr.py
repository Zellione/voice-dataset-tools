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

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
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

    args = parser.parse_args()

    if args.mode == "align" and not args.text:
        parser.error(
            "--text is required when --mode=align"
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

    if not args.input.is_file():
        raise ValueError(
            f"Input audio does not exist: {args.input}"
        )

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
        result = model.transcribe(
            audio=str(args.input),
            language=language,
            return_time_stamps=True,
        )[0]

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

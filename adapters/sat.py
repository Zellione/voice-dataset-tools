from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODEL_ID = "sat-3l-sm"


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

    return parser.parse_args()


def load_words(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(
            f"Input does not exist: {path}"
        )

    document = json.loads(
        path.read_text(encoding="utf-8")
    )

    words = document.get("words")

    if not isinstance(words, list):
        raise ValueError(
            "Input document must contain a words list"
        )

    return words


def build_text(
    words: list[dict[str, Any]],
) -> tuple[
    str,
    list[tuple[int, int]],
]:
    parts: list[str] = []
    spans: list[tuple[int, int]] = []

    offset = 0

    for index, word in enumerate(words):
        text = str(word["text"]).strip()

        if not text:
            raise ValueError(
                f"Word {index} is empty"
            )

        if index:
            parts.append(" ")
            offset += 1

        start = offset

        parts.append(text)
        offset += len(text)

        spans.append(
            (start, offset)
        )

    return "".join(parts), spans


def segment_end_offsets(
    segments: list[str],
) -> list[int]:
    offsets: list[int] = []
    offset = 0

    for segment in segments:
        offset += len(segment)
        offsets.append(offset)

    return offsets


def map_boundaries_to_words(
    text: str,
    word_spans: list[tuple[int, int]],
    segments: list[str],
) -> list[int]:
    reconstructed = "".join(segments)

    if reconstructed != text:
        raise RuntimeError(
            "SaT segments do not reconstruct input text"
        )

    boundaries: list[int] = []

    for offset in segment_end_offsets(segments)[:-1]:
        boundary = offset

        while (
            boundary > 0
            and text[boundary - 1].isspace()
        ):
            boundary -= 1

        matching = [
            index
            for index, (_, end) in enumerate(
                word_spans
            )
            if end == boundary
        ]

        if len(matching) != 1:
            raise RuntimeError(
                "SaT boundary does not fall on a "
                f"word boundary: char={boundary}"
            )

        boundaries.append(
            matching[0]
        )

    return boundaries


def main() -> None:
    args = parse_args()

    words = load_words(args.input)

    text, word_spans = build_text(words)

    from wtpsplit import SaT
    from transformers.utils import logging

    logging.disable_progress_bar()

    model = SaT(MODEL_ID)

    segments = list(
        model.split(
            text,
            do_paragraph_segmentation=False,
        )
    )

    boundary_after_word_indices = (
        map_boundaries_to_words(
            text,
            word_spans,
            segments,
        )
    )

    document = {
        "format": "voice-dataset-utterance-boundary-output",
        "version": 1,
        "segmenter": {
            "name": "sat",
            "model": MODEL_ID,
        },
        "word_count": len(words),
        "boundary_after_word_indices": (
            boundary_after_word_indices
        ),
        "segments": [
            {
                "start_word_index": (
                    0
                    if index == 0
                    else (
                        boundary_after_word_indices[
                            index - 1
                        ]
                        + 1
                    )
                ),
                "end_word_index": (
                    boundary_after_word_indices[index]
                    if index
                    < len(
                        boundary_after_word_indices
                    )
                    else len(words) - 1
                ),
                "text": segment.strip(),
            }
            for index, segment in enumerate(
                segments
            )
        ],
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

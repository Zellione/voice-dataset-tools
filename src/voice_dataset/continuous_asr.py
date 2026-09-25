from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sources import resolve_source_representation
from .storage import DatasetStorage
from .workers import run_worker, worker


QWEN3_ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"

_SENTENCE_RE = re.compile(r".+?(?:[.!?]+(?=\s|$)|$)", re.DOTALL)
_TOKEN_RE = re.compile(r"\w+(?:['’]\w+)*", re.UNICODE)


@dataclass(frozen=True)
class ContinuousAsrResult:
    language: str | None
    text: str
    words: list[dict[str, Any]]
    utterances: list[dict[str, Any]]


@dataclass(frozen=True)
class ForcedAlignmentResult:
    language: str | None
    text: str
    words: list[dict[str, Any]]


def _tokens(text: str) -> list[str]:
    return [
        match.group(0).casefold().replace("’", "'")
        for match in _TOKEN_RE.finditer(text)
    ]


def _derive_utterances(
    text: str,
    words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sentences = [
        match.group(0).strip()
        for match in _SENTENCE_RE.finditer(text)
        if match.group(0).strip()
    ]

    def lexical_text(value: str) -> str:
        return "".join(_tokens(value))

    text_lexical = lexical_text(text)
    aligned_lexical = "".join(
        lexical_text(word["text"])
        for word in words
    )

    if text_lexical != aligned_lexical:
        mismatch = next(
            (
                index
                for index, (text_char, aligned_char)
                in enumerate(
                    zip(text_lexical, aligned_lexical)
                )
                if text_char != aligned_char
            ),
            min(
                len(text_lexical),
                len(aligned_lexical),
            ),
        )

        start = max(0, mismatch - 30)
        end = mismatch + 31

        raise ValueError(
            "Continuous ASR text does not match "
            "aligned word sequence at lexical character "
            f"{mismatch}: "
            f"text={text_lexical[start:end]!r}, "
            f"aligned={aligned_lexical[start:end]!r}; "
            f"character counts: "
            f"text={len(text_lexical)}, "
            f"aligned={len(aligned_lexical)}"
        )

    utterances: list[dict[str, Any]] = []
    word_index = 0

    for sentence in sentences:
        sentence_lexical = lexical_text(sentence)

        if not sentence_lexical:
            continue

        start_index = word_index
        aligned_sentence = ""

        while (
            word_index < len(words)
            and len(aligned_sentence)
            < len(sentence_lexical)
        ):
            aligned_sentence += lexical_text(
                words[word_index]["text"]
            )
            word_index += 1

            if not sentence_lexical.startswith(
                aligned_sentence
            ):
                raise ValueError(
                    "Continuous ASR sentence does not "
                    "match aligned word sequence: "
                    f"{sentence!r}"
                )

        if aligned_sentence != sentence_lexical:
            raise ValueError(
                "Continuous ASR sentence has incomplete "
                f"word alignment: {sentence!r}"
            )

        utterances.append(
            {
                "text": sentence,
                "word_start": start_index,
                "word_end": word_index,
            }
        )

    if word_index != len(words):
        raise ValueError(
            "Continuous ASR has aligned words outside "
            "derived utterances"
        )

    return utterances


def attach_continuous_asr(
    storage: DatasetStorage,
    source_id: str,
    name: str,
    *,
    model: str,
    representation: str,
    language: str | None,
    text: str,
    words: list[dict[str, Any]],
    utterances: list[dict[str, Any]],
) -> dict[str, Any]:
    if not name:
        raise ValueError(
            "Continuous ASR name must not be empty"
        )

    source = storage.get_source(source_id)

    if source is None:
        raise KeyError(
            f"Source does not exist: {source_id}"
        )

    representations = source.get(
        "representations",
        {},
    )

    if not isinstance(representations, dict):
        raise ValueError(
            f"Source has invalid representations: "
            f"{source_id}"
        )

    if representation not in representations:
        raise ValueError(
            f"Source representation does not exist: "
            f"{source_id}/{representation}"
        )

    evidence = {
        "model": model,
        "representation": representation,
        "language": language,
        "text": text,
        "words": [
            {
                "text": word["text"],
                "start": float(word["start"]),
                "end": float(word["end"]),
            }
            for word in words
        ],
        "utterances": [
            {
                "text": utterance["text"],
                "word_start": int(
                    utterance["word_start"]
                ),
                "word_end": int(
                    utterance["word_end"]
                ),
            }
            for utterance in utterances
        ],
    }

    def update(
        record: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = record.get("metadata", {})

        if not isinstance(metadata, dict):
            raise ValueError(
                f"Source has invalid metadata: "
                f"{source_id}"
            )

        continuous_asr = metadata.get(
            "continuous_asr",
            {},
        )

        if not isinstance(continuous_asr, dict):
            raise ValueError(
                f"Source has invalid continuous_asr "
                f"metadata: {source_id}"
            )

        existing = continuous_asr.get(name)

        if existing is not None:
            if existing == evidence:
                return record

            raise ValueError(
                "Continuous ASR evidence already exists "
                "with different data: "
                f"{source_id}/{name}"
            )

        continuous_asr[name] = evidence
        metadata["continuous_asr"] = continuous_asr
        record["metadata"] = metadata

        return record

    return storage.update_source(
        source_id,
        update,
    )


def _load_qwen_output(
    path: Path,
) -> ContinuousAsrResult:
    document = json.loads(
        path.read_text(encoding="utf-8")
    )

    if (
        document.get("format")
        != "voice-dataset-continuous-asr-output"
    ):
        raise ValueError(
            "Invalid continuous ASR output format"
        )

    if document.get("version") != 1:
        raise ValueError(
            "Unsupported continuous ASR output version"
        )

    transcriber = document.get("transcriber")

    if not isinstance(transcriber, dict):
        raise ValueError(
            "Continuous ASR output has no transcriber"
        )

    if transcriber.get("model") != QWEN3_ASR_MODEL:
        raise ValueError(
            "Unexpected Qwen ASR model: "
            f"{transcriber.get('model')!r}"
        )

    text = document.get("text")

    if not isinstance(text, str):
        raise ValueError(
            "Continuous ASR text must be a string"
        )

    language = document.get("language")

    if (
        language is not None
        and not isinstance(language, str)
    ):
        raise ValueError(
            "Continuous ASR language must be "
            "a string or null"
        )

    words_value = document.get("words")

    if not isinstance(words_value, list):
        raise ValueError(
            "Continuous ASR words must be a list"
        )

    words: list[dict[str, Any]] = []

    for index, item in enumerate(words_value):
        if not isinstance(item, dict):
            raise ValueError(
                f"Continuous ASR word {index} "
                "must be an object"
            )

        word_text = item.get("text")
        start = item.get("start")
        end = item.get("end")

        if not isinstance(word_text, str):
            raise ValueError(
                f"Continuous ASR word {index} "
                "has invalid text"
            )

        if (
            isinstance(start, bool)
            or not isinstance(start, (int, float))
            or isinstance(end, bool)
            or not isinstance(end, (int, float))
        ):
            raise ValueError(
                f"Continuous ASR word {index} "
                "has invalid timestamps"
            )

        start = float(start)
        end = float(end)

        if start < 0 or end < start:
            raise ValueError(
                f"Continuous ASR word {index} "
                "has invalid range: "
                f"{start}-{end}"
            )

        words.append(
            {
                "text": word_text,
                "start": start,
                "end": end,
            }
        )

    utterances = _derive_utterances(
        text,
        words,
    )

    return ContinuousAsrResult(
        language=language,
        text=text,
        words=words,
        utterances=utterances,
    )


def _load_qwen_alignment_output(
    path: Path,
) -> ForcedAlignmentResult:
    document = json.loads(
        path.read_text(encoding="utf-8")
    )

    if (
        document.get("format")
        != "voice-dataset-forced-alignment-output"
    ):
        raise ValueError(
            "Unexpected Qwen forced alignment "
            "output format"
        )

    if document.get("version") != 1:
        raise ValueError(
            "Unsupported Qwen forced alignment "
            "output version"
        )

    language = document.get("language")

    if (
        language is not None
        and not isinstance(language, str)
    ):
        raise ValueError(
            "Qwen forced alignment has invalid language"
        )

    text = document.get("text")

    if not isinstance(text, str):
        raise ValueError(
            "Qwen forced alignment has invalid text"
        )

    raw_words = document.get("words")

    if not isinstance(raw_words, list):
        raise ValueError(
            "Qwen forced alignment has invalid words"
        )

    words: list[dict[str, Any]] = []

    for index, raw_word in enumerate(raw_words):
        if not isinstance(raw_word, dict):
            raise ValueError(
                "Qwen forced alignment word "
                f"{index} is invalid"
            )

        word_text = raw_word.get("text")
        start = raw_word.get("start")
        end = raw_word.get("end")

        if not isinstance(word_text, str):
            raise ValueError(
                "Qwen forced alignment word "
                f"{index} has invalid text"
            )

        if not isinstance(start, (int, float)):
            raise ValueError(
                "Qwen forced alignment word "
                f"{index} has invalid start"
            )

        if not isinstance(end, (int, float)):
            raise ValueError(
                "Qwen forced alignment word "
                f"{index} has invalid end"
            )

        start = float(start)
        end = float(end)

        if start < 0 or end < start:
            raise ValueError(
                "Qwen forced alignment word "
                f"{index} has invalid range: "
                f"{start}-{end}"
            )

        words.append(
            {
                "text": word_text,
                "start": start,
                "end": end,
            }
        )

    if "".join(_tokens(text)) != "".join(
        "".join(_tokens(word["text"]))
        for word in words
    ):
        raise ValueError(
            "Qwen forced alignment text does not match "
            "aligned word sequence"
        )

    return ForcedAlignmentResult(
        language=language,
        text=text,
        words=words,
    )


def align_text_qwen3(
    audio_path: Path,
    *,
    text: str,
    language: str,
) -> ForcedAlignmentResult:
    if not text.strip():
        raise ValueError(
            "Forced alignment text must not be empty"
        )

    with tempfile.TemporaryDirectory(
        prefix="voice-dataset-qwen3-align-"
    ) as temporary_directory:
        output_path = (
            Path(temporary_directory)
            / "forced-alignment.json"
        )

        run_worker(
            worker("qwen3-asr"),
            [
                "--mode",
                "align",
                "--input",
                audio_path,
                "--output",
                output_path,
                "--language",
                language,
                "--text",
                text,
            ],
        )

        return _load_qwen_alignment_output(
            output_path
        )


def transcribe_source_qwen3(
    storage: DatasetStorage,
    source_id: str,
    *,
    representation_name: str,
    evidence_name: str = "qwen3",
    language: str | None = None,
) -> ContinuousAsrResult:
    _, source_path = resolve_source_representation(
        storage,
        source_id,
        representation_name,
    )

    with tempfile.TemporaryDirectory(
        prefix="voice-dataset-qwen3-"
    ) as temporary_directory:
        output_path = (
            Path(temporary_directory)
            / "continuous-asr.json"
        )

        arguments: list[str | Path] = [
            "--input",
            source_path,
            "--output",
            output_path,
        ]

        if language is not None:
            arguments.extend(
                [
                    "--language",
                    language,
                ]
            )

        run_worker(
            worker("qwen3-asr"),
            arguments,
        )

        result = _load_qwen_output(
            output_path
        )

    attach_continuous_asr(
        storage,
        source_id,
        evidence_name,
        model=QWEN3_ASR_MODEL,
        representation=representation_name,
        language=result.language,
        text=result.text,
        words=result.words,
        utterances=result.utterances,
    )

    return result

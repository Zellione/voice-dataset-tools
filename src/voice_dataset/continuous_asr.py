from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .media import extract_audio_region
from .sources import resolve_source_representation
from .storage import DatasetStorage
from .workers import run_worker, worker


QWEN3_ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"

_SENTENCE_RE = re.compile(r".+?(?:[.!?]+(?=\s|$)|$)", re.DOTALL)
_TOKEN_RE = re.compile(r"\w+(?:['’]\w+)*", re.UNICODE)


@dataclass(frozen=True)
class ContinuousAsrChunk:
    index: int
    start: float
    end: float
    boundary: str


@dataclass(frozen=True)
class ContinuousAsrResult:
    language: str | None
    text: str
    words: list[dict[str, Any]]
    utterances: list[dict[str, Any]]
    chunks: tuple[ContinuousAsrChunk, ...] = ()


@dataclass(frozen=True)
class ForcedAlignmentResult:
    language: str | None
    text: str
    words: list[dict[str, Any]]


CONTINUOUS_ASR_TARGET_CHUNK_SECONDS = 60.0
CONTINUOUS_ASR_BOUNDARY_SEARCH_SECONDS = 15.0
CONTINUOUS_ASR_MAX_CHUNK_SECONDS = 240.0
CONTINUOUS_ASR_MIN_SPEECH_GAP_SECONDS = 0.50

def _region_whisper_text(
    region: dict[str, Any],
) -> str | None:
    transcripts = region.get("transcripts")

    if not isinstance(transcripts, dict):
        return None

    whisper = transcripts.get("whisper")

    if not isinstance(whisper, dict):
        return None

    text = whisper.get("text")

    if not isinstance(text, str):
        return None

    text = text.strip()

    return text or None


def _region_ends_sentence(
    region: dict[str, Any],
) -> bool:
    text = _region_whisper_text(region)

    if text is None:
        return False

    return text.rstrip().endswith(
        (".", "!", "?")
    )


def plan_continuous_asr_chunks(
    storage: DatasetStorage,
    source_id: str,
    *,
    duration: float,
    target_seconds: float = (
        CONTINUOUS_ASR_TARGET_CHUNK_SECONDS
    ),
    search_seconds: float = (
        CONTINUOUS_ASR_BOUNDARY_SEARCH_SECONDS
    ),
    max_chunk_seconds: float = (
        CONTINUOUS_ASR_MAX_CHUNK_SECONDS
    ),
    minimum_gap_seconds: float = (
        CONTINUOUS_ASR_MIN_SPEECH_GAP_SECONDS
    ),
) -> list[ContinuousAsrChunk]:
    if duration <= 0:
        raise ValueError(
            "Continuous ASR source duration "
            "must be positive"
        )

    if target_seconds <= 0:
        raise ValueError(
            "Continuous ASR target chunk size "
            "must be positive"
        )

    if search_seconds < 0:
        raise ValueError(
            "Continuous ASR boundary search "
            "must not be negative"
        )

    if max_chunk_seconds < target_seconds:
        raise ValueError(
            "Continuous ASR max chunk size "
            "must be at least the target size"
        )

    if minimum_gap_seconds < 0:
        raise ValueError(
            "Continuous ASR minimum gap "
            "must not be negative"
        )

    regions = [
        region
        for region in storage.regions.load()
        if region.get("source_id") == source_id
    ]

    regions.sort(
        key=lambda region: (
            float(region["source_start"]),
            float(region["source_end"]),
            region["id"],
        )
    )

    speech_blocks: list[
        tuple[float, float, dict[str, Any]]
    ] = []

    for region in regions:
        region_start = float(
            region["source_start"]
        )
        region_end = float(
            region["source_end"]
        )

        if region_end <= region_start:
            continue

        if not speech_blocks:
            speech_blocks.append(
                (
                    region_start,
                    region_end,
                    region,
                )
            )
            continue

        (
            block_start,
            block_end,
            block_last_region,
        ) = speech_blocks[-1]

        if region_start <= block_end:
            if region_end >= block_end:
                speech_blocks[-1] = (
                    block_start,
                    region_end,
                    region,
                )
            continue

        speech_blocks.append(
            (
                region_start,
                region_end,
                region,
            )
        )

    boundaries: list[
        tuple[float, str]
    ] = []

    for previous, following in zip(
        speech_blocks,
        speech_blocks[1:],
    ):
        previous_end = previous[1]
        following_start = following[0]
        previous_last_region = previous[2]

        gap = following_start - previous_end

        if gap < minimum_gap_seconds:
            continue

        boundary = (
            previous_end
            + gap / 2.0
        )

        reason = (
            "sentence_gap"
            if _region_ends_sentence(
                previous_last_region
            )
            else "speech_gap"
        )

        boundaries.append(
            (boundary, reason)
        )

    chunks: list[ContinuousAsrChunk] = []
    start = 0.0

    while start < duration:
        remaining = duration - start

        if remaining <= target_seconds + search_seconds:
            chunks.append(
                ContinuousAsrChunk(
                    index=len(chunks),
                    start=start,
                    end=duration,
                    boundary="source_end",
                )
            )
            break

        target = start + target_seconds

        preferred_low = max(
            start + 1.0,
            target - search_seconds,
        )
        preferred_high = min(
            duration,
            target + search_seconds,
            start + max_chunk_seconds,
        )

        preferred = [
            candidate
            for candidate in boundaries
            if (
                preferred_low
                <= candidate[0]
                <= preferred_high
            )
        ]

        if preferred:
            sentence_candidates = [
                candidate
                for candidate in preferred
                if candidate[1] == "sentence_gap"
            ]

            pool = (
                sentence_candidates
                or preferred
            )

            end, reason = min(
                pool,
                key=lambda candidate: abs(
                    candidate[0] - target
                ),
            )
        else:
            extension = [
                candidate
                for candidate in boundaries
                if (
                    preferred_high
                    < candidate[0]
                    <= start + max_chunk_seconds
                )
            ]

            if not extension:
                if duration <= start + max_chunk_seconds:
                    end = duration
                    reason = "source_end"
                else:
                    raise ValueError(
                        "Continuous ASR could not find a "
                        "safe speech boundary between "
                        f"{preferred_low:.3f}s and "
                        f"{start + max_chunk_seconds:.3f}s; "
                        "refusing to split inside continuous "
                        "speech"
                    )
            else:
                sentence_candidates = [
                    candidate
                    for candidate in extension
                    if (
                        candidate[1]
                        == "sentence_gap"
                    )
                ]

                pool = (
                    sentence_candidates
                    or extension
                )

                end, reason = min(
                    pool,
                    key=lambda candidate: (
                        candidate[0]
                    ),
                )

        if end <= start:
            raise RuntimeError(
                "Continuous ASR chunk planner "
                "did not advance"
            )

        chunks.append(
            ContinuousAsrChunk(
                index=len(chunks),
                start=start,
                end=end,
                boundary=reason,
            )
        )

        start = end

    return chunks


def _validate_word_timeline(
    words: list[dict[str, Any]],
) -> None:
    previous_start = -1.0

    for index, word in enumerate(words):
        start = float(word["start"])
        end = float(word["end"])

        if start < 0:
            raise ValueError(
                "Continuous ASR word has negative "
                f"start at index {index}: {start}"
            )

        if end < start:
            raise ValueError(
                "Continuous ASR word has negative "
                f"duration at index {index}: "
                f"{start}-{end}"
            )

        if start + 1e-6 < previous_start:
            raise ValueError(
                "Continuous ASR timestamps are not "
                "monotonic at word "
                f"{index}: {start} < "
                f"{previous_start}"
            )

        previous_start = start


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

    version = document.get("version")

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

    if version == 1:
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

        raw_words = document.get("words")

        if not isinstance(raw_words, list):
            raise ValueError(
                "Continuous ASR words must be a list"
            )

        words = [
            {
                "text": str(item["text"]),
                "start": float(item["start"]),
                "end": float(item["end"]),
            }
            for item in raw_words
        ]

        _validate_word_timeline(words)

        return ContinuousAsrResult(
            language=language,
            text=text,
            words=words,
            utterances=_derive_utterances(
                text,
                words,
            ),
        )

    if version != 2:
        raise ValueError(
            "Unsupported continuous ASR output "
            f"version: {version!r}"
        )

    raw_chunks = document.get("chunks")

    if not isinstance(raw_chunks, list):
        raise ValueError(
            "Chunked continuous ASR output "
            "must contain chunks"
        )

    combined_text: list[str] = []
    combined_words: list[dict[str, Any]] = []
    chunk_records: list[
        ContinuousAsrChunk
    ] = []
    languages: set[str] = set()

    for expected_index, item in enumerate(
        raw_chunks
    ):
        if not isinstance(item, dict):
            raise ValueError(
                "Continuous ASR chunk must "
                "be an object"
            )

        index = item.get("index")
        start = item.get("start")
        end = item.get("end")
        boundary = item.get("boundary")
        language = item.get("language")
        chunk_text = item.get("text")
        raw_words = item.get("words")

        if index != expected_index:
            raise ValueError(
                "Continuous ASR chunk indices "
                "must be contiguous"
            )

        if not isinstance(
            start,
            (int, float),
        ):
            raise ValueError(
                "Continuous ASR chunk has "
                "invalid start"
            )

        if not isinstance(
            end,
            (int, float),
        ):
            raise ValueError(
                "Continuous ASR chunk has "
                "invalid end"
            )

        start = float(start)
        end = float(end)

        if start < 0 or end <= start:
            raise ValueError(
                "Continuous ASR chunk has "
                f"invalid geometry: {start}-{end}"
            )

        if not isinstance(boundary, str):
            raise ValueError(
                "Continuous ASR chunk has "
                "invalid boundary"
            )

        if (
            language is not None
            and not isinstance(language, str)
        ):
            raise ValueError(
                "Continuous ASR chunk has "
                "invalid language"
            )

        if isinstance(language, str):
            languages.add(language)

        if not isinstance(chunk_text, str):
            raise ValueError(
                "Continuous ASR chunk has "
                "invalid text"
            )

        if not isinstance(raw_words, list):
            raise ValueError(
                "Continuous ASR chunk has "
                "invalid words"
            )

        local_words: list[
            dict[str, Any]
        ] = []

        for word in raw_words:
            if not isinstance(word, dict):
                raise ValueError(
                    "Continuous ASR word must "
                    "be an object"
                )

            local_words.append(
                {
                    "text": str(word["text"]),
                    "start": float(word["start"]),
                    "end": float(word["end"]),
                }
            )

        _validate_word_timeline(
            local_words
        )

        global_words = [
            {
                "text": word["text"],
                "start": word["start"] + start,
                "end": word["end"] + start,
            }
            for word in local_words
        ]

        combined_words.extend(
            global_words
        )

        if chunk_text.strip():
            combined_text.append(
                chunk_text.strip()
            )

        chunk_records.append(
            ContinuousAsrChunk(
                index=index,
                start=start,
                end=end,
                boundary=boundary,
            )
        )

    _validate_word_timeline(
        combined_words
    )

    if len(languages) > 1:
        raise ValueError(
            "Continuous ASR chunks disagree "
            f"on language: {sorted(languages)!r}"
        )

    text = " ".join(
        combined_text
    ).strip()

    language = (
        next(iter(languages))
        if languages
        else None
    )

    return ContinuousAsrResult(
        language=language,
        text=text,
        words=combined_words,
        utterances=_derive_utterances(
            text,
            combined_words,
        ),
        chunks=tuple(chunk_records),
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
    max_inference_batch_size: int = 8,
) -> ContinuousAsrResult:
    source_representation, source_path = (
        resolve_source_representation(
            storage,
            source_id,
            representation_name,
        )
    )

    duration = source_representation.get(
        "duration"
    )

    if not isinstance(
        duration,
        (int, float),
    ):
        raise ValueError(
            "Continuous ASR source "
            "representation has no duration"
        )

    duration = float(duration)

    chunks = plan_continuous_asr_chunks(
        storage,
        source_id,
        duration=duration,
    )

    with tempfile.TemporaryDirectory(
        prefix="voice-dataset-qwen3-"
    ) as temporary_directory:
        root = Path(
            temporary_directory
        )

        chunk_directory = (
            root / "chunks"
        )
        chunk_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest_chunks = []

        for chunk in chunks:
            chunk_path = (
                chunk_directory
                / f"chunk_{chunk.index:04d}.wav"
            )

            print(
                "Qwen chunk "
                f"{chunk.index + 1}/{len(chunks)}: "
                f"{chunk.start:.3f}-"
                f"{chunk.end:.3f}s "
                f"({chunk.end - chunk.start:.3f}s, "
                f"{chunk.boundary})",
                flush=True,
            )

            extract_audio_region(
                source=source_path,
                destination=chunk_path,
                start=chunk.start,
                end=chunk.end,
            )

            manifest_chunks.append(
                {
                    "index": chunk.index,
                    "path": str(chunk_path),
                    "start": chunk.start,
                    "end": chunk.end,
                    "boundary": chunk.boundary,
                }
            )

        manifest_path = (
            root / "chunks.json"
        )

        manifest_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "chunks": manifest_chunks,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        output_path = (
            root / "continuous-asr.json"
        )

        arguments: list[str | Path] = [
            "--input-manifest",
            manifest_path,
            "--output",
            output_path,
            "--max-inference-batch-size",
            str(max_inference_batch_size),
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

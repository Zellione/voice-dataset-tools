from __future__ import annotations

import pytest

from voice_dataset.continuous_asr import (
    _derive_utterances,
    _load_qwen_alignment_output,
    _load_qwen_output,
    _validate_word_timeline,
    plan_continuous_asr_chunks,
)
from voice_dataset.schema import CandidateRegion
from voice_dataset.storage import DatasetStorage


def word(
    text: str,
    start: float,
    end: float,
) -> dict:
    return {
        "text": text,
        "start": start,
        "end": end,
    }


def test_derive_utterances_allows_different_word_boundaries(
) -> None:
    text = (
        "Well, well, not the fresh-faced Academy pledge, "
        "are you? You want peace."
    )

    words = [
        word("Well", 0.0, 0.2),
        word("well", 0.3, 0.5),
        word("not", 0.6, 0.8),
        word("the", 0.9, 1.0),
        word("freshfaced", 1.1, 1.5),
        word("Academy", 1.6, 1.9),
        word("pledge", 2.0, 2.3),
        word("are", 2.4, 2.5),
        word("you", 2.6, 2.8),
        word("You", 3.0, 3.2),
        word("want", 3.3, 3.5),
        word("peace", 3.6, 3.9),
    ]

    result = _derive_utterances(text, words)

    assert result == [
        {
            "text": (
                "Well, well, not the fresh-faced "
                "Academy pledge, are you?"
            ),
            "word_start": 0,
            "word_end": 9,
        },
        {
            "text": "You want peace.",
            "word_start": 9,
            "word_end": 12,
        },
    ]


def test_derive_utterances_rejects_lexical_mismatch(
) -> None:
    words = [
        word("You", 0.0, 0.2),
        word("want", 0.3, 0.5),
        word("war", 0.6, 0.8),
    ]

    with pytest.raises(
        ValueError,
        match=(
            "Continuous ASR text does not match "
            "aligned word sequence"
        ),
    ):
        _derive_utterances(
            "You want peace.",
            words,
        )


def test_load_qwen_alignment_output(
    tmp_path,
) -> None:
    path = tmp_path / "alignment.json"

    path.write_text(
        """
{
  "format": "voice-dataset-forced-alignment-output",
  "version": 1,
  "language": "English",
  "text": "Bravo, sis.",
  "words": [
    {
      "text": "Bravo",
      "start": 2.24,
      "end": 2.88
    },
    {
      "text": "sis",
      "start": 2.88,
      "end": 3.44
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    result = _load_qwen_alignment_output(path)

    assert result.language == "English"
    assert result.text == "Bravo, sis."
    assert result.words == [
        word("Bravo", 2.24, 2.88),
        word("sis", 2.88, 3.44),
    ]


def test_load_qwen_alignment_output_rejects_text_mismatch(
    tmp_path,
) -> None:
    path = tmp_path / "alignment.json"

    path.write_text(
        """
{
  "format": "voice-dataset-forced-alignment-output",
  "version": 1,
  "language": "English",
  "text": "Bravo, sis.",
  "words": [
    {
      "text": "Wrong",
      "start": 2.24,
      "end": 2.88
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "forced alignment text does not match "
            "aligned word sequence"
        ),
    ):
        _load_qwen_alignment_output(path)


def _add_region(
    storage: DatasetStorage,
    region_id: str,
    start: float,
    end: float,
    text: str | None = None,
) -> None:
    region = CandidateRegion(
        id=region_id,
        source_id="source_001",
        source_start=start,
        source_end=end,
        detector="test",
        detector_label="speaker_0",
    )

    storage.add_region(region)

    if text is not None:
        storage.update_region(
            region_id,
            lambda record: {
                **record,
                "transcripts": {
                    "whisper": {
                        "text": text,
                    }
                },
            },
        )


def test_plan_continuous_asr_chunks_prefers_sentence_gap(
    tmp_path,
) -> None:
    storage = DatasetStorage(tmp_path)

    _add_region(
        storage,
        "region_000001",
        0.0,
        49.8,
        "A sentence ends here.",
    )
    _add_region(
        storage,
        "region_000002",
        50.8,
        110.0,
        "More speech",
    )

    chunks = plan_continuous_asr_chunks(
        storage,
        "source_001",
        duration=120.0,
    )

    assert len(chunks) == 2
    assert chunks[0].boundary == "sentence_gap"
    assert chunks[0].end == pytest.approx(50.3)
    assert chunks[1].start == pytest.approx(50.3)
    assert chunks[1].end == pytest.approx(120.0)


def test_plan_continuous_asr_chunks_extends_to_safe_gap(
    tmp_path,
) -> None:
    storage = DatasetStorage(tmp_path)

    _add_region(
        storage,
        "region_000001",
        0.0,
        79.0,
        "Still speaking",
    )
    _add_region(
        storage,
        "region_000002",
        80.0,
        130.0,
        "Next section",
    )

    chunks = plan_continuous_asr_chunks(
        storage,
        "source_001",
        duration=140.0,
        search_seconds=10.0,
    )

    assert chunks[0].end == pytest.approx(79.5)
    assert chunks[0].boundary == "speech_gap"


def test_plan_continuous_asr_chunks_refuses_unsafe_cut(
    tmp_path,
) -> None:
    storage = DatasetStorage(tmp_path)

    _add_region(
        storage,
        "region_000001",
        0.0,
        300.0,
        "Continuous speech",
    )

    with pytest.raises(
        ValueError,
        match="refusing to split inside continuous speech",
    ):
        plan_continuous_asr_chunks(
            storage,
            "source_001",
            duration=300.0,
        )


def test_validate_word_timeline_rejects_timestamp_reversal(
) -> None:
    with pytest.raises(
        ValueError,
        match="not monotonic",
    ):
        _validate_word_timeline(
            [
                word("first", 10.0, 10.2),
                word("second", 9.0, 9.2),
            ]
        )


def test_validate_word_timeline_rejects_implausible_duration(
) -> None:
    with pytest.raises(
        ValueError,
        match="implausibly long",
    ):
        _validate_word_timeline(
            [
                word("broken", 10.0, 20.0),
            ]
        )


def test_validate_word_timeline_rejects_long_zero_duration_run(
) -> None:
    with pytest.raises(
        ValueError,
        match="zero-duration",
    ):
        _validate_word_timeline(
            [
                word("one", 10.0, 10.0),
                word("two", 10.0, 10.0),
                word("three", 10.0, 10.0),
            ]
        )


def test_load_chunked_qwen_output_offsets_words(
    tmp_path,
) -> None:
    path = tmp_path / "continuous.json"

    path.write_text(
        """
{
  "format": "voice-dataset-continuous-asr-output",
  "version": 2,
  "transcriber": {
    "name": "qwen3-asr",
    "model": "Qwen/Qwen3-ASR-0.6B"
  },
  "aligner": {
    "name": "qwen3-forced-aligner",
    "model": "Qwen/Qwen3-ForcedAligner-0.6B"
  },
  "chunks": [
    {
      "index": 0,
      "start": 0.0,
      "end": 60.0,
      "boundary": "sentence_gap",
      "language": "English",
      "text": "Hello.",
      "words": [
        {
          "text": "Hello",
          "start": 1.0,
          "end": 1.4
        }
      ]
    },
    {
      "index": 1,
      "start": 60.0,
      "end": 120.0,
      "boundary": "source_end",
      "language": "English",
      "text": "World.",
      "words": [
        {
          "text": "World",
          "start": 2.0,
          "end": 2.5
        }
      ]
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    result = _load_qwen_output(path)

    assert result.text == "Hello. World."
    assert result.words == [
        word("Hello", 1.0, 1.4),
        word("World", 62.0, 62.5),
    ]

    assert len(result.chunks) == 2
    assert result.chunks[0].boundary == "sentence_gap"

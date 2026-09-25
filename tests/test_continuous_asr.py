from __future__ import annotations

import pytest

from voice_dataset.continuous_asr import (
    _derive_utterances,
    _load_qwen_alignment_output,
)


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

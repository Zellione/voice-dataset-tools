from __future__ import annotations

from pathlib import Path

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


def test_validate_word_timeline_allows_long_word_duration(
) -> None:
    _validate_word_timeline(
        [
            word("uncertain", 10.0, 66.16),
            word("next", 66.16, 66.40),
        ]
    )


def test_validate_word_timeline_allows_zero_duration_words(
) -> None:
    _validate_word_timeline(
        [
            word("one", 10.0, 10.0),
            word("two", 10.0, 10.0),
            word("three", 10.0, 10.0),
            word("four", 10.0, 10.4),
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


def test_plan_continuous_asr_chunks_allows_source_end_after_extension(
    tmp_path,
) -> None:
    storage = DatasetStorage(tmp_path)

    _add_region(
        storage,
        "region_000001",
        0.0,
        80.0,
        "First section.",
    )
    _add_region(
        storage,
        "region_000002",
        81.0,
        200.0,
        "Continuous final section",
    )

    chunks = plan_continuous_asr_chunks(
        storage,
        "source_001",
        duration=200.0,
        target_seconds=60.0,
        search_seconds=10.0,
        max_chunk_seconds=240.0,
    )

    assert chunks[0].end == pytest.approx(80.5)
    assert chunks[0].boundary == "sentence_gap"

    assert chunks[1].start == pytest.approx(80.5)
    assert chunks[1].end == pytest.approx(200.0)
    assert chunks[1].boundary == "source_end"


def test_plan_continuous_asr_chunks_uses_union_of_overlapping_regions(
    tmp_path,
) -> None:
    storage = DatasetStorage(tmp_path)

    _add_region(
        storage,
        "region_000001",
        0.0,
        50.0,
        "First part.",
    )

    _add_region(
        storage,
        "region_000002",
        45.0,
        80.0,
        "Overlapping speech continues",
    )

    _add_region(
        storage,
        "region_000003",
        81.0,
        140.0,
        "Next block.",
    )

    chunks = plan_continuous_asr_chunks(
        storage,
        "source_001",
        duration=160.0,
        target_seconds=60.0,
        search_seconds=30.0,
    )

    # The apparent 50-81 s gap is not safe because
    # region_000002 extends speech through 80 s.
    assert chunks[0].end == pytest.approx(80.5)
    assert chunks[0].boundary == "speech_gap"


def test_load_qwen_alignment_batch_output(
    tmp_path,
) -> None:
    from voice_dataset.continuous_asr import (
        _load_qwen_alignment_batch_output,
    )

    path = tmp_path / "batch.json"

    path.write_text(
        """
{
  "format": "voice-dataset-forced-alignment-batch-output",
  "version": 1,
  "alignments": [
    {
      "index": 0,
      "language": "English",
      "text": "Hello.",
      "words": [
        {
          "text": "Hello",
          "start": 0.1,
          "end": 0.5
        }
      ]
    },
    {
      "index": 1,
      "language": "English",
      "text": "World.",
      "words": [
        {
          "text": "World",
          "start": 0.2,
          "end": 0.7
        }
      ]
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    results = _load_qwen_alignment_batch_output(
        path
    )

    assert len(results) == 2

    assert results[0].text == "Hello."
    assert results[0].words == [
        word("Hello", 0.1, 0.5),
    ]

    assert results[1].text == "World."
    assert results[1].words == [
        word("World", 0.2, 0.7),
    ]


def test_align_texts_qwen3_runs_one_worker_for_batch(
    tmp_path,
    monkeypatch,
) -> None:
    from voice_dataset.continuous_asr import (
        ForcedAlignmentRequest,
        align_texts_qwen3,
    )

    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"

    first.touch()
    second.touch()

    calls = []

    def fake_run_worker(
        worker_arg,
        arguments,
        **kwargs,
    ):
        calls.append(
            (worker_arg, list(arguments))
        )

        output_index = arguments.index(
            "--output"
        ) + 1

        output = Path(
            arguments[output_index]
        )

        output.write_text(
            """
{
  "format": "voice-dataset-forced-alignment-batch-output",
  "version": 1,
  "alignments": [
    {
      "index": 0,
      "language": "English",
      "text": "Hello.",
      "words": [
        {
          "text": "Hello",
          "start": 0.1,
          "end": 0.5
        }
      ]
    },
    {
      "index": 1,
      "language": "English",
      "text": "World.",
      "words": [
        {
          "text": "World",
          "start": 0.2,
          "end": 0.7
        }
      ]
    }
  ]
}
""".strip(),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "voice_dataset.continuous_asr.run_worker",
        fake_run_worker,
    )

    results = align_texts_qwen3(
        [
            ForcedAlignmentRequest(
                audio_path=first,
                text="Hello.",
                language="English",
            ),
            ForcedAlignmentRequest(
                audio_path=second,
                text="World.",
                language="English",
            ),
        ],
        max_inference_batch_size=16,
    )

    assert len(calls) == 1
    assert len(results) == 2

    arguments = calls[0][1]

    assert "--mode" in arguments
    assert "align" in arguments
    assert "--input-manifest" in arguments
    assert "--max-inference-batch-size" in arguments
    assert "16" in arguments


def test_continuous_asr_compatibility_accepts_legacy_evidence(
    tmp_path,
):
    from voice_dataset.continuous_asr import (
        QWEN3_ASR_MODEL,
        continuous_asr_evidence_is_compatible,
    )
    from voice_dataset.schema import SourceRecord

    storage = DatasetStorage(tmp_path)

    storage.add_source(
        SourceRecord(
            id="source_001",
            metadata={
                "continuous_asr": {
                    "qwen3": {
                        "model": QWEN3_ASR_MODEL,
                        "representation": "center",
                        "language": "English",
                        "text": "hello",
                        "words": [],
                        "utterances": [],
                    },
                },
            },
        )
    )

    assert continuous_asr_evidence_is_compatible(
        storage,
        "source_001",
        "qwen3",
        representation_name="center",
        language="English",
    )


def test_continuous_asr_compatibility_rejects_new_language_override(
    tmp_path,
):
    from voice_dataset.continuous_asr import (
        CONTINUOUS_ASR_EVIDENCE_VERSION,
        QWEN3_ASR_MODEL,
        continuous_asr_evidence_is_compatible,
    )
    from voice_dataset.schema import SourceRecord

    storage = DatasetStorage(tmp_path)

    storage.add_source(
        SourceRecord(
            id="source_001",
            metadata={
                "continuous_asr": {
                    "qwen3": {
                        "model": QWEN3_ASR_MODEL,
                        "representation": "center",
                        "provenance": {
                            "version": (
                                CONTINUOUS_ASR_EVIDENCE_VERSION
                            ),
                            "language_override": "English",
                            "chunking": {
                                "target_chunk_seconds": 60.0,
                                "boundary_search_seconds": 15.0,
                                "max_chunk_seconds": 240.0,
                                "minimum_gap_seconds": 0.5,
                            },
                        },
                    },
                },
            },
        )
    )

    with pytest.raises(
        ValueError,
        match="different language override",
    ):
        continuous_asr_evidence_is_compatible(
            storage,
            "source_001",
            "qwen3",
            representation_name="center",
            language="German",
        )


def test_continuous_asr_compatibility_rejects_model_change(
    tmp_path,
):
    from voice_dataset.continuous_asr import (
        continuous_asr_evidence_is_compatible,
    )
    from voice_dataset.schema import SourceRecord

    storage = DatasetStorage(tmp_path)

    storage.add_source(
        SourceRecord(
            id="source_001",
            metadata={
                "continuous_asr": {
                    "qwen3": {
                        "model": "old-model",
                        "representation": "center",
                    },
                },
            },
        )
    )

    with pytest.raises(
        ValueError,
        match="different model",
    ):
        continuous_asr_evidence_is_compatible(
            storage,
            "source_001",
            "qwen3",
            representation_name="center",
            language=None,
        )


def test_plan_continuous_asr_chunks_uses_trailing_source_gap(
    tmp_path,
) -> None:
    storage = DatasetStorage(tmp_path)

    _add_region(
        storage,
        "region_000001",
        0.0,
        60.0,
        "First section.",
    )

    _add_region(
        storage,
        "region_000002",
        61.0,
        221.0,
        "Final detected speech",
    )

    chunks = plan_continuous_asr_chunks(
        storage,
        "source_001",
        duration=500.0,
        target_seconds=60.0,
        search_seconds=15.0,
        max_chunk_seconds=240.0,
        minimum_gap_seconds=0.5,
    )

    assert chunks[0].end == pytest.approx(60.5)
    assert chunks[0].boundary == "sentence_gap"

    assert chunks[1].start == pytest.approx(60.5)
    assert chunks[1].end == pytest.approx(221.25)
    assert chunks[1].boundary == "source_gap"

    assert chunks[2].start == pytest.approx(221.25)
    assert chunks[2].end == pytest.approx(281.25)
    assert chunks[2].boundary == "source_gap"

    assert chunks[3].start == pytest.approx(281.25)
    assert chunks[3].end == pytest.approx(341.25)
    assert chunks[3].boundary == "source_gap"

    assert chunks[4].start == pytest.approx(341.25)
    assert chunks[4].end == pytest.approx(401.25)
    assert chunks[4].boundary == "source_gap"

    assert chunks[5].start == pytest.approx(401.25)
    assert chunks[5].end == pytest.approx(461.25)
    assert chunks[5].boundary == "source_gap"

    assert chunks[6].start == pytest.approx(461.25)
    assert chunks[6].end == pytest.approx(500.0)
    assert chunks[6].boundary == "source_end"


def test_plan_continuous_asr_chunks_handles_no_detected_speech(
    tmp_path,
) -> None:
    storage = DatasetStorage(tmp_path)

    chunks = plan_continuous_asr_chunks(
        storage,
        "source_001",
        duration=300.0,
        target_seconds=60.0,
        search_seconds=15.0,
        max_chunk_seconds=240.0,
        minimum_gap_seconds=0.5,
    )

    assert [
        (
            chunk.start,
            chunk.end,
            chunk.boundary,
        )
        for chunk in chunks
    ] == [
        (0.0, 60.0, "source_gap"),
        (60.0, 120.0, "source_gap"),
        (120.0, 180.0, "source_gap"),
        (180.0, 240.0, "source_gap"),
        (240.0, 300.0, "source_end"),
    ]


def test_plan_continuous_asr_chunks_handles_long_leading_source_gap(
    tmp_path,
) -> None:
    storage = DatasetStorage(tmp_path)

    _add_region(
        storage,
        "region_000001",
        200.0,
        260.0,
        "Detected speech.",
    )

    chunks = plan_continuous_asr_chunks(
        storage,
        "source_001",
        duration=300.0,
        target_seconds=60.0,
        search_seconds=15.0,
        max_chunk_seconds=240.0,
        minimum_gap_seconds=0.5,
    )

    assert chunks[0].start == pytest.approx(0.0)
    assert chunks[0].end == pytest.approx(60.0)
    assert chunks[0].boundary == "source_gap"

    assert chunks[1].start == pytest.approx(60.0)
    assert chunks[1].end == pytest.approx(120.0)
    assert chunks[1].boundary == "source_gap"

    assert chunks[2].start == pytest.approx(120.0)
    assert chunks[2].end == pytest.approx(180.0)
    assert chunks[2].boundary == "source_gap"

    assert chunks[-1].end == pytest.approx(300.0)
    assert chunks[-1].boundary == "source_end"


def test_plan_continuous_asr_chunks_handles_long_internal_gap(
    tmp_path,
) -> None:
    storage = DatasetStorage(tmp_path)

    _add_region(
        storage,
        "region_000001",
        0.0,
        30.0,
        "First speech.",
    )

    _add_region(
        storage,
        "region_000002",
        300.0,
        340.0,
        "Second speech.",
    )

    chunks = plan_continuous_asr_chunks(
        storage,
        "source_001",
        duration=380.0,
        target_seconds=60.0,
        search_seconds=15.0,
        max_chunk_seconds=240.0,
        minimum_gap_seconds=0.5,
    )

    assert chunks[0].start == pytest.approx(0.0)
    assert chunks[0].end == pytest.approx(60.0)
    assert chunks[0].boundary == "source_gap"

    assert chunks[1].start == pytest.approx(60.0)
    assert chunks[1].end == pytest.approx(120.0)
    assert chunks[1].boundary == "source_gap"

    assert chunks[2].start == pytest.approx(120.0)
    assert chunks[2].end == pytest.approx(180.0)
    assert chunks[2].boundary == "source_gap"

    assert chunks[3].start == pytest.approx(180.0)
    assert chunks[3].end == pytest.approx(240.0)
    assert chunks[3].boundary == "source_gap"

    assert chunks[-1].end == pytest.approx(380.0)
    assert chunks[-1].boundary == "source_end"


def test_continuous_asr_compatibility_rejects_old_provenance_version(
    tmp_path,
) -> None:
    from voice_dataset.continuous_asr import (
        QWEN3_ASR_MODEL,
        continuous_asr_evidence_is_compatible,
    )
    from voice_dataset.schema import SourceRecord

    storage = DatasetStorage(tmp_path)

    storage.add_source(
        SourceRecord(
            id="source_001",
            metadata={
                "continuous_asr": {
                    "qwen3": {
                        "model": QWEN3_ASR_MODEL,
                        "representation": "center",
                        "provenance": {
                            "version": 1,
                            "language_override": "English",
                            "chunking": {
                                "target_chunk_seconds": 60.0,
                                "boundary_search_seconds": 15.0,
                                "max_chunk_seconds": 240.0,
                                "minimum_gap_seconds": 0.5,
                            },
                        },
                    },
                },
            },
        )
    )

    with pytest.raises(
        ValueError,
        match="evidence version is stale",
    ):
        continuous_asr_evidence_is_compatible(
            storage,
            "source_001",
            "qwen3",
            representation_name="center",
            language="English",
        )


def test_filter_continuous_asr_chunks_skips_non_speech(
    tmp_path,
) -> None:
    from voice_dataset.continuous_asr import (
        ContinuousAsrChunk,
        _filter_continuous_asr_chunks_for_speech,
    )

    storage = DatasetStorage(tmp_path)

    _add_region(
        storage,
        "region_000001",
        10.0,
        20.0,
        "Speech.",
    )

    _add_region(
        storage,
        "region_000002",
        125.0,
        135.0,
        "More speech.",
    )

    chunks = [
        ContinuousAsrChunk(
            index=0,
            start=0.0,
            end=60.0,
            boundary="source_gap",
        ),
        ContinuousAsrChunk(
            index=1,
            start=60.0,
            end=120.0,
            boundary="source_gap",
        ),
        ContinuousAsrChunk(
            index=2,
            start=120.0,
            end=180.0,
            boundary="source_gap",
        ),
        ContinuousAsrChunk(
            index=3,
            start=180.0,
            end=240.0,
            boundary="source_end",
        ),
    ]

    result = _filter_continuous_asr_chunks_for_speech(
        storage,
        "source_001",
        chunks,
    )

    assert result == [
        ContinuousAsrChunk(
            index=0,
            start=0.0,
            end=60.0,
            boundary="source_gap",
        ),
        ContinuousAsrChunk(
            index=1,
            start=120.0,
            end=180.0,
            boundary="source_gap",
        ),
    ]


def test_filter_continuous_asr_chunks_ignores_boundary_touch(
    tmp_path,
) -> None:
    from voice_dataset.continuous_asr import (
        ContinuousAsrChunk,
        _filter_continuous_asr_chunks_for_speech,
    )

    storage = DatasetStorage(tmp_path)

    _add_region(
        storage,
        "region_000001",
        60.0,
        70.0,
        "Speech.",
    )

    chunks = [
        ContinuousAsrChunk(
            index=0,
            start=0.0,
            end=60.0,
            boundary="source_gap",
        ),
        ContinuousAsrChunk(
            index=1,
            start=60.0,
            end=120.0,
            boundary="source_end",
        ),
    ]

    result = _filter_continuous_asr_chunks_for_speech(
        storage,
        "source_001",
        chunks,
    )

    assert result == [
        ContinuousAsrChunk(
            index=0,
            start=60.0,
            end=120.0,
            boundary="source_end",
        ),
    ]


def test_transcribe_source_qwen3_skips_non_speech_chunks(
    tmp_path,
    monkeypatch,
) -> None:
    import json

    from voice_dataset.continuous_asr import (
        ContinuousAsrChunk,
        ContinuousAsrResult,
        transcribe_source_qwen3,
    )

    storage = DatasetStorage(tmp_path)

    _add_region(
        storage,
        "region_000001",
        10.0,
        20.0,
        "Speech.",
    )

    source_path = tmp_path / "source.wav"
    source_path.touch()

    monkeypatch.setattr(
        "voice_dataset.continuous_asr."
        "resolve_source_representation",
        lambda *args, **kwargs: (
            {
                "duration": 120.0,
            },
            source_path,
        ),
    )

    monkeypatch.setattr(
        "voice_dataset.continuous_asr."
        "plan_continuous_asr_chunks",
        lambda *args, **kwargs: [
            ContinuousAsrChunk(
                index=0,
                start=0.0,
                end=60.0,
                boundary="source_gap",
            ),
            ContinuousAsrChunk(
                index=1,
                start=60.0,
                end=120.0,
                boundary="source_end",
            ),
        ],
    )

    extracted = []

    def fake_extract_audio_region(
        *,
        source,
        destination,
        start,
        end,
    ):
        extracted.append(
            (start, end)
        )
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        destination.touch()

    monkeypatch.setattr(
        "voice_dataset.continuous_asr."
        "extract_audio_region",
        fake_extract_audio_region,
    )

    worker_manifests = []

    def fake_run_worker(
        worker_arg,
        arguments,
        **kwargs,
    ):
        manifest_index = (
            arguments.index(
                "--input-manifest"
            )
            + 1
        )

        manifest_path = Path(
            arguments[manifest_index]
        )

        worker_manifests.append(
            json.loads(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )
        )

    monkeypatch.setattr(
        "voice_dataset.continuous_asr."
        "run_worker",
        fake_run_worker,
    )

    monkeypatch.setattr(
        "voice_dataset.continuous_asr."
        "_load_qwen_output",
        lambda path: ContinuousAsrResult(
            language="English",
            text="Speech.",
            words=[
                {
                    "text": "Speech",
                    "start": 10.0,
                    "end": 10.5,
                }
            ],
            utterances=[
                {
                    "text": "Speech.",
                    "word_start": 0,
                    "word_end": 1,
                }
            ],
            chunks=(
                ContinuousAsrChunk(
                    index=0,
                    start=0.0,
                    end=60.0,
                    boundary="source_gap",
                ),
            ),
        ),
    )

    attached = []

    monkeypatch.setattr(
        "voice_dataset.continuous_asr."
        "attach_continuous_asr",
        lambda *args, **kwargs: (
            attached.append(
                (args, kwargs)
            )
        ),
    )

    result = transcribe_source_qwen3(
        storage,
        "source_001",
        representation_name="center",
        language="en",
    )

    assert result.text == "Speech."

    assert extracted == [
        (0.0, 60.0),
    ]

    assert len(worker_manifests) == 1

    assert worker_manifests[0]["chunks"] == [
        {
            "index": 0,
            "path": worker_manifests[0][
                "chunks"
            ][0]["path"],
            "start": 0.0,
            "end": 60.0,
            "boundary": "source_gap",
        }
    ]

    assert len(attached) == 1


def test_transcribe_source_qwen3_handles_no_speech_chunks(
    tmp_path,
    monkeypatch,
) -> None:
    from voice_dataset.continuous_asr import (
        ContinuousAsrChunk,
        transcribe_source_qwen3,
    )

    storage = DatasetStorage(tmp_path)

    source_path = tmp_path / "source.wav"
    source_path.touch()

    monkeypatch.setattr(
        "voice_dataset.continuous_asr."
        "resolve_source_representation",
        lambda *args, **kwargs: (
            {
                "duration": 120.0,
            },
            source_path,
        ),
    )

    monkeypatch.setattr(
        "voice_dataset.continuous_asr."
        "plan_continuous_asr_chunks",
        lambda *args, **kwargs: [
            ContinuousAsrChunk(
                index=0,
                start=0.0,
                end=60.0,
                boundary="source_gap",
            ),
            ContinuousAsrChunk(
                index=1,
                start=60.0,
                end=120.0,
                boundary="source_end",
            ),
        ],
    )

    worker_calls = []

    monkeypatch.setattr(
        "voice_dataset.continuous_asr."
        "run_worker",
        lambda *args, **kwargs: (
            worker_calls.append(
                (args, kwargs)
            )
        ),
    )

    extracted = []

    monkeypatch.setattr(
        "voice_dataset.continuous_asr."
        "extract_audio_region",
        lambda **kwargs: (
            extracted.append(kwargs)
        ),
    )

    attached = []

    def fake_attach(
        *args,
        **kwargs,
    ):
        attached.append(
            (args, kwargs)
        )

    monkeypatch.setattr(
        "voice_dataset.continuous_asr."
        "attach_continuous_asr",
        fake_attach,
    )

    result = transcribe_source_qwen3(
        storage,
        "source_001",
        representation_name="center",
        language="en",
    )

    assert worker_calls == []
    assert extracted == []

    assert result.language is None
    assert result.text == ""
    assert result.words == []
    assert result.utterances == []
    assert result.chunks == ()

    assert len(attached) == 1

    _, attach_kwargs = attached[0]

    assert attach_kwargs["language"] is None
    assert attach_kwargs["text"] == ""
    assert attach_kwargs["words"] == []
    assert attach_kwargs["utterances"] == []

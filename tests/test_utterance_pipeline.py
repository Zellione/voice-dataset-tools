import pytest

from pathlib import Path

from unittest.mock import patch

from voice_dataset.speaker_words import (
    SpeakerAttributedWord,
)
from voice_dataset.utterance_pipeline import (
    build_source_utterances,
    UtterancePipelineResult,
    apply_source_utterance_turns,
    source_utterance_reconciliation_is_complete,
    source_utterance_reconciliation_is_curated,
    mark_source_utterance_reconciliation_curated,
)

from voice_dataset.utterance_reconciliation import (
    SpeakerAttributedWord,
    UtteranceCandidate,
)

from voice_dataset.word_alignment import (
    AlignmentRegionEvidence,
    AlignmentTextMatch,
    EffectiveWordAlignment,
)
from voice_dataset.representations import MaterializeTurnsResult
from voice_dataset.storage import DatasetStorage


def test_pipeline_suppresses_word_without_renumbering():
    raw_words = [
        {
            "text": "kid",
            "start": 73.280,
            "end": 73.440,
        },
        {
            "text": "Out",
            "start": 73.440,
            "end": 73.680,
        },
        {
            "text": "but",
            "start": 74.720,
            "end": 74.880,
        },
    ]

    recovered_words = [
        {
            "text": "kid",
            "start": 73.253,
            "end": 73.63971875,
        },
        raw_words[1],
        raw_words[2],
    ]

    alignment = EffectiveWordAlignment(
        words=tuple(recovered_words),
        recoveries=(),
        suppressed_word_indices=(1,),
    )

    attributed_arguments = {}

    def fake_attribute_speakers_to_words(
        storage,
        source_id,
        *,
        evidence_name,
        words,
        word_indices,
    ):
        attributed_arguments["words"] = words
        attributed_arguments["word_indices"] = (
            word_indices
        )

        return [
            SpeakerAttributedWord(
                index=index,
                text=word["text"],
                start=word["start"],
                end=word["end"],
                overlaps=(),
                speaker=None,
                assignment_method=(
                    "no_overlapping_region"
                ),
            )
            for index, word in zip(
                word_indices,
                words,
                strict=True,
            )
        ]

    with (
        patch(
            "voice_dataset.utterance_pipeline."
            "_continuous_asr_evidence",
            return_value={
                "words": raw_words,
                "language": "English",
            },
        ),
        patch(
            "voice_dataset.utterance_pipeline."
            "build_effective_word_alignment",
            return_value=alignment,
        ) as build_alignment,
        patch(
            "voice_dataset.utterance_pipeline."
            "_run_sat",
            return_value={0},
        ),
        patch(
            "voice_dataset.utterance_pipeline."
            "recover_sat_boundary_alignments",
            return_value=alignment,
        ) as recover_boundaries,
        patch(
            "voice_dataset.utterance_pipeline."
            "collect_region_evidence",
            return_value=[],
        ),
        patch(
            "voice_dataset.utterance_pipeline."
            "attribute_speakers_to_words",
            side_effect=(
                fake_attribute_speakers_to_words
            ),
        ),
        patch(
            "voice_dataset.utterance_pipeline."
            "resolve_fragmented_speaker_words",
            side_effect=lambda *args, **kwargs: args[2],
        ),
        patch(
            "voice_dataset.utterance_pipeline."
            "resolve_source_representation_for_purpose",
            return_value=(
                "analysis_audio",
                {
                    "kind": "analysis",
                    "purposes": ["boundary_analysis"],
                },
                Path("/unused/analysis.wav"),
            ),
        ),
    ):
        result = build_source_utterances(
            storage=None,
            source_id="source",
        )

    assert (
        build_alignment.call_args.kwargs[
            "representation_name"
        ]
        == "analysis_audio"
    )

    assert (
        recover_boundaries.call_args.kwargs[
            "representation_name"
        ]
        == "analysis_audio"
    )

    assert [
        word["text"]
        for word in attributed_arguments["words"]
    ] == [
        "kid",
        "but",
    ]

    assert attributed_arguments[
        "word_indices"
    ] == [0, 2]

    assert [
        (word.index, word.text)
        for word in result.words
    ] == [
        (0, "kid"),
        (2, "but"),
    ]

    assert [
        candidate.text
        for candidate in result.candidates
    ] == [
        "kid",
        "but",
    ]

    assert result.candidates[1].word_indices == (
        2,
    )

    assert result.alignment.words[1]["text"] == "Out"
    assert (
        result.alignment.suppressed_word_indices
        == (1,)
    )


def test_pipeline_rejects_sat_boundary_spanned_by_exact_region_match():
    raw_words = [
        {
            "text": "They're",
            "start": 207.426,
            "end": 207.506,
        },
        {
            "text": "right",
            "start": 207.506,
            "end": 207.826,
        },
        {
            "text": "not",
            "start": 207.906,
            "end": 208.146,
        },
        {
            "text": "to",
            "start": 208.146,
            "end": 208.226,
        },
        {
            "text": "trust",
            "start": 208.226,
            "end": 208.546,
        },
        {
            "text": "us",
            "start": 208.546,
            "end": 208.786,
        },
    ]

    alignment = EffectiveWordAlignment(
        words=tuple(raw_words),
        recoveries=(),
    )

    attributed = [
        SpeakerAttributedWord(
            index=index,
            text=word["text"],
            start=word["start"],
            end=word["end"],
            overlaps=(),
            speaker="SPEAKER_04",
            assignment_method="detector_overlap",
        )
        for index, word in enumerate(raw_words)
    ]

    with (
        patch(
            "voice_dataset.utterance_pipeline."
            "_continuous_asr_evidence",
            return_value={
                "words": raw_words,
                "language": "English",
            },
        ),
        patch(
            "voice_dataset.utterance_pipeline."
            "build_effective_word_alignment",
            return_value=alignment,
        ),
        patch(
            "voice_dataset.utterance_pipeline."
            "_run_sat",
            return_value={1},
        ),
        patch(
            "voice_dataset.utterance_pipeline."
            "recover_sat_boundary_alignments",
            return_value=alignment,
        ),
        patch(
            "voice_dataset.utterance_pipeline."
            "attribute_speakers_to_words",
            return_value=attributed,
        ),
        patch(
            "voice_dataset.utterance_pipeline."
            "resolve_consistent_speaker_context",
            return_value=attributed,
        ),
        patch(
            "voice_dataset.utterance_pipeline."
            "resolve_fragmented_speaker_words",
            return_value=attributed,
        ),
        patch(
            "voice_dataset.utterance_pipeline."
            "collect_region_evidence",
            return_value=[
                AlignmentRegionEvidence(
                    region_id="region_000020",
                    start=207.390969,
                    end=208.774719,
                    speaker="SPEAKER_04",
                    whisper_text=(
                        "They're right not to trust us."
                    ),
                    whisper_tokens=(
                        "they're",
                        "right",
                        "not",
                        "to",
                        "trust",
                        "us",
                    ),
                    text_matches=(
                        AlignmentTextMatch(
                            start_word_index=0,
                            end_word_index=5,
                            word_indices=tuple(
                                range(6)
                            ),
                            tokens=(
                                "they're",
                                "right",
                                "not",
                                "to",
                                "trust",
                                "us",
                            ),
                        ),
                    ),
                ),
            ],
        ),
        patch(
            "voice_dataset.utterance_pipeline."
            "resolve_source_representation_for_purpose",
            return_value=(
                "analysis_audio",
                {
                    "kind": "analysis",
                    "purposes": ["boundary_analysis"],
                },
                Path("/unused/analysis.wav"),
            ),
        ),
    ):
        result = build_source_utterances(
            storage=None,
            source_id="source",
        )

    assert result.sat_boundary_after_word_indices == ()
    assert [
        candidate.text
        for candidate in result.candidates
    ] == [
        "They're right not to trust us",
    ]


def test_pipeline_does_not_split_on_region_boundary_without_sat():
    raw_words = [
        {
            "text": "I",
            "start": 1.760,
            "end": 1.840,
        },
        {
            "text": "wish",
            "start": 1.840,
            "end": 2.080,
        },
        {
            "text": "I",
            "start": 2.080,
            "end": 2.160,
        },
        {
            "text": "could",
            "start": 2.160,
            "end": 2.320,
        },
        {
            "text": "say",
            "start": 2.320,
            "end": 2.640,
        },
        {
            "text": "it",
            "start": 2.640,
            "end": 2.880,
        },
        {
            "text": "gets",
            "start": 2.880,
            "end": 2.880,
        },
        {
            "text": "easier",
            "start": 2.880,
            "end": 3.280,
        },
        {
            "text": "kiddo",
            "start": 3.280,
            "end": 3.680,
        },
        {
            "text": "but",
            "start": 3.680,
            "end": 4.880,
        },
        {
            "text": "I'll",
            "start": 4.880,
            "end": 4.960,
        },
        {
            "text": "be",
            "start": 4.960,
            "end": 5.040,
        },
        {
            "text": "lying",
            "start": 5.040,
            "end": 5.520,
        },
    ]

    alignment = EffectiveWordAlignment(
        words=tuple(raw_words),
        recoveries=(),
    )

    attributed = [
        SpeakerAttributedWord(
            index=index,
            text=word["text"],
            start=word["start"],
            end=word["end"],
            overlaps=(),
            speaker="SPEAKER_00",
            assignment_method="detector_overlap",
        )
        for index, word in enumerate(raw_words)
    ]

    region_evidence = [
        AlignmentRegionEvidence(
            region_id="region_000001",
            start=1.752,
            end=3.642,
            speaker="SPEAKER_00",
            whisper_text=(
                "I wish I could say it gets easier, kiddo."
            ),
            whisper_tokens=(
                "i",
                "wish",
                "i",
                "could",
                "say",
                "it",
                "gets",
                "easier",
                "kiddo",
            ),
            text_matches=(
                AlignmentTextMatch(
                    start_word_index=0,
                    end_word_index=8,
                    word_indices=tuple(range(9)),
                    tokens=(
                        "i",
                        "wish",
                        "i",
                        "could",
                        "say",
                        "it",
                        "gets",
                        "easier",
                        "kiddo",
                    ),
                ),
            ),
        ),
        AlignmentRegionEvidence(
            region_id="region_000002",
            start=4.756,
            end=5.532,
            speaker="SPEAKER_00",
            whisper_text="but I'll be lying.",
            whisper_tokens=(
                "but",
                "i'll",
                "be",
                "lying",
            ),
            text_matches=(
                AlignmentTextMatch(
                    start_word_index=9,
                    end_word_index=12,
                    word_indices=tuple(range(9, 13)),
                    tokens=(
                        "but",
                        "i'll",
                        "be",
                        "lying",
                    ),
                ),
            ),
        ),
    ]

    with (
        patch(
            "voice_dataset.utterance_pipeline."
            "_continuous_asr_evidence",
            return_value={
                "words": raw_words,
                "language": "English",
            },
        ),
        patch(
            "voice_dataset.utterance_pipeline."
            "build_effective_word_alignment",
            return_value=alignment,
        ),
        patch(
            "voice_dataset.utterance_pipeline."
            "_run_sat",
            return_value=set(),
        ),
        patch(
            "voice_dataset.utterance_pipeline."
            "recover_sat_boundary_alignments",
            return_value=alignment,
        ),
        patch(
            "voice_dataset.utterance_pipeline."
            "collect_region_evidence",
            return_value=region_evidence,
        ),
        patch(
            "voice_dataset.utterance_pipeline."
            "attribute_speakers_to_words",
            return_value=attributed,
        ),
        patch(
            "voice_dataset.utterance_pipeline."
            "resolve_consistent_speaker_context",
            return_value=attributed,
        ),
        patch(
            "voice_dataset.utterance_pipeline."
            "resolve_fragmented_speaker_words",
            return_value=attributed,
        ),
        patch(
            "voice_dataset.utterance_pipeline."
            "resolve_source_representation_for_purpose",
            return_value=(
                "analysis_audio",
                {
                    "kind": "analysis",
                    "purposes": ["boundary_analysis"],
                },
                Path("/unused/analysis.wav"),
            ),
        ),
    ):
        result = build_source_utterances(
            storage=None,
            source_id="source",
        )

    assert (
        result.sat_boundary_after_word_indices
        == ()
    )

    assert [
        candidate.text
        for candidate in result.candidates
    ] == [
        (
            "I wish I could say it gets easier kiddo "
            "but I'll be lying"
        ),
    ]


def test_prepare_source_review_audio_uses_review_purpose(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage = DatasetStorage(tmp_path / "dataset")
    source_path = tmp_path / "speech.wav"
    source_path.touch()

    calls = []

    def resolve(
        storage_arg,
        source_id,
        purpose,
    ):
        calls.append(
            ("resolve", source_id, purpose)
        )
        return (
            "speech",
            {
                "kind": "separated_speech",
                "path": "speech.wav",
                "purposes": [
                    "review",
                    "tts_candidate",
                ],
            },
            source_path,
        )

    def materialize(**kwargs):
        calls.append(
            (
                "materialize",
                kwargs["source_id"],
                kwargs["source"],
                kwargs["representation_name"],
                kwargs["kind"],
                kwargs["purposes"],
            )
        )
        return MaterializeTurnsResult(
            created=2,
            skipped=1,
        )

    monkeypatch.setattr(
        "voice_dataset.utterance_pipeline."
        "resolve_source_representation_for_purpose",
        resolve,
    )
    monkeypatch.setattr(
        "voice_dataset.utterance_pipeline."
        "materialize_turns",
        materialize,
    )

    from voice_dataset.utterance_pipeline import (
        prepare_source_review_audio,
    )

    result = prepare_source_review_audio(
        storage,
        "source_001",
    )

    assert result.created == 2
    assert result.skipped == 1
    assert calls == [
        (
            "resolve",
            "source_001",
            "review",
        ),
        (
            "materialize",
            "source_001",
            source_path,
            "review",
            "separated_speech",
            ["review"],
        ),
    ]


def test_apply_source_utterance_turns_rejects_stale_automatic_turn(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)

    storage.turns.append(
        {
            "schema_version": 1,
            "record_type": "turn",
            "id": "turn_000001",
            "source_id": "source",
            "source_start": 1.760,
            "source_end": 5.520,
            "source_regions": [
                "region_000001",
                "region_000002",
            ],
            "language": "en",
            "transcript": (
                "I wish I could say it gets easier "
                "kiddo but I'll be lying"
            ),
            "assignment": {
                "status": "unknown",
                "voice_id": None,
                "method": None,
                "confidence": None,
            },
            "review": {
                "status": "pending",
            },
            "representations": {},
            "embeddings": {},
            "metadata": {
                "creation": {
                    "method": "continuous_asr_utterance",
                },
                "word_range": {
                    "start": 0,
                    "end": 12,
                },
            },
        }
    )

    words = tuple(
        SpeakerAttributedWord(
            index=index,
            text=text,
            start=float(index),
            end=float(index) + 0.5,
            overlaps=(),
            speaker="SPEAKER_00",
            assignment_method="detector_overlap",
        )
        for index, text in enumerate(
            [
                "I",
                "wish",
                "I",
                "could",
                "say",
                "it",
                "gets",
                "easier",
                "kiddo",
                "but",
                "I'll",
                "be",
                "lying",
            ]
        )
    )

    alignment = EffectiveWordAlignment(
        words=tuple(
            {
                "text": word.text,
                "start": word.start,
                "end": word.end,
            }
            for word in words
        ),
        recoveries=(),
    )

    pipeline = UtterancePipelineResult(
        words=words,
        candidates=(
            UtteranceCandidate(
                start_word_index=0,
                end_word_index=8,
                word_indices=tuple(range(0, 9)),
                start=1.760,
                end=3.680,
                text=(
                    "I wish I could say it gets "
                    "easier kiddo"
                ),
                speaker="SPEAKER_00",
                known_speakers=("SPEAKER_00",),
                unresolved_word_indices=(),
                alignment_issue_word_indices=(),
                end_boundary="region",
            ),
            UtteranceCandidate(
                start_word_index=9,
                end_word_index=12,
                word_indices=tuple(range(9, 13)),
                start=3.680,
                end=5.520,
                text="but I'll be lying",
                speaker="SPEAKER_00",
                known_speakers=("SPEAKER_00",),
                unresolved_word_indices=(),
                alignment_issue_word_indices=(),
                end_boundary="sat",
            ),
        ),
        sat_boundary_after_word_indices=(8,),
        alignment=alignment,
    )

    with pytest.raises(
        ValueError,
        match="stale automatic turn",
    ):
        apply_source_utterance_turns(
            storage,
            source_id="source",
            result=pipeline,
            language="en",
        )

    turns = storage.turns.load()

    assert [turn["id"] for turn in turns] == [
        "turn_000001",
    ]


def test_apply_source_utterance_turns_skips_matching_automatic_turn(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)

    storage.sources.append(
        {
            "schema_version": 1,
            "record_type": "source",
            "id": "source",
            "metadata": {},
        }
    )

    storage.turns.append(
        {
            "schema_version": 1,
            "record_type": "turn",
            "id": "turn_000001",
            "source_id": "source",
            "source_start": 1.760,
            "source_end": 3.680,
            "source_regions": [],
            "language": "en",
            "transcript": (
                "I wish I could say it gets easier kiddo"
            ),
            "assignment": {
                "status": "unknown",
                "voice_id": None,
                "method": None,
                "confidence": None,
            },
            "review": {
                "status": "pending",
            },
            "representations": {},
            "embeddings": {},
            "metadata": {
                "creation": {
                    "method": "continuous_asr_utterance",
                },
                "word_range": {
                    "start": 0,
                    "end": 8,
                },
            },
        }
    )

    words = tuple(
        SpeakerAttributedWord(
            index=index,
            text=text,
            start=float(index),
            end=float(index) + 0.5,
            overlaps=(),
            speaker="SPEAKER_00",
            assignment_method="detector_overlap",
        )
        for index, text in enumerate(
            [
                "I",
                "wish",
                "I",
                "could",
                "say",
                "it",
                "gets",
                "easier",
                "kiddo",
            ]
        )
    )

    alignment = EffectiveWordAlignment(
        words=tuple(
            {
                "text": word.text,
                "start": word.start,
                "end": word.end,
            }
            for word in words
        ),
        recoveries=(),
    )

    pipeline = UtterancePipelineResult(
        words=words,
        candidates=(
            UtteranceCandidate(
                start_word_index=0,
                end_word_index=8,
                word_indices=tuple(range(9)),
                start=1.760,
                end=3.680,
                text=(
                    "I wish I could say it gets easier kiddo"
                ),
                speaker="SPEAKER_00",
                known_speakers=("SPEAKER_00",),
                unresolved_word_indices=(),
                alignment_issue_word_indices=(),
                end_boundary="region",
            ),
        ),
        sat_boundary_after_word_indices=(8,),
        alignment=alignment,
    )

    result = apply_source_utterance_turns(
        storage,
        source_id="source",
        result=pipeline,
        language="en",
    )

    assert result.created == 0
    assert result.skipped == 1
    assert result.review == 0

    turns = storage.turns.load()

    assert [turn["id"] for turn in turns] == [
        "turn_000001",
    ]

    source = storage.get_source("source")

    assert source is not None

    assert source["metadata"][
        "utterance_reconciliation"
    ] == {
        "mode": "automatic",
        "asr_evidence": "qwen3",
        "word_ranges": [
            [0, 8],
        ],
    }

    assert source_utterance_reconciliation_is_complete(
        storage,
        "source",
        asr_evidence_name="qwen3",
    )


def test_source_utterance_reconciliation_requires_matching_turns(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)

    storage.sources.append(
        {
            "schema_version": 1,
            "record_type": "source",
            "id": "source",
            "metadata": {
                "utterance_reconciliation": {
                    "asr_evidence": "qwen3",
                    "word_ranges": [
                        [0, 8],
                    ],
                },
            },
        }
    )

    assert not source_utterance_reconciliation_is_complete(
        storage,
        "source",
        asr_evidence_name="qwen3",
    )


def test_source_utterance_reconciliation_rejects_duplicate_turn_ranges(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)

    storage.sources.append(
        {
            "schema_version": 1,
            "record_type": "source",
            "id": "source",
            "metadata": {
                "utterance_reconciliation": {
                    "asr_evidence": "qwen3",
                    "word_ranges": [
                        [0, 8],
                    ],
                },
            },
        }
    )

    for turn_id in (
        "turn_000001",
        "turn_000002",
    ):
        storage.turns.append(
            {
                "schema_version": 1,
                "record_type": "turn",
                "id": turn_id,
                "source_id": "source",
                "metadata": {
                    "creation": {
                        "method": (
                            "continuous_asr_utterance"
                        ),
                    },
                    "word_range": {
                        "start": 0,
                        "end": 8,
                    },
                },
            }
        )

    assert not source_utterance_reconciliation_is_complete(
        storage,
        "source",
        asr_evidence_name="qwen3",
    )


def test_source_utterance_reconciliation_detects_curated_source(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)

    storage.sources.append(
        {
            "schema_version": 1,
            "record_type": "source",
            "id": "source",
            "metadata": {
                "utterance_reconciliation": {
                    "mode": "curated",
                    "asr_evidence": "qwen3",
                    "word_ranges": [
                        [0, 8],
                    ],
                },
            },
        }
    )

    assert source_utterance_reconciliation_is_curated(
        storage,
        "source",
        asr_evidence_name="qwen3",
    )


def test_legacy_utterance_reconciliation_is_not_curated(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)

    storage.sources.append(
        {
            "schema_version": 1,
            "record_type": "source",
            "id": "source",
            "metadata": {
                "utterance_reconciliation": {
                    "asr_evidence": "qwen3",
                    "word_ranges": [],
                },
            },
        }
    )

    assert not source_utterance_reconciliation_is_curated(
        storage,
        "source",
        asr_evidence_name="qwen3",
    )

    assert source_utterance_reconciliation_is_complete(
        storage,
        "source",
        asr_evidence_name="qwen3",
    )


def test_mark_source_utterance_reconciliation_curated_preserves_plan(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)

    storage.sources.append(
        {
            "schema_version": 1,
            "record_type": "source",
            "id": "source",
            "metadata": {
                "utterance_reconciliation": {
                    "mode": "automatic",
                    "asr_evidence": "qwen3",
                    "word_ranges": [
                        [0, 8],
                        [9, 12],
                    ],
                },
            },
        }
    )

    mark_source_utterance_reconciliation_curated(
        storage,
        "source",
        asr_evidence_name="qwen3",
    )

    source = storage.get_source("source")

    assert source is not None
    assert source["metadata"][
        "utterance_reconciliation"
    ] == {
        "mode": "curated",
        "asr_evidence": "qwen3",
        "word_ranges": [
            [0, 8],
            [9, 12],
        ],
    }

    assert source_utterance_reconciliation_is_curated(
        storage,
        "source",
        asr_evidence_name="qwen3",
    )

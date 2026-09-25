from pathlib import Path

from unittest.mock import patch

from voice_dataset.speaker_words import (
    SpeakerAttributedWord,
)
from voice_dataset.utterance_pipeline import (
    build_source_utterances,
)
from voice_dataset.word_alignment import (
    AlignmentRegionEvidence,
    AlignmentTextMatch,
    EffectiveWordAlignment,
)


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

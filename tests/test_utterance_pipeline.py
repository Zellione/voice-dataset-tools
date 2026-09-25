from unittest.mock import patch

from voice_dataset.speaker_words import (
    SpeakerAttributedWord,
)
from voice_dataset.utterance_pipeline import (
    build_source_utterances,
)
from voice_dataset.word_alignment import (
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
        ),
        patch(
            "voice_dataset.utterance_pipeline."
            "_run_sat",
            return_value={0},
        ),
        patch(
            "voice_dataset.utterance_pipeline."
            "recover_sat_boundary_alignments",
            return_value=alignment,
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
    ):
        result = build_source_utterances(
            storage=None,
            source_id="source",
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

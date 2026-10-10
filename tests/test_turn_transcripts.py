from voice_dataset.turn_transcripts import project_continuous_asr_to_range


def test_projection_keeps_zero_duration_word_at_rounded_boundary(
) -> None:
    evidence = {
        "words": [
            {
                "text": "Did",
                "start": 1459.62284375,
                "end": 1459.62284375,
            },
            {
                "text": "you",
                "start": 1459.62284375,
                "end": 1459.78284375,
            },
        ],
        "utterances": [
            {
                "word_start": 0,
                "word_end": 2,
            },
        ],
    }

    projection = project_continuous_asr_to_range(
        evidence,
        1459.622844,
        1460.0,
    )

    assert projection.word_indices == (
        0,
        1,
    )

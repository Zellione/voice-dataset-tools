from unittest.mock import patch

from voice_dataset.diarization_conflicts import (
    ContinuousWordRegionConflict,
)
from voice_dataset.speaker_words import (
    SpeakerAttributedWord,
    resolve_fragmented_speaker_words,
)


def _word(
    index: int,
    text: str,
    start: float,
    end: float,
    speaker: str | None,
    assignment_method: str,
) -> SpeakerAttributedWord:
    return SpeakerAttributedWord(
        index=index,
        text=text,
        start=start,
        end=end,
        overlaps=(),
        speaker=speaker,
        assignment_method=assignment_method,
    )


def test_fragmented_speaker_tail_inside_sat_utterance_is_resolved():
    words = [
        _word(
            113,
            "You",
            223.266,
            223.426,
            "SPEAKER_04",
            "single_overlapping_region",
        ),
        _word(
            114,
            "were",
            223.426,
            223.506,
            "SPEAKER_04",
            "single_overlapping_region",
        ),
        _word(
            115,
            "always",
            223.506,
            223.826,
            None,
            "conflicting_overlapping_speakers",
        ),
        _word(
            116,
            "right",
            223.826,
            223.986,
            "SPEAKER_02",
            "single_overlapping_region",
        ),
        _word(
            117,
            "My",
            225.426,
            225.586,
            "SPEAKER_04",
            "single_overlapping_region",
        ),
    ]

    conflict = ContinuousWordRegionConflict(
        word="always",
        word_start=223.506,
        word_end=223.826,
        region_ids=(
            "region_000029",
            "region_000030",
        ),
        region_speakers=(
            "SPEAKER_04",
            "SPEAKER_02",
        ),
    )

    with patch(
        "voice_dataset.speaker_words."
        "analyze_continuous_word_region_conflicts",
        return_value=[conflict],
    ):
        resolved = resolve_fragmented_speaker_words(
            storage=None,
            source_id="source",
            words=words,
            boundary_after_word_indices={116},
        )

    assert [
        word.speaker
        for word in resolved
    ] == [
        "SPEAKER_04",
        "SPEAKER_04",
        "SPEAKER_04",
        "SPEAKER_04",
        "SPEAKER_04",
    ]

    assert resolved[2].assignment_method == (
        "fragmentation_sat_context"
    )

    assert resolved[3].assignment_method == (
        "fragmentation_sat_context"
    )


def test_fragmented_speaker_tail_requires_return_to_leading_speaker():
    words = [
        _word(
            0,
            "You",
            0.0,
            0.2,
            "SPEAKER_A",
            "single_overlapping_region",
        ),
        _word(
            1,
            "always",
            0.2,
            0.5,
            None,
            "conflicting_overlapping_speakers",
        ),
        _word(
            2,
            "right",
            0.5,
            0.7,
            "SPEAKER_B",
            "single_overlapping_region",
        ),
        _word(
            3,
            "Next",
            1.0,
            1.2,
            "SPEAKER_B",
            "single_overlapping_region",
        ),
    ]

    conflict = ContinuousWordRegionConflict(
        word="always",
        word_start=0.2,
        word_end=0.5,
        region_ids=("region_a", "region_b"),
        region_speakers=(
            "SPEAKER_A",
            "SPEAKER_B",
        ),
    )

    with patch(
        "voice_dataset.speaker_words."
        "analyze_continuous_word_region_conflicts",
        return_value=[conflict],
    ):
        resolved = resolve_fragmented_speaker_words(
            storage=None,
            source_id="source",
            words=words,
            boundary_after_word_indices={2},
        )

    assert resolved[1].speaker is None
    assert resolved[2].speaker == "SPEAKER_B"


def test_fragmented_speaker_resolution_does_not_cross_sat_boundary():
    words = [
        _word(
            0,
            "Alpha",
            0.0,
            0.2,
            "SPEAKER_A",
            "single_overlapping_region",
        ),
        _word(
            1,
            "boundary",
            0.2,
            0.5,
            None,
            "conflicting_overlapping_speakers",
        ),
        _word(
            2,
            "Bravo",
            0.5,
            0.7,
            "SPEAKER_B",
            "single_overlapping_region",
        ),
        _word(
            3,
            "Return",
            1.0,
            1.2,
            "SPEAKER_A",
            "single_overlapping_region",
        ),
    ]

    conflict = ContinuousWordRegionConflict(
        word="boundary",
        word_start=0.2,
        word_end=0.5,
        region_ids=("region_a", "region_b"),
        region_speakers=(
            "SPEAKER_A",
            "SPEAKER_B",
        ),
    )

    with patch(
        "voice_dataset.speaker_words."
        "analyze_continuous_word_region_conflicts",
        return_value=[conflict],
    ):
        resolved = resolve_fragmented_speaker_words(
            storage=None,
            source_id="source",
            words=words,
            boundary_after_word_indices={1, 2},
        )

    assert resolved[1].speaker is None
    assert resolved[2].speaker == "SPEAKER_B"

from voice_dataset.word_alignment import (
    AlignmentRegionEvidence,
    AlignmentTextMatch,
    LocalAlignmentRecovery,
    AlignmentComparison,
    compare_alignment_recovery,
    collect_boundary_text_evidence,
    detect_alignment_issues,
    find_text_matches,
    apply_alignment_recoveries,
    alignment_spans_overlap
)
import pytest


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


def test_detect_alignment_issue_for_long_word(
) -> None:
    words = [
        word("Really", 246.466, 247.186),
        word("thought", 247.186, 282.786),
        word("I", 282.786, 282.866),
    ]

    issues = detect_alignment_issues(words)

    assert len(issues) == 1
    assert issues[0].word_indices == (1,)
    assert issues[0].boundary_after_word_index is None
    assert issues[0].reasons == (
        "excessive_word_duration",
    )


def test_detect_alignment_issue_for_large_gap() -> None:
    words = [
        word("god", 92.400, 92.400),
        word("Bravo", 92.400, 92.400),
        word("sis", 92.400, 92.400),
        word("Councillors", 185.426, 185.986),
    ]

    issues = detect_alignment_issues(words)

    assert len(issues) == 1

    issue = issues[0]

    assert issue.word_indices == ()
    assert issue.boundary_after_word_index == 2
    assert issue.reasons == (
        "excessive_inter_word_gap",
    )


def test_normal_alignment_has_no_issues(
) -> None:
    words = [
        word("Really", 281.920, 282.400),
        word("thought", 282.400, 282.720),
        word("I", 282.720, 282.880),
        word("buried", 282.880, 283.360),
        word("this", 283.440, 283.600),
        word("place", 283.600, 284.080),
    ]

    assert detect_alignment_issues(words) == []


def test_find_bravo_text_span() -> None:
    words = [
        word("That's", 91.520, 91.760),
        word("my", 91.760, 91.920),
        word("girl", 91.920, 92.320),
        word("So", 92.320, 92.400),
        word("god", 92.400, 92.400),
        word("Bravo", 92.400, 92.400),
        word("sis", 92.400, 92.400),
        word(
            "Councillors",
            185.426,
            185.986,
        ),
    ]

    matches = find_text_matches(
        words,
        "Bravo, sis.",
    )

    assert len(matches) == 1
    assert matches[0].word_indices == (5, 6)
    assert matches[0].tokens == (
        "bravo",
        "sis",
    )


def test_find_buried_text_span() -> None:
    words = [
        word("Really", 246.466, 247.186),
        word("thought", 247.186, 282.786),
        word("I", 282.786, 282.866),
        word("buried", 282.866, 283.346),
        word("this", 283.426, 283.586),
        word("place", 283.586, 284.066),
    ]

    matches = find_text_matches(
        words,
        "Really thought I buried this place.",
    )

    assert len(matches) == 1
    assert matches[0].word_indices == (
        0,
        1,
        2,
        3,
        4,
        5,
    )


def test_text_match_reports_ambiguity() -> None:
    words = [
        word("You", 1.0, 1.1),
        word("were", 1.1, 1.2),
        word("right", 1.2, 1.3),
        word("and", 1.3, 1.4),
        word("you", 1.4, 1.5),
        word("were", 1.5, 1.6),
        word("right", 1.6, 1.7),
    ]

    matches = find_text_matches(
        words,
        "You were right.",
    )

    assert [
        match.word_indices
        for match in matches
    ] == [
        (0, 1, 2),
        (4, 5, 6),
    ]


def test_gap_issue_does_not_create_word_recovery_candidate() -> None:
    from voice_dataset.word_alignment import (
        AlignmentRegionEvidence,
        AlignmentTextMatch,
        build_recovery_candidates,
    )

    words = [
        word("Bravo", 92.400, 92.400),
        word("sis", 92.400, 92.400),
        word("Councillors", 185.426, 185.986),
    ]

    issues = detect_alignment_issues(words)

    evidence = [
        AlignmentRegionEvidence(
            region_id="region_000011",
            start=178.231,
            end=179.868,
            speaker="SPEAKER_02",
            whisper_text="Bravo, sis.",
            whisper_tokens=("bravo", "sis"),
            text_matches=(
                AlignmentTextMatch(
                    start_word_index=0,
                    end_word_index=1,
                    word_indices=(0, 1),
                    tokens=("bravo", "sis"),
                ),
            ),
        )
    ]

    assert build_recovery_candidates(
        issues,
        evidence,
    ) == []


def region_evidence(
    region_id: str,
    start_word_index: int,
    end_word_index: int,
) -> AlignmentRegionEvidence:
    return AlignmentRegionEvidence(
        region_id=region_id,
        start=0.0,
        end=1.0,
        speaker=None,
        whisper_text="test",
        whisper_tokens=("test",),
        text_matches=(
            AlignmentTextMatch(
                start_word_index=(
                    start_word_index
                ),
                end_word_index=(
                    end_word_index
                ),
                word_indices=tuple(
                    range(
                        start_word_index,
                        end_word_index + 1,
                    )
                ),
                tokens=("test",),
            ),
        ),
    )


def test_collect_boundary_text_evidence() -> None:
    evidence = [
        region_evidence(
            "left",
            30,
            34,
        ),
        region_evidence(
            "right",
            35,
            37,
        ),
        region_evidence(
            "unrelated",
            40,
            41,
        ),
    ]

    result = collect_boundary_text_evidence(
        34,
        evidence,
    )

    assert [
        item.region_id
        for item in result.left
    ] == ["left"]

    assert [
        item.region_id
        for item in result.right
    ] == ["right"]


def test_boundary_text_evidence_allows_missing_side() -> None:
    evidence = [
        region_evidence(
            "left",
            40,
            41,
        ),
        region_evidence(
            "later",
            43,
            54,
        ),
    ]

    result = collect_boundary_text_evidence(
        41,
        evidence,
    )

    assert [
        item.region_id
        for item in result.left
    ] == ["left"]

    assert result.right == ()


def test_compare_alignment_recovery_consistent() -> None:
    words = [
        {
            "text": "So",
            "start": 82.800,
            "end": 82.960,
        },
        {
            "text": "say",
            "start": 83.200,
            "end": 83.680,
        },
    ]

    recovery = LocalAlignmentRecovery(
        word_indices=(0, 1),
        region_id="region",
        region_start=82.820,
        region_end=83.630,
        text="So say",
        words=(
            {
                "text": "So",
                "start": 82.820,
                "end": 82.980,
            },
            {
                "text": "say",
                "start": 83.220,
                "end": 83.620,
            },
        ),
    )

    comparison = compare_alignment_recovery(
        words,
        recovery,
    )

    assert comparison.start_delta == pytest.approx(
        0.020
    )
    assert comparison.end_delta == pytest.approx(
        -0.060
    )


def test_compare_alignment_recovery_displaced() -> None:
    words = [
        {
            "text": "Bravo",
            "start": 92.400,
            "end": 92.400,
        },
        {
            "text": "sis",
            "start": 92.400,
            "end": 92.400,
        },
    ]

    recovery = LocalAlignmentRecovery(
        word_indices=(0, 1),
        region_id="region",
        region_start=178.231,
        region_end=179.868,
        text="Bravo sis",
        words=(
            {
                "text": "Bravo",
                "start": 178.231,
                "end": 178.871,
            },
            {
                "text": "sis",
                "start": 178.871,
                "end": 179.431,
            },
        ),
    )

    comparison = compare_alignment_recovery(
        words,
        recovery,
    )

    assert comparison.start_delta == pytest.approx(
        85.831
    )
    assert comparison.end_delta == pytest.approx(
        87.031
    )
    assert comparison.original_duration == 0.0
    assert comparison.recovered_duration == pytest.approx(
        1.200
    )


def test_compare_alignment_recovery_distorted() -> None:
    words = [
        {
            "text": "Really",
            "start": 246.466,
            "end": 247.186,
        },
        {
            "text": "thought",
            "start": 247.186,
            "end": 282.786,
        },
        {
            "text": "place",
            "start": 283.586,
            "end": 284.066,
        },
    ]

    recovery = LocalAlignmentRecovery(
        word_indices=(0, 1, 2),
        region_id="region",
        region_start=281.810,
        region_end=284.273,
        text="Really thought place",
        words=(
            {
                "text": "Really",
                "start": 281.890,
                "end": 282.450,
            },
            {
                "text": "thought",
                "start": 282.450,
                "end": 282.770,
            },
            {
                "text": "place",
                "start": 283.570,
                "end": 284.050,
            },
        ),
    )

    comparison = compare_alignment_recovery(
        words,
        recovery,
    )

    assert comparison.start_delta == pytest.approx(
        35.424
    )
    assert comparison.end_delta == pytest.approx(
        -0.016
    )
    assert comparison.original_duration == pytest.approx(
        37.600
    )
    assert comparison.recovered_duration == pytest.approx(
        2.160
    )


def test_apply_alignment_recoveries_copies_words() -> None:
    words = [
        {
            "text": "Bravo",
            "start": 92.400,
            "end": 92.400,
            "metadata": {"source": "qwen"},
        },
        {
            "text": "sis",
            "start": 92.400,
            "end": 92.400,
        },
        {
            "text": "Councillors",
            "start": 185.426,
            "end": 185.986,
        },
    ]

    recovery = LocalAlignmentRecovery(
        word_indices=(0, 1),
        region_id="region_000011",
        region_start=178.231,
        region_end=179.868,
        text="Bravo sis",
        words=(
            {
                "text": "Bravo",
                "start": 178.231,
                "end": 178.871,
            },
            {
                "text": "sis",
                "start": 178.871,
                "end": 179.431,
            },
        ),
    )

    effective = apply_alignment_recoveries(
        words,
        [recovery],
    )

    assert effective is not words
    assert effective[0] is not words[0]

    assert words[0]["start"] == 92.400
    assert words[1]["end"] == 92.400

    assert effective[0]["start"] == pytest.approx(
        178.231
    )
    assert effective[1]["end"] == pytest.approx(
        179.431
    )

    assert effective[0]["text"] == "Bravo"
    assert effective[0]["metadata"] == {
        "source": "qwen"
    }

    assert effective[2] == words[2]


def test_apply_alignment_recoveries_rejects_overlap() -> None:
    words = [
        {
            "text": "Bravo",
            "start": 92.400,
            "end": 92.400,
        },
    ]

    recovery = LocalAlignmentRecovery(
        word_indices=(0,),
        region_id="region",
        region_start=178.0,
        region_end=179.0,
        text="Bravo",
        words=(
            {
                "text": "Bravo",
                "start": 178.2,
                "end": 178.8,
            },
        ),
    )

    with pytest.raises(
        ValueError,
        match="Overlapping alignment recoveries",
    ):
        apply_alignment_recoveries(
            words,
            [recovery, recovery],
        )


def test_alignment_spans_overlap_for_consistent_alignment() -> None:
    comparison = AlignmentComparison(
        word_indices=(30, 31, 32, 33, 34),
        original_start=82.800,
        original_end=83.680,
        recovered_start=82.820,
        recovered_end=83.620,
        start_delta=0.020,
        end_delta=-0.060,
        original_duration=0.880,
        recovered_duration=0.800,
    )

    assert alignment_spans_overlap(comparison)


def test_alignment_spans_do_not_overlap_when_displaced() -> None:
    comparison = AlignmentComparison(
        word_indices=(40, 41),
        original_start=92.400,
        original_end=92.400,
        recovered_start=178.231,
        recovered_end=179.431,
        start_delta=85.831,
        end_delta=87.031,
        original_duration=0.0,
        recovered_duration=1.200,
    )

    assert not alignment_spans_overlap(comparison)


def test_apply_alignment_recoveries_rejects_words_outside_region(
) -> None:
    words = [
        word("Bravo", 92.4, 92.4),
        word("sis", 92.4, 92.4),
    ]

    recovery = LocalAlignmentRecovery(
        word_indices=(0, 1),
        region_id="region_000011",
        region_start=178.231,
        region_end=179.868,
        text="Bravo sis",
        words=(
            word("Bravo", 178.231, 178.871),
            word("sis", 178.871, 180.100),
        ),
    )

    with pytest.raises(
        ValueError,
        match="outside recovery region",
    ):
        apply_alignment_recoveries(
            words,
            [recovery],
        )

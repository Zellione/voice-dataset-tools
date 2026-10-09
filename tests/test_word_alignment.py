from pathlib import Path

import voice_dataset.word_alignment as word_alignment
from voice_dataset.continuous_asr import ForcedAlignmentResult
from voice_dataset.storage import DatasetStorage
from voice_dataset.word_alignment import (
    AlignmentRegionEvidence,
    AlignmentTextMatch,
    AlignmentRecoveryCandidate,
    LocalAlignmentRecovery,
    AlignmentComparison,
    StrandedAlignmentCandidate,
    EffectiveWordAlignment,
    PostRecoveryWordConflict,
    find_unclaimed_post_recovery_words,
    compare_alignment_recovery,
    collect_boundary_text_evidence,
    detect_alignment_issues,
    find_text_matches,
    apply_alignment_recoveries,
    alignment_spans_overlap,
    find_stranded_alignment_candidates,
    recover_stranded_alignment,
    build_sat_boundary_recovery_candidates,
    recover_sat_boundary_alignments,
    recover_candidate_alignment,
    alignment_recovery_is_valid,
    find_post_recovery_word_conflicts,
    validate_sat_boundaries,
    infer_region_boundaries,
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


def test_find_stranded_alignment_candidates_rejects_ambiguous_regions(
) -> None:
    words = [
        word("That's", 91.520, 91.760),
        word("my", 91.760, 91.920),
        word("girl", 91.920, 92.320),
        word("So", 92.320, 92.400),
        word("god", 92.400, 92.400),
        word("Bravo", 92.400, 92.400),
        word("sis", 92.400, 92.400),
    ]

    evidence = [
        AlignmentRegionEvidence(
            region_id="region_previous",
            start=91.510,
            end=92.354,
            speaker="SPEAKER_00",
            whisper_text="That's my girl.",
            whisper_tokens=("that's", "my", "girl"),
            text_matches=(
                AlignmentTextMatch(
                    start_word_index=0,
                    end_word_index=2,
                    word_indices=(0, 1, 2),
                    tokens=("that's", "my", "girl"),
                ),
            ),
        ),
        AlignmentRegionEvidence(
            region_id="region_unknown_a",
            start=120.0,
            end=121.0,
            speaker="SPEAKER_02",
            whisper_text="unmatched a",
            whisper_tokens=("unmatched", "a"),
            text_matches=(),
        ),
        AlignmentRegionEvidence(
            region_id="region_unknown_b",
            start=163.617,
            end=164.410,
            speaker="SPEAKER_02",
            whisper_text="unmatched b",
            whisper_tokens=("unmatched", "b"),
            text_matches=(),
        ),
        AlignmentRegionEvidence(
            region_id="region_recovered",
            start=178.231,
            end=179.868,
            speaker="SPEAKER_02",
            whisper_text="Bravo, sis.",
            whisper_tokens=("bravo", "sis"),
            text_matches=(
                AlignmentTextMatch(
                    start_word_index=5,
                    end_word_index=6,
                    word_indices=(5, 6),
                    tokens=("bravo", "sis"),
                ),
            ),
        ),
    ]

    recovery = LocalAlignmentRecovery(
        word_indices=(5, 6),
        region_id="region_recovered",
        region_start=178.231,
        region_end=179.868,
        text="Bravo sis",
        words=(
            word("Bravo", 178.231, 178.871),
            word("sis", 178.871, 179.431),
        ),
    )

    assert find_stranded_alignment_candidates(
        words,
        evidence,
        [recovery],
    ) == []


def test_find_stranded_alignment_candidates_rejects_displaced_previous_anchor():
    words = [
        word("That's", 50.000, 50.240),
        word("my", 50.240, 50.400),
        word("girl", 50.400, 50.800),
        word("So", 92.320, 92.400),
        word("god", 92.400, 92.400),
        word("Bravo", 92.400, 92.400),
        word("sis", 92.400, 92.400),
    ]

    region_evidence = [
        AlignmentRegionEvidence(
            region_id="region_000009",
            start=91.510,
            end=92.354,
            speaker="SPEAKER_00",
            whisper_text="That's my girl.",
            whisper_tokens=("that's", "my", "girl"),
            text_matches=(
                AlignmentTextMatch(
                    start_word_index=0,
                    end_word_index=2,
                    word_indices=(0, 1, 2),
                    tokens=("that's", "my", "girl"),
                ),
            ),
        ),
        AlignmentRegionEvidence(
            region_id="region_000010",
            start=163.617,
            end=164.410,
            speaker="SPEAKER_02",
            whisper_text="So go.",
            whisper_tokens=("so", "go"),
            text_matches=(),
        ),
        AlignmentRegionEvidence(
            region_id="region_000011",
            start=178.231,
            end=179.868,
            speaker="SPEAKER_02",
            whisper_text="Bravo, sis.",
            whisper_tokens=("bravo", "sis"),
            text_matches=(
                AlignmentTextMatch(
                    start_word_index=5,
                    end_word_index=6,
                    word_indices=(5, 6),
                    tokens=("bravo", "sis"),
                ),
            ),
        ),
    ]

    recoveries = [
        LocalAlignmentRecovery(
            word_indices=(5, 6),
            region_id="region_000011",
            region_start=178.231,
            region_end=179.868,
            text="Bravo sis",
            words=(
                word(
                    "Bravo",
                    178.231,
                    178.871,
                ),
                word(
                    "sis",
                    178.871,
                    179.431,
                ),
            ),
        ),
    ]

    candidates = (
        find_stranded_alignment_candidates(
            words,
            region_evidence,
            recoveries,
        )
    )

    assert candidates == []


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


def test_sat_boundary_at_region_end_does_not_create_candidate() -> None:
    words = [
        word("One", 10.0, 10.2),
        word("two", 10.2, 10.4),
        word("three", 10.4, 10.6),
        word("Four", 10.7, 10.9),
    ]

    evidence = AlignmentRegionEvidence(
        region_id="region_000001",
        start=9.9,
        end=10.6,
        whisper_text=None,
        whisper_tokens=(),
        speaker="SPEAKER_00",
        text_matches=(),
    )

    candidates = build_sat_boundary_recovery_candidates(
        words,
        [evidence],
        {2},
    )

    assert candidates == []


def test_find_stranded_alignment_candidate_between_confirmed_anchors(
) -> None:
    words = [
        word("That's", 91.520, 91.760),
        word("my", 91.760, 91.920),
        word("girl", 91.920, 92.320),
        word("So", 92.320, 92.400),
        word("god", 92.400, 92.400),
        word("Bravo", 92.400, 92.400),
        word("sis", 92.400, 92.400),
        word("Councillors", 185.426, 185.986),
    ]

    evidence = [
        AlignmentRegionEvidence(
            region_id="region_000009",
            start=91.510,
            end=92.354,
            speaker="SPEAKER_00",
            whisper_text="That's my girl.",
            whisper_tokens=("that's", "my", "girl"),
            text_matches=(
                AlignmentTextMatch(
                    start_word_index=0,
                    end_word_index=2,
                    word_indices=(0, 1, 2),
                    tokens=("that's", "my", "girl"),
                ),
            ),
        ),
        AlignmentRegionEvidence(
            region_id="region_000010",
            start=163.617,
            end=164.410,
            speaker="SPEAKER_02",
            whisper_text="So go.",
            whisper_tokens=("so", "go"),
            text_matches=(),
        ),
        AlignmentRegionEvidence(
            region_id="region_000011",
            start=178.231,
            end=179.868,
            speaker="SPEAKER_02",
            whisper_text="Bravo, sis.",
            whisper_tokens=("bravo", "sis"),
            text_matches=(
                AlignmentTextMatch(
                    start_word_index=5,
                    end_word_index=6,
                    word_indices=(5, 6),
                    tokens=("bravo", "sis"),
                ),
            ),
        ),
    ]

    bravo_recovery = LocalAlignmentRecovery(
        word_indices=(5, 6),
        region_id="region_000011",
        region_start=178.231,
        region_end=179.868,
        text="Bravo sis",
        words=(
            word("Bravo", 178.231, 178.871),
            word("sis", 178.871, 179.431),
        ),
    )

    candidates = find_stranded_alignment_candidates(
        words,
        evidence,
        [bravo_recovery],
    )

    assert candidates == [
        StrandedAlignmentCandidate(
            word_indices=(3, 4),
            region_id="region_000010",
            region_start=163.617,
            region_end=164.410,
            speaker="SPEAKER_02",
        )
    ]


def test_effective_alignment_recovers_stranded_words_between_anchors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = DatasetStorage(tmp_path / "dataset")

    words = [
        word("That's", 91.520, 91.760),
        word("my", 91.760, 91.920),
        word("girl", 91.920, 92.320),
        word("So", 92.320, 92.400),
        word("god", 92.400, 92.400),
        word("Bravo", 92.400, 92.400),
        word("sis", 92.400, 92.400),
        word("Councillors", 185.426, 185.986),
    ]

    def add_region(
        region_id: str,
        start: float,
        end: float,
        text: str,
    ) -> None:
        storage.regions.append(
            {
                "schema_version": 2,
                "record_type": "candidate_region",
                "id": region_id,
                "source_id": "source_001",
                "source_start": start,
                "source_end": end,
                "representations": {},
                "transcripts": {
                    "whisper": {
                        "text": text,
                        "language": "English",
                    },
                },
                "embeddings": {},
                "reconciliation": {
                    "status": "pending",
                    "reason": None,
                    "notes": None,
                },
                "metadata": {},
            }
        )

    add_region(
        "region_000009",
        91.510,
        92.354,
        "That's my girl.",
    )
    add_region(
        "region_000010",
        163.617,
        164.410,
        "So go.",
    )
    add_region(
        "region_000011",
        178.231,
        179.868,
        "Bravo, sis.",
    )
    add_region(
        "region_000012",
        185.403,
        186.027,
        "Counselors.",
    )

    def fake_recover_region_alignment(
        storage: DatasetStorage,
        source_id: str,
        evidence: AlignmentRegionEvidence,
        words: list[dict],
        *,
        representation_name: str,
        language: str,
    ) -> LocalAlignmentRecovery:
        assert evidence.region_id == "region_000011"

        return LocalAlignmentRecovery(
            word_indices=(5, 6),
            region_id=evidence.region_id,
            region_start=evidence.start,
            region_end=evidence.end,
            text="Bravo sis",
            words=(
                word("Bravo", 178.231, 178.871),
                word("sis", 178.871, 179.431),
            ),
        )

    def fake_recover_stranded_alignment(
        storage: DatasetStorage,
        source_id: str,
        candidate: StrandedAlignmentCandidate,
        words: list[dict],
        *,
        representation_name: str,
        language: str,
    ) -> LocalAlignmentRecovery:
        assert source_id == "source_001"
        assert candidate.word_indices == (3, 4)
        assert candidate.region_id == "region_000010"
        assert candidate.region_start == pytest.approx(
            163.617
        )
        assert candidate.region_end == pytest.approx(
            164.410
        )
        assert representation_name == "center"
        assert language == "English"

        return LocalAlignmentRecovery(
            word_indices=candidate.word_indices,
            region_id=candidate.region_id,
            region_start=candidate.region_start,
            region_end=candidate.region_end,
            text="So god",
            words=(
                word("So", 163.617, 163.937),
                word("god", 163.937, 164.177),
            ),
        )

    def fake_recover_alignment_candidates_batch(
        storage: DatasetStorage,
        source_id: str,
        candidates,
        words: list[dict],
        *,
        representation_name: str,
        language: str,
        max_inference_batch_size: int = 8,
    ):
        assert source_id == "source_001"
        assert representation_name == "center"
        assert language == "English"

        recoveries = []

        for candidate in candidates:
            if candidate.region_id == "region_000011":
                recoveries.append(
                    LocalAlignmentRecovery(
                        word_indices=(5, 6),
                        region_id=candidate.region_id,
                        region_start=candidate.region_start,
                        region_end=candidate.region_end,
                        text="Bravo sis",
                        words=(
                            word("Bravo", 178.231, 178.871),
                            word("sis", 178.871, 179.431),
                        ),
                    )
                )
            elif candidate.region_id == "region_000010":
                assert candidate.word_indices == (3, 4)

                recoveries.append(
                    LocalAlignmentRecovery(
                        word_indices=candidate.word_indices,
                        region_id=candidate.region_id,
                        region_start=candidate.region_start,
                        region_end=candidate.region_end,
                        text="So god",
                        words=(
                            word("So", 163.617, 163.937),
                            word("god", 163.937, 164.177),
                        ),
                    )
                )
            else:
                raise AssertionError(
                    f"unexpected recovery region: "
                    f"{candidate.region_id}"
                )

        return recoveries

    monkeypatch.setattr(
        word_alignment,
        "recover_alignment_candidates_batch",
        fake_recover_alignment_candidates_batch,
    )

    result = word_alignment.build_effective_word_alignment(
        storage,
        "source_001",
        words,
        representation_name="center",
        language="English",
    )

    assert result.words[3]["text"] == "So"
    assert result.words[4]["text"] == "god"

    assert result.words[3]["start"] == pytest.approx(
        163.617
    )
    assert result.words[4]["end"] <= 164.410


def test_find_stranded_alignment_candidates_rejects_noncontiguous_words():
    words = [
        word("left", 10.000, 10.400),
        word("stranded", 10.400, 10.500),
        word("matched", 10.500, 10.600),
        word("words", 10.600, 10.700),
        word("right", 30.000, 30.400),
    ]

    evidence = [
        AlignmentRegionEvidence(
            region_id="region_left",
            start=9.900,
            end=10.500,
            speaker="SPEAKER_00",
            whisper_text="left",
            whisper_tokens=("left",),
            text_matches=(
                AlignmentTextMatch(
                    start_word_index=0,
                    end_word_index=0,
                    word_indices=(0,),
                    tokens=("left",),
                ),
            ),
        ),
        AlignmentRegionEvidence(
            region_id="region_candidate",
            start=20.000,
            end=21.000,
            speaker="SPEAKER_01",
            whisper_text="something else",
            whisper_tokens=("something", "else"),
            text_matches=(),
        ),
        AlignmentRegionEvidence(
            region_id="region_matched",
            start=22.000,
            end=23.000,
            speaker="SPEAKER_01",
            whisper_text="matched",
            whisper_tokens=("matched",),
            text_matches=(
                AlignmentTextMatch(
                    start_word_index=2,
                    end_word_index=2,
                    word_indices=(2,),
                    tokens=("matched",),
                ),
            ),
        ),
        AlignmentRegionEvidence(
            region_id="region_right",
            start=29.900,
            end=30.500,
            speaker="SPEAKER_02",
            whisper_text="right",
            whisper_tokens=("right",),
            text_matches=(
                AlignmentTextMatch(
                    start_word_index=4,
                    end_word_index=4,
                    word_indices=(4,),
                    tokens=("right",),
                ),
            ),
        ),
    ]

    right_recovery = LocalAlignmentRecovery(
        word_indices=(4,),
        region_id="region_right",
        region_start=29.900,
        region_end=30.500,
        text="right",
        words=(
            word("right", 30.000, 30.400),
        ),
    )

    candidates = find_stranded_alignment_candidates(
        words,
        evidence,
        [right_recovery],
    )

    assert candidates == []


def test_find_stranded_alignment_candidates_requires_recovered_anchor(
) -> None:
    words = [
        word("That's", 91.520, 91.760),
        word("my", 91.760, 91.920),
        word("girl", 91.920, 92.320),
        word("So", 92.320, 92.400),
        word("god", 92.400, 92.400),
        word("Bravo", 92.400, 92.400),
        word("sis", 92.400, 92.400),
    ]

    evidence = [
        AlignmentRegionEvidence(
            region_id="region_previous",
            start=91.510,
            end=92.354,
            speaker="SPEAKER_00",
            whisper_text="That's my girl.",
            whisper_tokens=("that's", "my", "girl"),
            text_matches=(
                AlignmentTextMatch(
                    start_word_index=0,
                    end_word_index=2,
                    word_indices=(0, 1, 2),
                    tokens=("that's", "my", "girl"),
                ),
            ),
        ),
        AlignmentRegionEvidence(
            region_id="region_unknown",
            start=163.617,
            end=164.410,
            speaker="SPEAKER_02",
            whisper_text="So go.",
            whisper_tokens=("so", "go"),
            text_matches=(),
        ),
        AlignmentRegionEvidence(
            region_id="region_later",
            start=178.231,
            end=179.868,
            speaker="SPEAKER_02",
            whisper_text="Bravo, sis.",
            whisper_tokens=("bravo", "sis"),
            text_matches=(
                AlignmentTextMatch(
                    start_word_index=5,
                    end_word_index=6,
                    word_indices=(5, 6),
                    tokens=("bravo", "sis"),
                ),
            ),
        ),
    ]

    assert find_stranded_alignment_candidates(
        words,
        evidence,
        [],
    ) == []


def test_recover_stranded_alignment_uses_local_forced_alignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = DatasetStorage(tmp_path / "dataset")

    source_audio = (
        storage.root
        / "audio"
        / "source_001"
        / "center.wav"
    )
    source_audio.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    source_audio.touch()

    storage.sources.append(
        {
            "schema_version": 2,
            "record_type": "source",
            "id": "source_001",
            "representations": {
                "center": {
                    "path": (
                        "audio/source_001/center.wav"
                    ),
                    "kind": "center",
                    "processor": "test",
                    "sample_rate": 48000,
                    "channels": 1,
                    "duration": 300.0,
                },
            },
            "metadata": {},
        }
    )

    words = [
        word("So", 92.320, 92.400),
        word("god", 92.400, 92.400),
    ]

    candidate = StrandedAlignmentCandidate(
        word_indices=(0, 1),
        region_id="region_000010",
        region_start=163.617,
        region_end=164.410,
        speaker="SPEAKER_02",
    )

    extracted: dict[str, object] = {}

    def fake_extract_audio_region(
        *,
        source: Path,
        destination: Path,
        start: float,
        end: float,
    ) -> None:
        extracted["input_path"] = source
        extracted["start"] = start
        extracted["end"] = end
        destination.touch()

    def fake_align_text_qwen3(
        audio_path: Path,
        *,
        text: str,
        language: str,
    ) -> ForcedAlignmentResult:
        assert audio_path.exists()
        assert text == "So god"
        assert language == "English"

        return ForcedAlignmentResult(
            language="English",
            text="So god",
            words=[
                word("So", 0.000, 0.320),
                word("god", 0.320, 0.560),
            ],
        )

    monkeypatch.setattr(
        word_alignment,
        "extract_audio_region",
        fake_extract_audio_region,
    )
    monkeypatch.setattr(
        word_alignment,
        "align_text_qwen3",
        fake_align_text_qwen3,
    )

    recovery = recover_stranded_alignment(
        storage,
        "source_001",
        candidate,
        words,
        representation_name="center",
        language="English",
    )

    assert extracted["input_path"] == source_audio
    assert extracted["start"] == pytest.approx(
        163.617
    )
    assert extracted["end"] == pytest.approx(
        164.410
    )

    assert recovery.word_indices == (0, 1)
    assert recovery.region_id == "region_000010"
    assert recovery.text == "So god"

    assert recovery.words[0]["text"] == "So"
    assert recovery.words[0]["start"] == pytest.approx(
        163.617
    )
    assert recovery.words[0]["end"] == pytest.approx(
        163.937
    )

    assert recovery.words[1]["text"] == "god"
    assert recovery.words[1]["start"] == pytest.approx(
        163.937
    )
    assert recovery.words[1]["end"] == pytest.approx(
        164.177
    )


def test_effective_alignment_skips_invalid_stranded_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = DatasetStorage(tmp_path / "dataset")

    words = [
        word("left", 10.000, 10.400),
        word("stranded", 10.400, 10.500),
        word("right", 10.500, 10.500),
    ]

    region_evidence = [
        AlignmentRegionEvidence(
            region_id="region_left",
            start=9.900,
            end=10.500,
            speaker="SPEAKER_00",
            whisper_text="left",
            whisper_tokens=("left",),
            text_matches=(
                AlignmentTextMatch(
                    start_word_index=0,
                    end_word_index=0,
                    word_indices=(0,),
                    tokens=("left",),
                ),
            ),
        ),
        AlignmentRegionEvidence(
            region_id="region_stranded",
            start=20.000,
            end=21.000,
            speaker="SPEAKER_01",
            whisper_text="something else",
            whisper_tokens=("something", "else"),
            text_matches=(),
        ),
        AlignmentRegionEvidence(
            region_id="region_right",
            start=30.000,
            end=31.000,
            speaker="SPEAKER_02",
            whisper_text="right",
            whisper_tokens=("right",),
            text_matches=(
                AlignmentTextMatch(
                    start_word_index=2,
                    end_word_index=2,
                    word_indices=(2,),
                    tokens=("right",),
                ),
            ),
        ),
    ]

    invalid_stranded_recovery = LocalAlignmentRecovery(
        word_indices=(1,),
        region_id="region_stranded",
        region_start=20.000,
        region_end=21.000,
        text="stranded",
        words=(word("stranded", 20.100, 22.000),),
    )

    monkeypatch.setattr(
        word_alignment,
        "collect_region_evidence",
        lambda *args, **kwargs: region_evidence,
    )
    monkeypatch.setattr(
        word_alignment,
        "detect_alignment_issues",
        lambda words: [],
    )
    monkeypatch.setattr(
        word_alignment,
        "build_recovery_candidates",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        word_alignment,
        "find_stranded_alignment_candidates",
        lambda *args, **kwargs: [
            StrandedAlignmentCandidate(
                word_indices=(1,),
                region_id="region_stranded",
                region_start=20.000,
                region_end=21.000,
                speaker="SPEAKER_01",
            )
        ],
    )
    def fake_batch_recovery(
        storage,
        source_id,
        candidates,
        words,
        **kwargs,
    ):
        if not candidates:
            return []

        assert len(candidates) == 1
        assert (
            candidates[0].region_id
            == "region_stranded"
        )

        return [
            invalid_stranded_recovery
        ]

    monkeypatch.setattr(
        word_alignment,
        "recover_alignment_candidates_batch",
        fake_batch_recovery,
    )

    result = word_alignment.build_effective_word_alignment(
        storage,
        "source_001",
        words,
        representation_name="center",
        language="English",
    )

    assert result.words[1]["start"] == pytest.approx(10.400)
    assert result.words[1]["end"] == pytest.approx(10.500)
    assert result.recoveries == ()


def test_effective_alignment_propagates_stranded_recovery_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = DatasetStorage(tmp_path / "dataset")

    words = [
        word("left", 10.000, 10.400),
        word("stranded", 10.400, 10.500),
    ]

    monkeypatch.setattr(
        word_alignment,
        "collect_region_evidence",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        word_alignment,
        "detect_alignment_issues",
        lambda words: [],
    )
    monkeypatch.setattr(
        word_alignment,
        "build_recovery_candidates",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        word_alignment,
        "find_stranded_alignment_candidates",
        lambda *args, **kwargs: [
            StrandedAlignmentCandidate(
                word_indices=(1,),
                region_id="region_stranded",
                region_start=20.000,
                region_end=21.000,
                speaker="SPEAKER_01",
            )
        ],
    )

    def fail_recovery(
        storage,
        source_id,
        candidates,
        words,
        **kwargs,
    ):
        if not candidates:
            return []

        assert len(candidates) == 1
        assert (
            candidates[0].region_id
            == "region_stranded"
        )

        raise RuntimeError(
            "forced aligner failed"
        )

    monkeypatch.setattr(
        word_alignment,
        "recover_alignment_candidates_batch",
        fail_recovery,
    )

    with pytest.raises(
        RuntimeError,
        match="forced aligner failed",
    ):
        word_alignment.build_effective_word_alignment(
            storage,
            "source_001",
            words,
            representation_name="center",
            language="English",
        )


def test_build_sat_boundary_recovery_candidate_for_boundary_inside_region():
    words = [
        word("I", 71.760, 71.840),
        word("wish", 71.840, 72.080),
        word("I", 72.080, 72.160),
        word("could", 72.160, 72.320),
        word("say", 72.320, 72.640),
        word("it's", 72.640, 72.880),
        word("easier", 72.880, 73.280),
        word("kid", 73.280, 73.440),
        word("Out", 73.440, 73.680),
        word("but", 74.720, 74.880),
    ]

    evidence = [
        AlignmentRegionEvidence(
            region_id="region_000003",
            start=71.73284375,
            end=73.63971875,
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
            text_matches=(),
        ),
    ]

    candidates = build_sat_boundary_recovery_candidates(
        words,
        evidence,
        {7},
    )

    assert len(candidates) == 1

    candidate = candidates[0]

    assert candidate.region_id == "region_000003"
    assert candidate.word_indices == tuple(range(8))
    assert candidate.start_word_index == 0
    assert candidate.end_word_index == 7

    assert candidate.region_start == pytest.approx(
        71.73284375
    )
    assert candidate.region_end == pytest.approx(
        73.63971875
    )

    assert 8 not in candidate.word_indices


def test_sat_boundary_candidate_skips_exact_match_across_boundary() -> None:
    words = [
        word("They're", 207.426, 207.506),
        word("right", 207.506, 207.826),
        word("not", 207.906, 208.146),
        word("to", 208.146, 208.226),
        word("trust", 208.226, 208.546),
        word("us", 208.546, 208.786),
    ]

    evidence = [
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
                    word_indices=tuple(range(6)),
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
    ]

    candidates = build_sat_boundary_recovery_candidates(
        words,
        evidence,
        {1},
    )

    assert candidates == []


@pytest.mark.parametrize(
    (
        "words",
        "region_start",
        "region_end",
        "whisper_text",
        "boundary_index",
    ),
    [
        (
            [
                word("They're", 207.426, 207.506),
                word("right", 207.506, 207.826),
                word("not", 207.906, 208.146),
                word("to", 208.146, 208.226),
                word("trust", 208.226, 208.546),
                word("us", 208.546, 208.786),
            ],
            207.390969,
            208.774719,
            "They're right not to trust us.",
            1,
        ),
        (
            [
                word("You're", 209.506, 209.746),
                word("walking", 209.746, 210.066),
                word("a", 210.066, 210.146),
                word("fine", 210.146, 210.626),
                word("line", 210.626, 211.186),
                word("Jace", 211.186, 211.506),
            ],
            209.533719,
            211.710969,
            "You're walking a fine line, Jace.",
            4,
        ),
        (
            [
                word("With", 213.186, 213.426),
                word("respect", 213.426, 213.826),
                word("I", 213.826, 213.906),
                word("don't", 213.906, 214.146),
                word("give", 214.146, 214.306),
                word("a", 214.306, 214.386),
                word("shit", 214.386, 214.626),
                word("what", 214.626, 214.786),
                word("any", 214.786, 214.946),
                word("of", 214.946, 215.026),
                word("you", 215.026, 215.186),
                word("think", 215.186, 215.426),
                word("of", 215.426, 215.506),
                word("me", 215.506, 215.666),
                word("anymore", 215.666, 216.146),
            ],
            213.127969,
            216.217344,
            (
                "With respect, I don't give a shit "
                "what any of you think of me anymore."
            ),
            1,
        ),
    ],
)


def test_validate_sat_boundaries_rejects_exact_region_match(
    words,
    region_start,
    region_end,
    whisper_text,
    boundary_index,
):
    matches = find_text_matches(
        words,
        whisper_text,
    )

    evidence = [
        AlignmentRegionEvidence(
            region_id="region",
            start=region_start,
            end=region_end,
            speaker="SPEAKER",
            whisper_text=whisper_text,
            whisper_tokens=(),
            text_matches=tuple(matches),
        ),
    ]

    validated = validate_sat_boundaries(
        words,
        evidence,
        {boundary_index},
    )

    assert validated == set()


def test_infer_region_boundary_from_adjacent_exact_matches():
    words = [
        word("I", 1.760, 1.840),
        word("wish", 1.840, 2.080),
        word("kiddo", 3.280, 3.680),
        word("but", 4.880, 4.960),
        word("lying", 5.040, 5.520),
    ]

    evidence = [
        AlignmentRegionEvidence(
            region_id="region_000001",
            start=1.752,
            end=3.642,
            speaker="SPEAKER_00",
            whisper_text="I wish kiddo.",
            whisper_tokens=("i", "wish", "kiddo"),
            text_matches=tuple(
                find_text_matches(
                    words,
                    "I wish kiddo.",
                )
            ),
        ),
        AlignmentRegionEvidence(
            region_id="region_000002",
            start=4.756,
            end=5.532,
            speaker="SPEAKER_00",
            whisper_text="but lying.",
            whisper_tokens=("but", "lying"),
            text_matches=tuple(
                find_text_matches(
                    words,
                    "but lying.",
                )
            ),
        ),
    ]

    assert infer_region_boundaries(
        evidence
    ) == {2}


def test_infer_region_boundary_rejects_overlapping_regions():
    match_left = AlignmentTextMatch(
        start_word_index=0,
        end_word_index=0,
        word_indices=(0,),
        tokens=("hello",),
    )
    match_right = AlignmentTextMatch(
        start_word_index=1,
        end_word_index=1,
        word_indices=(1,),
        tokens=("world",),
    )

    evidence = [
        AlignmentRegionEvidence(
            region_id="region_000001",
            start=10.000,
            end=10.600,
            speaker="SPEAKER_00",
            whisper_text="hello",
            whisper_tokens=("hello",),
            text_matches=(match_left,),
        ),
        AlignmentRegionEvidence(
            region_id="region_000002",
            start=10.500,
            end=11.000,
            speaker="SPEAKER_00",
            whisper_text="world",
            whisper_tokens=("world",),
            text_matches=(match_right,),
        ),
    ]

    assert infer_region_boundaries(
        evidence
    ) == set()


def test_infer_region_boundary_rejects_ambiguous_match():
    left_match = AlignmentTextMatch(
        start_word_index=0,
        end_word_index=0,
        word_indices=(0,),
        tokens=("hello",),
    )
    right_match = AlignmentTextMatch(
        start_word_index=1,
        end_word_index=1,
        word_indices=(1,),
        tokens=("world",),
    )

    evidence = [
        AlignmentRegionEvidence(
            region_id="region_000001",
            start=10.000,
            end=10.400,
            speaker="SPEAKER_00",
            whisper_text="hello",
            whisper_tokens=("hello",),
            text_matches=(
                left_match,
                left_match,
            ),
        ),
        AlignmentRegionEvidence(
            region_id="region_000002",
            start=10.500,
            end=11.000,
            speaker="SPEAKER_00",
            whisper_text="world",
            whisper_tokens=("world",),
            text_matches=(right_match,),
        ),
    ]

    assert infer_region_boundaries(
        evidence
    ) == set()


def test_infer_region_boundary_requires_adjacent_word_matches():
    left_match = AlignmentTextMatch(
        start_word_index=0,
        end_word_index=0,
        word_indices=(0,),
        tokens=("hello",),
    )
    right_match = AlignmentTextMatch(
        start_word_index=2,
        end_word_index=2,
        word_indices=(2,),
        tokens=("world",),
    )

    evidence = [
        AlignmentRegionEvidence(
            region_id="region_000001",
            start=10.000,
            end=10.400,
            speaker="SPEAKER_00",
            whisper_text="hello",
            whisper_tokens=("hello",),
            text_matches=(left_match,),
        ),
        AlignmentRegionEvidence(
            region_id="region_000002",
            start=10.800,
            end=11.200,
            speaker="SPEAKER_00",
            whisper_text="world",
            whisper_tokens=("world",),
            text_matches=(right_match,),
        ),
    ]

    assert infer_region_boundaries(
        evidence
    ) == set()


def test_validate_sat_boundaries_keeps_boundary_without_exact_match():
    words = [
        word("hello", 10.000, 10.400),
        word("world", 10.400, 10.800),
    ]

    evidence = [
        AlignmentRegionEvidence(
            region_id="region_001",
            start=9.900,
            end=11.000,
            speaker="SPEAKER_00",
            whisper_text="different text",
            whisper_tokens=(
                "different",
                "text",
            ),
            text_matches=(),
        ),
    ]

    validated = validate_sat_boundaries(
        words,
        evidence,
        {0},
    )

    assert validated == {0}


def test_validate_sat_boundaries_keeps_boundary_at_region_end():
    words = [
        word("hello", 10.000, 10.500),
        word("world", 10.500, 11.000),
    ]

    evidence = [
        AlignmentRegionEvidence(
            region_id="region_001",
            start=9.900,
            end=10.500,
            speaker="SPEAKER_00",
            whisper_text="hello world",
            whisper_tokens=(
                "hello",
                "world",
            ),
            text_matches=(
                AlignmentTextMatch(
                    start_word_index=0,
                    end_word_index=1,
                    word_indices=(0, 1),
                    tokens=("hello", "world"),
                ),
            ),
        ),
    ]

    validated = validate_sat_boundaries(
        words,
        evidence,
        {0},
    )

    assert validated == {0}


def test_validate_sat_boundaries_keeps_boundary_with_ambiguous_regions():
    words = [
        word("hello", 10.000, 10.400),
        word("world", 10.400, 10.800),
    ]

    match = AlignmentTextMatch(
        start_word_index=0,
        end_word_index=1,
        word_indices=(0, 1),
        tokens=("hello", "world"),
    )

    evidence = [
        AlignmentRegionEvidence(
            region_id="region_001",
            start=9.900,
            end=11.000,
            speaker="SPEAKER_00",
            whisper_text="hello world",
            whisper_tokens=("hello", "world"),
            text_matches=(match,),
        ),
        AlignmentRegionEvidence(
            region_id="region_002",
            start=9.950,
            end=10.900,
            speaker="SPEAKER_01",
            whisper_text="hello world",
            whisper_tokens=("hello", "world"),
            text_matches=(match,),
        ),
    ]

    validated = validate_sat_boundaries(
        words,
        evidence,
        {0},
    )

    assert validated == {0}


def test_sat_boundary_recovery_skips_boundary_at_region_end():
    words = [
        word("hello", 10.000, 10.500),
        word("next", 11.000, 11.400),
    ]

    evidence = [
        AlignmentRegionEvidence(
            region_id="region_001",
            start=9.900,
            end=10.500,
            speaker="SPEAKER_00",
            whisper_text="hello",
            whisper_tokens=("hello",),
            text_matches=(),
        ),
    ]

    candidates = build_sat_boundary_recovery_candidates(
        words,
        evidence,
        {0},
    )

    assert candidates == []


def test_sat_boundary_recovery_skips_boundary_without_unique_region():
    words = [
        word("hello", 10.000, 10.500),
        word("next", 10.500, 10.800),
    ]

    evidence = [
        AlignmentRegionEvidence(
            region_id="region_001",
            start=9.900,
            end=10.700,
            speaker="SPEAKER_00",
            whisper_text="hello",
            whisper_tokens=("hello",),
            text_matches=(),
        ),
        AlignmentRegionEvidence(
            region_id="region_002",
            start=10.400,
            end=10.900,
            speaker="SPEAKER_01",
            whisper_text="something",
            whisper_tokens=("something",),
            text_matches=(),
        ),
    ]

    candidates = build_sat_boundary_recovery_candidates(
        words,
        evidence,
        {0},
    )

    assert candidates == []


def test_sat_boundary_candidate_stops_at_previous_sat_boundary() -> None:
    words = [
        word("One", 10.0, 10.2),
        word("two", 10.2, 10.4),
        word("three", 10.4, 10.6),
        word("Four", 10.7, 10.9),
        word("five", 10.9, 11.1),
        word("six", 11.1, 11.3),
        word("seven", 11.3, 11.5),
        word("eight", 11.5, 11.7),
        word("nine", 11.7, 11.9),
    ]

    evidence = AlignmentRegionEvidence(
        region_id="region_000001",
        start=9.9,
        end=12.0,
        whisper_text=None,
        whisper_tokens=(),
        speaker="SPEAKER_00",
        text_matches=(),
    )

    candidates = build_sat_boundary_recovery_candidates(
        words,
        [evidence],
        {2, 7},
    )

    later = next(
        candidate
        for candidate in candidates
        if candidate.end_word_index == 7
    )

    assert later.word_indices == tuple(
        range(3, 8)
    )
    assert later.start_word_index == 3


def test_recovers_effective_geometry_at_sat_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = DatasetStorage(
        tmp_path / "dataset"
    )

    words = [
        word("I", 71.760, 71.840),
        word("wish", 71.840, 72.080),
        word("I", 72.080, 72.160),
        word("could", 72.160, 72.320),
        word("say", 72.320, 72.640),
        word("it's", 72.640, 72.880),
        word("easier", 72.880, 73.280),
        word("kid", 73.280, 73.440),
        word("Out", 73.440, 73.680),
        word("but", 74.720, 74.880),
    ]

    alignment = EffectiveWordAlignment(
        words=tuple(words),
        recoveries=(),
    )

    evidence = [
        AlignmentRegionEvidence(
            region_id="region_000003",
            start=71.73284375,
            end=73.63971875,
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
            text_matches=(),
        ),
    ]

    monkeypatch.setattr(
        word_alignment,
        "collect_region_evidence",
        lambda *args, **kwargs: evidence,
    )

    def fake_recover_candidate_alignment(
        storage: DatasetStorage,
        source_id: str,
        candidate: AlignmentRecoveryCandidate,
        words: list[dict],
        *,
        representation_name: str,
        language: str,
    ) -> LocalAlignmentRecovery:
        assert candidate.word_indices == tuple(
            range(8)
        )

        assert candidate.end_word_index == 7

        return LocalAlignmentRecovery(
            word_indices=candidate.word_indices,
            region_id=candidate.region_id,
            region_start=candidate.region_start,
            region_end=candidate.region_end,
            text=(
                "I wish I could say it's easier kid"
            ),
            words=(
                word("I", 71.733, 71.813),
                word("wish", 71.813, 72.133),
                word("I", 72.133, 72.133),
                word("could", 72.213, 72.293),
                word("say", 72.293, 72.613),
                word("it's", 72.613, 72.933),
                word("easier", 72.933, 73.253),
                word("kid", 73.253, 73.63971875),
            ),
        )

    monkeypatch.setattr(
        word_alignment,
        "recover_candidate_alignment",
        fake_recover_candidate_alignment,
    )

    result = recover_sat_boundary_alignments(
        storage,
        "source_001",
        alignment,
        {7},
        representation_name="center",
        language="English",
    )

    assert result.words[7]["text"] == "kid"
    assert result.words[7]["end"] == 73.63971875

    assert result.words[8]["text"] == "Out"
    assert result.words[8]["start"] == pytest.approx(
        73.440
    )
    assert result.words[8]["end"] == pytest.approx(
        73.680
    )

    assert result.suppressed_word_indices == (8,)

    assert len(result.recoveries) == 1
    assert result.recoveries[0].word_indices == tuple(
        range(8)
    )


def test_finds_word_overlapping_accepted_recovery() -> None:
    words = [
        word("kid", 73.253, 73.63971875),
        word("Out", 73.440, 73.680),
        word("but", 74.720, 74.880),
    ]

    recovery = LocalAlignmentRecovery(
        word_indices=(0,),
        region_id="region_000003",
        region_start=71.73284375,
        region_end=73.63971875,
        text="kid",
        words=(
            word("kid", 73.253, 73.63971875),
        ),
    )

    conflicts = find_post_recovery_word_conflicts(
        words,
        [recovery],
    )

    assert len(conflicts) == 1

    conflict = conflicts[0]

    assert conflict.recovered_word_index == 0
    assert conflict.conflicting_word_index == 1
    assert conflict.recovery_region_id == "region_000003"
    assert conflict.reason == "overlaps_recovered_word"


def test_post_recovery_conflict_allows_non_overlapping_next_word() -> None:
    words = [
        word("kid", 73.253, 73.63971875),
        word("but", 74.720, 74.880),
    ]

    recovery = LocalAlignmentRecovery(
        word_indices=(0,),
        region_id="region_000003",
        region_start=71.73284375,
        region_end=73.63971875,
        text="kid",
        words=(
            word("kid", 73.253, 73.63971875),
        ),
    )

    conflicts = find_post_recovery_word_conflicts(
        words,
        [recovery],
    )

    assert conflicts == []


def test_post_recovery_conflict_ignores_words_in_same_recovery() -> None:
    words = [
        word("easier", 72.933, 73.253),
        word("kid", 73.253, 73.63971875),
        word("but", 74.720, 74.880),
    ]

    recovery = LocalAlignmentRecovery(
        word_indices=(0, 1),
        region_id="region_000003",
        region_start=71.73284375,
        region_end=73.63971875,
        text="easier kid",
        words=(
            word("easier", 72.933, 73.253),
            word("kid", 73.253, 73.63971875),
        ),
    )

    conflicts = find_post_recovery_word_conflicts(
        words,
        [recovery],
    )

    assert conflicts == []


def test_post_recovery_conflict_ignores_next_recovered_word() -> None:
    words = [
        word("left", 10.000, 10.600),
        word("right", 10.500, 11.000),
    ]

    left_recovery = LocalAlignmentRecovery(
        word_indices=(0,),
        region_id="region_left",
        region_start=10.000,
        region_end=10.600,
        text="left",
        words=(
            word("left", 10.000, 10.600),
        ),
    )

    right_recovery = LocalAlignmentRecovery(
        word_indices=(1,),
        region_id="region_right",
        region_start=10.500,
        region_end=11.000,
        text="right",
        words=(
            word("right", 10.500, 11.000),
        ),
    )

    conflicts = find_post_recovery_word_conflicts(
        words,
        [left_recovery, right_recovery],
    )

    assert conflicts == []


def test_rejects_sat_boundary_recovery_that_does_not_reach_region_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = DatasetStorage(
        tmp_path / "dataset"
    )

    words = [
        word("One", 10.0, 10.2),
        word("two", 10.2, 10.4),
        word("three", 10.4, 10.6),
        word("Four", 10.7, 10.9),
    ]

    alignment = EffectiveWordAlignment(
        words=tuple(words),
        recoveries=(),
    )

    evidence = [
        AlignmentRegionEvidence(
            region_id="region_000001",
            start=9.9,
            end=15.0,
            speaker="SPEAKER_00",
            whisper_text=None,
            whisper_tokens=(),
            text_matches=(),
        ),
    ]

    monkeypatch.setattr(
        word_alignment,
        "collect_region_evidence",
        lambda *args, **kwargs: evidence,
    )

    def fake_recover_candidate_alignment(
        storage: DatasetStorage,
        source_id: str,
        candidate: AlignmentRecoveryCandidate,
        words: list[dict],
        *,
        representation_name: str,
        language: str,
    ) -> LocalAlignmentRecovery:
        assert candidate.word_indices == tuple(
            range(3)
        )
        assert candidate.end_word_index == 2

        return LocalAlignmentRecovery(
            word_indices=candidate.word_indices,
            region_id=candidate.region_id,
            region_start=candidate.region_start,
            region_end=candidate.region_end,
            text="One two three",
            words=(
                word("One", 10.0, 10.2),
                word("two", 10.2, 10.4),
                word("three", 10.4, 10.62),
            ),
        )

    monkeypatch.setattr(
        word_alignment,
        "recover_candidate_alignment",
        fake_recover_candidate_alignment,
    )

    result = recover_sat_boundary_alignments(
        storage,
        "source_001",
        alignment,
        {2},
        representation_name="center",
        language="English",
    )

    assert result == alignment


def test_recover_candidate_alignment_clamps_quantized_times_to_region(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = DatasetStorage(
        tmp_path / "dataset"
    )

    words = [
        word("I", 71.760, 71.840),
        word("wish", 71.840, 72.080),
        word("I", 72.080, 72.160),
        word("could", 72.160, 72.320),
        word("say", 72.320, 72.640),
        word("it's", 72.640, 72.880),
        word("easier", 72.880, 73.280),
        word("kid", 73.280, 73.440),
    ]

    candidate = AlignmentRecoveryCandidate(
        word_indices=tuple(range(8)),
        start_word_index=0,
        end_word_index=7,
        region_id="region_000003",
        region_start=71.73284375,
        region_end=73.63971875,
        whisper_text=(
            "I wish I could say it gets easier, kiddo."
        ),
        speaker="SPEAKER_00",
        issue_word_indices=(7,),
        issue_reasons=(
            "sat_boundary_inside_region",
        ),
    )

    monkeypatch.setattr(
        word_alignment,
        "resolve_source_representation",
        lambda *args, **kwargs: (
            None,
            tmp_path / "center.wav",
        ),
    )

    monkeypatch.setattr(
        word_alignment,
        "extract_audio_region",
        lambda **kwargs: None,
    )

    monkeypatch.setattr(
        word_alignment,
        "align_text_qwen3",
        lambda *args, **kwargs: ForcedAlignmentResult(
            language="English",
            text=(
                "I wish I could say it's easier kid"
            ),
            words=(
                word("I", 0.000, 0.080),
                word("wish", 0.080, 0.400),
                word("I", 0.400, 0.400),
                word("could", 0.480, 0.560),
                word("say", 0.560, 0.880),
                word("it's", 0.880, 1.200),
                word("easier", 1.200, 1.520),
                word("kid", 1.520, 1.920),
            ),
        ),
    )

    recovery = recover_candidate_alignment(
        storage,
        "source_001",
        candidate,
        words,
        representation_name="center",
        language="English",
    )

    assert recovery.words[-1]["end"] == pytest.approx(
        73.63971875
    )

    assert alignment_recovery_is_valid(
        words,
        recovery,
    )


def test_finds_unclaimed_word_after_recovery() -> None:
    words = [
        word("kid", 73.253, 73.63971875),
        word("Out", 73.440, 73.680),
    ]

    conflicts = [
        PostRecoveryWordConflict(
            recovered_word_index=0,
            conflicting_word_index=1,
            recovery_region_id="region_000003",
            reason="overlaps_recovered_word",
        ),
    ]

    evidence = [
        AlignmentRegionEvidence(
            region_id="region_000003",
            start=71.73284375,
            end=73.63971875,
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
            text_matches=(),
        ),
        AlignmentRegionEvidence(
            region_id="region_000004",
            start=74.73659375,
            end=75.52971875,
            speaker="SPEAKER_00",
            whisper_text="But I'll be lying.",
            whisper_tokens=("but", "i'll", "be", "lying"),
            text_matches=(),
        ),
    ]

    assert find_unclaimed_post_recovery_words(
        words,
        conflicts,
        evidence,
    ) == (1,)


def test_post_recovery_word_is_claimed_by_other_region() -> None:
    words = [
        word("left", 10.000, 10.600),
        word("right", 10.500, 10.900),
    ]

    conflicts = [
        PostRecoveryWordConflict(
            recovered_word_index=0,
            conflicting_word_index=1,
            recovery_region_id="region_left",
            reason="overlaps_recovered_word",
        ),
    ]

    evidence = [
        AlignmentRegionEvidence(
            region_id="region_left",
            start=10.000,
            end=10.600,
            speaker="SPEAKER_00",
            whisper_text="left",
            whisper_tokens=("left",),
            text_matches=(),
        ),
        AlignmentRegionEvidence(
            region_id="region_right",
            start=10.500,
            end=11.000,
            speaker="SPEAKER_01",
            whisper_text="right",
            whisper_tokens=("right",),
            text_matches=(),
        ),
    ]

    assert find_unclaimed_post_recovery_words(
        words,
        conflicts,
        evidence,
    ) == ()


def test_boundary_recovery_planner_deduplicates_same_region():
    from voice_dataset.word_alignment import (
        AlignmentIssue,
        AlignmentRegionEvidence,
        AlignmentTextMatch,
        _plan_boundary_recovery_anchors,
    )

    match = AlignmentTextMatch(
        start_word_index=142,
        end_word_index=142,
        word_indices=(142,),
        tokens=("thanks",),
    )

    evidence = AlignmentRegionEvidence(
        region_id="region_000051",
        start=411.7,
        end=412.3,
        speaker="SPEAKER_00",
        whisper_text="Thanks.",
        whisper_tokens=("thanks",),
        text_matches=(match,),
    )

    issues = [
        AlignmentIssue(
            word_indices=(),
            reasons=("excessive_inter_word_gap",),
            boundary_after_word_index=141,
        ),
        AlignmentIssue(
            word_indices=(),
            reasons=("excessive_inter_word_gap",),
            boundary_after_word_index=142,
        ),
    ]

    planned = _plan_boundary_recovery_anchors(
        issues,
        [evidence],
        claimed_word_indices=set(),
    )

    assert planned == [evidence]


def test_boundary_recovery_planner_rejects_ambiguous_regions():
    from voice_dataset.word_alignment import (
        AlignmentIssue,
        AlignmentRegionEvidence,
        AlignmentTextMatch,
        _plan_boundary_recovery_anchors,
    )

    match = AlignmentTextMatch(
        start_word_index=142,
        end_word_index=142,
        word_indices=(142,),
        tokens=("thanks",),
    )

    first = AlignmentRegionEvidence(
        region_id="region_000051",
        start=411.7,
        end=412.3,
        speaker="SPEAKER_00",
        whisper_text="Thanks.",
        whisper_tokens=("thanks",),
        text_matches=(match,),
    )

    second = AlignmentRegionEvidence(
        region_id="region_000130",
        start=800.0,
        end=800.6,
        speaker="SPEAKER_01",
        whisper_text="Thanks.",
        whisper_tokens=("thanks",),
        text_matches=(match,),
    )

    issues = [
        AlignmentIssue(
            word_indices=(),
            reasons=("excessive_inter_word_gap",),
            boundary_after_word_index=141,
        ),
        AlignmentIssue(
            word_indices=(),
            reasons=("excessive_inter_word_gap",),
            boundary_after_word_index=142,
        ),
    ]

    planned = _plan_boundary_recovery_anchors(
        issues,
        [first, second],
        claimed_word_indices=set(),
    )

    assert planned == []


def test_boundary_recovery_planner_respects_claimed_words():
    from voice_dataset.word_alignment import (
        AlignmentIssue,
        AlignmentRegionEvidence,
        AlignmentTextMatch,
        _plan_boundary_recovery_anchors,
    )

    match = AlignmentTextMatch(
        start_word_index=142,
        end_word_index=142,
        word_indices=(142,),
        tokens=("thanks",),
    )

    evidence = AlignmentRegionEvidence(
        region_id="region_000051",
        start=411.7,
        end=412.3,
        speaker="SPEAKER_00",
        whisper_text="Thanks.",
        whisper_tokens=("thanks",),
        text_matches=(match,),
    )

    planned = _plan_boundary_recovery_anchors(
        [
            AlignmentIssue(
                word_indices=(),
                reasons=("excessive_inter_word_gap",),
                boundary_after_word_index=141,
            ),
        ],
        [evidence],
        claimed_word_indices={142},
    )

    assert planned == []


def test_recover_alignment_candidates_batch_uses_one_qwen_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voice_dataset.continuous_asr import (
        ForcedAlignmentResult,
    )
    from voice_dataset.word_alignment import (
        AlignmentRecoveryCandidate,
        recover_alignment_candidates_batch,
    )

    storage = DatasetStorage(
        tmp_path / "dataset"
    )

    source_audio = (
        storage.root / "center.wav"
    )
    source_audio.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    source_audio.touch()

    storage.sources.append(
        {
            "schema_version": 1,
            "record_type": "source",
            "id": "source_001",
            "representations": {
                "center": {
                    "path": "center.wav",
                    "kind": "center",
                },
            },
            "metadata": {},
        }
    )

    words = [
        word("Hello", 10.0, 10.4),
        word("world", 20.0, 20.4),
    ]

    candidates = [
        AlignmentRecoveryCandidate(
            word_indices=(0,),
            start_word_index=0,
            end_word_index=0,
            region_id="region_1",
            region_start=9.9,
            region_end=10.5,
            whisper_text="Hello",
            speaker="SPEAKER_0",
            issue_word_indices=(0,),
            issue_reasons=("test",),
        ),
        AlignmentRecoveryCandidate(
            word_indices=(1,),
            start_word_index=1,
            end_word_index=1,
            region_id="region_2",
            region_start=19.9,
            region_end=20.5,
            whisper_text="world",
            speaker="SPEAKER_1",
            issue_word_indices=(1,),
            issue_reasons=("test",),
        ),
    ]

    monkeypatch.setattr(
        word_alignment,
        "extract_audio_region",
        lambda **kwargs: Path(
            kwargs["destination"]
        ).touch(),
    )

    calls = []

    def fake_align_texts(
        requests,
        *,
        max_inference_batch_size,
    ):
        calls.append(
            (
                list(requests),
                max_inference_batch_size,
            )
        )

        return [
            ForcedAlignmentResult(
                language="English",
                text="Hello",
                words=[
                    word("Hello", 0.1, 0.4),
                ],
            ),
            ForcedAlignmentResult(
                language="English",
                text="world",
                words=[
                    word("world", 0.1, 0.4),
                ],
            ),
        ]

    monkeypatch.setattr(
        word_alignment,
        "align_texts_qwen3",
        fake_align_texts,
    )

    recoveries = (
        recover_alignment_candidates_batch(
            storage,
            "source_001",
            candidates,
            words,
            representation_name="center",
            language="English",
            max_inference_batch_size=16,
        )
    )

    assert len(calls) == 1
    assert calls[0][1] == 16
    assert len(calls[0][0]) == 2

    assert len(recoveries) == 2

    assert recoveries[0].words[0]["start"] == (
        pytest.approx(10.0)
    )

    assert recoveries[1].words[0]["start"] == (
        pytest.approx(20.0)
    )

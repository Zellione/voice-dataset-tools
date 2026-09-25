from pathlib import Path

import voice_dataset.word_alignment as word_alignment
from voice_dataset.continuous_asr import ForcedAlignmentResult
from voice_dataset.storage import DatasetStorage
from voice_dataset.word_alignment import (
    AlignmentRegionEvidence,
    AlignmentTextMatch,
    LocalAlignmentRecovery,
    AlignmentComparison,
    StrandedAlignmentCandidate,
    compare_alignment_recovery,
    collect_boundary_text_evidence,
    detect_alignment_issues,
    find_text_matches,
    apply_alignment_recoveries,
    alignment_spans_overlap,
    find_stranded_alignment_candidates,
    recover_stranded_alignment,
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

    monkeypatch.setattr(
        word_alignment,
        "recover_region_alignment",
        fake_recover_region_alignment,
    )

    monkeypatch.setattr(
        word_alignment,
        "recover_stranded_alignment",
        fake_recover_stranded_alignment,
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
    monkeypatch.setattr(
        word_alignment,
        "recover_stranded_alignment",
        lambda *args, **kwargs: invalid_stranded_recovery,
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

    def fail_recovery(*args, **kwargs):
        raise RuntimeError("forced aligner failed")

    monkeypatch.setattr(
        word_alignment,
        "recover_stranded_alignment",
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

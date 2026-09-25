from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voice_dataset.diarization_conflicts import (
    analyze_continuous_word_region_conflicts,
    analyze_diarization_boundary_conflicts,
    group_continuous_word_region_conflicts,
    group_diarization_boundary_conflicts,
)
from voice_dataset.schema import (
    AudioRepresentation,
    CandidateRegion,
    SourceRecord,
)
from voice_dataset.storage import DatasetStorage


def test_analyze_diarization_boundary_conflicts(
    tmp_path: Path,
) -> None:
    storage = DatasetStorage(tmp_path / "dataset")

    activity_path = (
        storage.root
        / "evidence"
        / "diarization"
        / "sortformer"
        / "speaker_activity.npy"
    )
    activity_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    activity = np.zeros(
        (1, 50, 4),
        dtype=np.float32,
    )

    # Boundary at 1.0 s with 80 ms frames:
    # before -> frames 10:12 = 0.80-0.96
    # after  -> frames 12:14 = 0.96-1.12
    activity[0, 10:14, 2] = 0.9

    # Second boundary at 2.0 s.
    activity[0, 23:27, 1] = 0.7

    np.save(
        activity_path,
        activity,
        allow_pickle=False,
    )

    storage.sources.append(
        SourceRecord(
            id="source_001",
            representations={
                "center": AudioRepresentation(
                    path="audio/source_001/center.wav",
                    kind="center",
                    processor="test",
                    sample_rate=48000,
                    channels=1,
                    duration=4.0,
                ),
            },
            metadata={
                "continuous_asr": {
                    "qwen3": {
                        "words": [
                            {
                                "text": "hello",
                                "start": 0.8,
                                "end": 1.2,
                            },
                            {
                                "text": "world",
                                "start": 1.8,
                                "end": 2.2,
                            },
                        ],
                        "utterances": [],
                    },
                },
                "diarization": {
                    "sortformer": {
                        "model": "test",
                        "representation": "center",
                        "segments": [],
                        "activity": {
                            "path": (
                                "evidence/diarization/"
                                "sortformer/"
                                "speaker_activity.npy"
                            ),
                            "shape": [1, 50, 4],
                            "dtype": "float32",
                            "frame_duration": 0.08,
                            "sha256": "test",
                        },
                    },
                },
            },
        ).to_dict()
    )

    regions = [
        CandidateRegion(
            id="region_001",
            source_id="source_001",
            source_start=0.5,
            source_end=1.0,
            detector="test",
            detector_label="SPEAKER_00",
        ),
        CandidateRegion(
            id="region_002",
            source_id="source_001",
            source_start=1.0,
            source_end=2.0,
            detector="test",
            detector_label="SPEAKER_01",
        ),
        CandidateRegion(
            id="region_003",
            source_id="source_001",
            source_start=2.5,
            source_end=3.0,
            detector="test",
            detector_label="SPEAKER_02",
        ),
    ]

    for region in regions:
        storage.regions.append(
            region.to_dict()
        )

    conflicts = (
        analyze_diarization_boundary_conflicts(
            storage,
            "source_001",
        )
    )

    assert len(conflicts) == 2

    first = conflicts[0]

    assert first.left_region_id == "region_001"
    assert first.right_region_id == "region_002"
    assert first.left_speaker == "SPEAKER_00"
    assert first.right_speaker == "SPEAKER_01"
    assert first.boundary == pytest.approx(1.0)
    assert first.region_gap == pytest.approx(0.0)

    assert first.crossing_word == "hello"
    assert first.word_start == pytest.approx(0.8)
    assert first.word_end == pytest.approx(1.2)
    assert first.word_boundary_offset == pytest.approx(
        0.2
    )
    assert first.word_boundary_fraction == pytest.approx(
        0.5
    )

    assert first.before.start_frame == 10
    assert first.before.end_frame == 12
    assert first.before.frame_start == pytest.approx(
        0.8
    )
    assert first.before.frame_end == pytest.approx(
        0.96
    )
    assert first.before.dominant_speaker == 2
    assert (
        first.before.dominant_activity
        == pytest.approx(0.9)
    )

    assert first.after.start_frame == 12
    assert first.after.end_frame == 14
    assert first.after.frame_start == pytest.approx(
        0.96
    )
    assert first.after.frame_end == pytest.approx(
        1.12
    )
    assert first.after.dominant_speaker == 2
    assert (
        first.after.dominant_activity
        == pytest.approx(0.9)
    )

    second = conflicts[1]

    assert second.left_region_id == "region_002"
    assert second.right_region_id == "region_003"
    assert second.boundary == pytest.approx(2.0)

    # Word crossing is evidence about the left boundary.
    # The 500 ms gap is preserved rather than interpreted
    # as permission to merge the right region.
    assert second.region_gap == pytest.approx(0.5)
    assert second.crossing_word == "world"

    touching = [
        conflict
        for conflict in conflicts
        if conflict.region_gap == 0.0
    ]

    runs = group_diarization_boundary_conflicts(
        touching
    )

    assert len(runs) == 1

    assert runs[0].region_ids == (
        "region_001",
        "region_002",
    )

    assert runs[0].conflicts == (
        conflicts[0],
    )


def test_analyze_and_group_continuous_word_region_conflicts(
    tmp_path: Path,
) -> None:
    storage = DatasetStorage(tmp_path / "dataset")

    storage.sources.append(
        SourceRecord(
            id="source_001",
            representations={
                "center": AudioRepresentation(
                    path="audio/source_001/center.wav",
                    kind="center",
                    processor="test",
                    sample_rate=48000,
                    channels=1,
                    duration=8.0,
                ),
            },
            metadata={
                "continuous_asr": {
                    "qwen3": {
                        "words": [
                            {
                                "text": "first",
                                "start": 0.8,
                                "end": 1.2,
                            },
                            {
                                "text": "second",
                                "start": 2.8,
                                "end": 3.2,
                            },
                            {
                                "text": "third",
                                "start": 3.8,
                                "end": 4.2,
                            },
                            {
                                "text": "gapped",
                                "start": 5.8,
                                "end": 6.7,
                            },
                        ],
                        "utterances": [],
                    },
                },
            },
        ).to_dict()
    )

    regions = [
        CandidateRegion(
            id="region_001",
            source_id="source_001",
            source_start=0.5,
            source_end=1.0,
            detector="test",
            detector_label="SPEAKER_00",
        ),
        CandidateRegion(
            id="region_002",
            source_id="source_001",
            source_start=1.0,
            source_end=1.5,
            detector="test",
            detector_label="SPEAKER_01",
        ),
        CandidateRegion(
            id="region_003",
            source_id="source_001",
            source_start=2.5,
            source_end=3.0,
            detector="test",
            detector_label="SPEAKER_00",
        ),
        CandidateRegion(
            id="region_004",
            source_id="source_001",
            source_start=3.0,
            source_end=4.0,
            detector="test",
            detector_label="SPEAKER_01",
        ),
        CandidateRegion(
            id="region_005",
            source_id="source_001",
            source_start=4.0,
            source_end=4.5,
            detector="test",
            detector_label="SPEAKER_00",
        ),
        CandidateRegion(
            id="region_006",
            source_id="source_001",
            source_start=5.5,
            source_end=6.0,
            detector="test",
            detector_label="SPEAKER_00",
        ),
        CandidateRegion(
            id="region_007",
            source_id="source_001",
            source_start=6.5,
            source_end=7.0,
            detector="test",
            detector_label="SPEAKER_01",
        ),
    ]

    for region in regions:
        storage.regions.append(
            region.to_dict()
        )

    conflicts = (
        analyze_continuous_word_region_conflicts(
            storage,
            "source_001",
        )
    )

    assert len(conflicts) == 3

    assert conflicts[0].word == "first"
    assert conflicts[0].region_ids == (
        "region_001",
        "region_002",
    )
    assert conflicts[0].region_speakers == (
        "SPEAKER_00",
        "SPEAKER_01",
    )

    assert conflicts[1].word == "second"
    assert conflicts[1].region_ids == (
        "region_003",
        "region_004",
    )

    assert conflicts[2].word == "third"
    assert conflicts[2].region_ids == (
        "region_004",
        "region_005",
    )

    # "gapped" overlaps regions 006 and 007, but
    # those regions are not a contiguous Community chain.
    assert all(
        conflict.word != "gapped"
        for conflict in conflicts
    )

    runs = group_continuous_word_region_conflicts(
        conflicts
    )

    assert len(runs) == 2

    assert runs[0].region_ids == (
        "region_001",
        "region_002",
    )
    assert tuple(
        conflict.word
        for conflict in runs[0].conflicts
    ) == ("first",)

    # "second" and "third" share region_004, so their
    # region spans form one transitive component.
    assert runs[1].region_ids == (
        "region_003",
        "region_004",
        "region_005",
    )
    assert tuple(
        conflict.word
        for conflict in runs[1].conflicts
    ) == (
        "second",
        "third",
    )


def test_continuous_word_region_conflicts_use_injected_words(
    tmp_path: Path,
) -> None:
    storage = DatasetStorage(tmp_path / "dataset")

    storage.sources.append(
        SourceRecord(
            id="source_001",
            representations={
                "center": AudioRepresentation(
                    path="audio/source_001/center.wav",
                    kind="center",
                    processor="test",
                    sample_rate=48000,
                    channels=1,
                    duration=4.0,
                ),
            },
            metadata={
                "continuous_asr": {
                    "qwen3": {
                        "words": [
                            {
                                "text": "raw",
                                "start": 2.0,
                                "end": 2.5,
                            },
                        ],
                        "utterances": [],
                    },
                },
            },
        ).to_dict()
    )

    for region in (
        CandidateRegion(
            id="region_001",
            source_id="source_001",
            source_start=0.0,
            source_end=1.0,
            detector="test",
            detector_label="SPEAKER_00",
        ),
        CandidateRegion(
            id="region_002",
            source_id="source_001",
            source_start=1.0,
            source_end=2.0,
            detector="test",
            detector_label="SPEAKER_01",
        ),
    ):
        storage.regions.append(
            region.to_dict()
        )

    conflicts = analyze_continuous_word_region_conflicts(
        storage,
        "source_001",
        words=[
            {
                "text": "effective",
                "start": 0.8,
                "end": 1.2,
            },
        ],
    )

    assert len(conflicts) == 1

    conflict = conflicts[0]

    assert conflict.word == "effective"
    assert conflict.word_start == pytest.approx(0.8)
    assert conflict.word_end == pytest.approx(1.2)
    assert conflict.region_ids == (
        "region_001",
        "region_002",
    )
    assert conflict.region_speakers == (
        "SPEAKER_00",
        "SPEAKER_01",
    )

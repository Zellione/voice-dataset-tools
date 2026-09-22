from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from voice_dataset.diarization import (
    import_sortformer_evidence,
)
from voice_dataset.schema import (
    AudioRepresentation,
    SourceRecord,
)
from voice_dataset.storage import DatasetStorage


def make_storage(
    tmp_path: Path,
) -> DatasetStorage:
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
                    duration=10.0,
                    purposes=[
                        "boundary_analysis",
                    ],
                ),
            },
        ).to_dict()
    )

    return storage


def make_sortformer_output(
    tmp_path: Path,
) -> tuple[Path, Path, np.ndarray]:
    segments_path = tmp_path / "segments.json"
    activity_path = tmp_path / "speaker_activity.npy"

    segments_path.write_text(
        json.dumps(
            [
                "1.000 2.000 speaker_0",
                "3.500 4.250 speaker_1",
            ]
        ),
        encoding="utf-8",
    )

    activity = np.zeros(
        (1, 125, 4),
        dtype=np.float32,
    )
    activity[0, 12:25, 0] = 0.9
    activity[0, 44:54, 1] = 0.8

    np.save(
        activity_path,
        activity,
        allow_pickle=False,
    )

    return (
        segments_path,
        activity_path,
        activity,
    )


def test_import_sortformer_evidence(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    (
        segments_path,
        activity_path,
        activity,
    ) = make_sortformer_output(tmp_path)

    updated = import_sortformer_evidence(
        storage,
        "source_001",
        "sortformer",
        model="nvidia/test-sortformer",
        representation="center",
        segments_path=segments_path,
        activity_path=activity_path,
        frame_duration=0.08,
    )

    evidence = updated["metadata"][
        "diarization"
    ]["sortformer"]

    assert evidence["model"] == (
        "nvidia/test-sortformer"
    )
    assert evidence["representation"] == "center"

    assert evidence["segments"] == [
        {
            "start": 1.0,
            "end": 2.0,
            "speaker": "speaker_0",
        },
        {
            "start": 3.5,
            "end": 4.25,
            "speaker": "speaker_1",
        },
    ]

    activity_reference = evidence["activity"]

    assert activity_reference["shape"] == [
        1,
        125,
        4,
    ]
    assert activity_reference["dtype"] == "float32"
    assert activity_reference[
        "frame_duration"
    ] == pytest.approx(0.08)

    stored_path = (
        storage.root
        / activity_reference["path"]
    )

    assert stored_path == (
        storage.root
        / "evidence"
        / "diarization"
        / "sortformer"
        / "speaker_activity.npy"
    )

    assert stored_path.is_file()

    stored_activity = np.load(
        stored_path,
        allow_pickle=False,
    )

    np.testing.assert_array_equal(
        stored_activity,
        activity,
    )


def test_import_sortformer_evidence_is_idempotent(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    (
        segments_path,
        activity_path,
        _,
    ) = make_sortformer_output(tmp_path)

    first = import_sortformer_evidence(
        storage,
        "source_001",
        "sortformer",
        model="nvidia/test-sortformer",
        representation="center",
        segments_path=segments_path,
        activity_path=activity_path,
        frame_duration=0.08,
    )

    second = import_sortformer_evidence(
        storage,
        "source_001",
        "sortformer",
        model="nvidia/test-sortformer",
        representation="center",
        segments_path=segments_path,
        activity_path=activity_path,
        frame_duration=0.08,
    )

    assert second == first

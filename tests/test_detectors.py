import json
from pathlib import Path

import pytest

from voice_dataset.detectors import (
    import_detector_regions,
    load_detector_output,
)
from voice_dataset.storage import DatasetStorage


def make_storage(
    tmp_path: Path,
) -> DatasetStorage:
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    return DatasetStorage(dataset)


def write_detector_output(
    tmp_path: Path,
    *,
    revision=...,
) -> Path:
    document = {
        "format": "voice-dataset-detector-output",
        "version": 1,
        "detector": {
            "name": "pyannote-community-1",
            "model": (
                "pyannote/"
                "speaker-diarization-community-1"
            ),
        },
        "source": {
            "path": str(
                tmp_path / "source.wav"
            ),
            "sample_rate": 48000,
            "channels": 1,
            "duration": 10.0,
        },
        "regions": [
            {
                "start": 1.0,
                "end": 2.0,
                "label": "SPEAKER_00",
            },
        ],
    }

    if revision is not ...:
        document["detector"]["revision"] = (
            revision
        )

    path = tmp_path / "detector-output.json"

    path.write_text(
        json.dumps(document),
        encoding="utf-8",
    )

    return path


def test_import_preserves_detector_revision(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    path = write_detector_output(
        tmp_path,
        revision="abc123",
    )

    output = load_detector_output(path)

    result = import_detector_regions(
        storage,
        output,
        source_id="source_001",
    )

    assert len(result.imported) == 1
    assert result.skipped == 0

    region = storage.regions.load()[0]

    assert (
        region["metadata"]["detector_model"]
        == (
            "pyannote/"
            "speaker-diarization-community-1"
        )
    )
    assert (
        region["metadata"]["detector_revision"]
        == "abc123"
    )


def test_import_without_revision_is_backward_compatible(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    path = write_detector_output(
        tmp_path,
    )

    output = load_detector_output(path)

    first = import_detector_regions(
        storage,
        output,
        source_id="source_001",
    )

    second = import_detector_regions(
        storage,
        output,
        source_id="source_001",
    )

    assert len(first.imported) == 1
    assert first.skipped == 0

    assert len(second.imported) == 0
    assert second.skipped == 1

    region = storage.regions.load()[0]

    assert (
        "detector_revision"
        not in region["metadata"]
    )


def test_null_revision_is_not_persisted(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    path = write_detector_output(
        tmp_path,
        revision=None,
    )

    output = load_detector_output(path)

    result = import_detector_regions(
        storage,
        output,
        source_id="source_001",
    )

    assert len(result.imported) == 1

    region = storage.regions.load()[0]

    assert (
        "detector_revision"
        not in region["metadata"]
    )


def test_load_rejects_invalid_detector_revision(
    tmp_path: Path,
) -> None:
    path = write_detector_output(
        tmp_path,
        revision=123,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Detector revision must be "
            "a string or null"
        ),
    ):
        load_detector_output(path)


def test_different_detector_revision_is_distinct_evidence(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    first_path = write_detector_output(
        tmp_path,
        revision="revision-a",
    )

    first_output = load_detector_output(
        first_path
    )

    first = import_detector_regions(
        storage,
        first_output,
        source_id="source_001",
    )

    second_path = write_detector_output(
        tmp_path,
        revision="revision-b",
    )

    second_output = load_detector_output(
        second_path
    )

    second = import_detector_regions(
        storage,
        second_output,
        source_id="source_001",
    )

    assert len(first.imported) == 1
    assert first.skipped == 0

    assert len(second.imported) == 1
    assert second.skipped == 0

    regions = storage.regions.load()

    assert len(regions) == 2

    assert {
        region["metadata"][
            "detector_revision"
        ]
        for region in regions
    } == {
        "revision-a",
        "revision-b",
    }


def test_same_detector_revision_is_idempotent(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    path = write_detector_output(
        tmp_path,
        revision="abc123",
    )

    output = load_detector_output(path)

    first = import_detector_regions(
        storage,
        output,
        source_id="source_001",
    )

    second = import_detector_regions(
        storage,
        output,
        source_id="source_001",
    )

    assert len(first.imported) == 1
    assert first.skipped == 0

    assert len(second.imported) == 0
    assert second.skipped == 1

    assert len(storage.regions.load()) == 1


def test_import_preserves_detector_parameters(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    path = write_detector_output(
        tmp_path,
        revision="abc123",
    )

    document = json.loads(
        path.read_text(encoding="utf-8")
    )

    document["detector"]["parameters"] = {
        "min_speakers": 2,
    }

    path.write_text(
        json.dumps(document),
        encoding="utf-8",
    )

    output = load_detector_output(path)

    result = import_detector_regions(
        storage,
        output,
        source_id="source_001",
    )

    assert len(result.imported) == 1

    region = storage.regions.load()[0]

    assert region["metadata"][
        "detector_parameters"
    ] == {
        "min_speakers": 2,
    }


def test_empty_detector_parameters_are_not_persisted(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    path = write_detector_output(
        tmp_path,
        revision="abc123",
    )

    document = json.loads(
        path.read_text(encoding="utf-8")
    )

    document["detector"]["parameters"] = {}

    path.write_text(
        json.dumps(document),
        encoding="utf-8",
    )

    output = load_detector_output(path)

    result = import_detector_regions(
        storage,
        output,
        source_id="source_001",
    )

    assert len(result.imported) == 1

    region = storage.regions.load()[0]

    assert (
        "detector_parameters"
        not in region["metadata"]
    )


def test_load_rejects_invalid_detector_parameters(
    tmp_path: Path,
) -> None:
    path = write_detector_output(
        tmp_path,
        revision="abc123",
    )

    document = json.loads(
        path.read_text(encoding="utf-8")
    )

    document["detector"]["parameters"] = [
        "not",
        "an",
        "object",
    ]

    path.write_text(
        json.dumps(document),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Detector parameters must be "
            "an object"
        ),
    ):
        load_detector_output(path)


def test_different_detector_parameters_are_distinct_evidence(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    path = write_detector_output(
        tmp_path,
        revision="abc123",
    )

    document = json.loads(
        path.read_text(encoding="utf-8")
    )

    document["detector"]["parameters"] = {}

    path.write_text(
        json.dumps(document),
        encoding="utf-8",
    )

    first_output = load_detector_output(path)

    first = import_detector_regions(
        storage,
        first_output,
        source_id="source_001",
    )

    document["detector"]["parameters"] = {
        "min_speakers": 2,
    }

    path.write_text(
        json.dumps(document),
        encoding="utf-8",
    )

    second_output = load_detector_output(path)

    second = import_detector_regions(
        storage,
        second_output,
        source_id="source_001",
    )

    assert len(first.imported) == 1
    assert first.skipped == 0

    assert len(second.imported) == 1
    assert second.skipped == 0

    assert len(storage.regions.load()) == 2


def test_same_detector_parameters_are_idempotent(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    path = write_detector_output(
        tmp_path,
        revision="abc123",
    )

    document = json.loads(
        path.read_text(encoding="utf-8")
    )

    document["detector"]["parameters"] = {
        "min_speakers": 2,
    }

    path.write_text(
        json.dumps(document),
        encoding="utf-8",
    )

    output = load_detector_output(path)

    first = import_detector_regions(
        storage,
        output,
        source_id="source_001",
    )

    second = import_detector_regions(
        storage,
        output,
        source_id="source_001",
    )

    assert len(first.imported) == 1
    assert first.skipped == 0

    assert len(second.imported) == 0
    assert second.skipped == 1

import pytest
from voice_dataset.schema import (
    SourceRecord,
    TurnRecord,
)
from voice_dataset.storage import DatasetStorage
from voice_dataset.turn_curation import (
    merge_and_prepare_turns,
    split_and_prepare_turn,
    prepare_curated_source_turns,
    project_curated_turn,
    trim_and_prepare_turn,
    _project_split,
    migrate_legacy_ignored_turns,
)


def _storage_with_qwen(tmp_path):
    storage = DatasetStorage(tmp_path)

    source = SourceRecord(
        id="source_001",
        media_path="/tmp/source.wav",
        metadata={
            "continuous_asr": {
                "qwen3": {
                    "words": [
                        {
                            "text": "hello",
                            "start": 1.0,
                            "end": 1.4,
                        },
                        {
                            "text": "there",
                            "start": 1.4,
                            "end": 1.8,
                        },
                        {
                            "text": "general",
                            "start": 3.0,
                            "end": 3.5,
                        },
                    ],
                    "utterances": [
                        {
                            "word_start": 0,
                            "word_end": 2,
                        },
                        {
                            "word_start": 2,
                            "word_end": 3,
                        },
                    ],
                }
            },
            "utterance_reconciliation": {
                "mode": "automatic",
                "asr_evidence": "qwen3",
                "word_ranges": [
                    [0, 1],
                    [2, 2],
                ],
            },
        },
    )

    storage.add_source(source)

    return storage


def test_project_curated_turn_projects_words(tmp_path):
    storage = _storage_with_qwen(tmp_path)

    result = project_curated_turn(
        storage,
        source_id="source_001",
        source_start=0.9,
        source_end=1.9,
        language="en",
    )

    assert result.evidence_name == "qwen3"
    assert result.projection.word_indices == (0, 1)
    assert result.transcript == "hello there"
    assert result.language == "en"
    assert result.start_word_index == 0
    assert result.end_word_index == 1


def test_project_curated_turn_allows_empty_projection(
    tmp_path,
):
    storage = _storage_with_qwen(tmp_path)

    result = project_curated_turn(
        storage,
        source_id="source_001",
        source_start=2.0,
        source_end=2.5,
    )

    assert result.projection.word_indices == ()
    assert result.transcript is None
    assert result.start_word_index is None
    assert result.end_word_index is None


def test_project_curated_turn_requires_asr_evidence(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    storage.add_source(
        SourceRecord(
            id="source_001",
            media_path="/tmp/source.wav",
        )
    )

    try:
        project_curated_turn(
            storage,
            source_id="source_001",
            source_start=1.0,
            source_end=2.0,
        )
    except ValueError as error:
        assert "no continuous ASR evidence" in str(
            error
        )
    else:
        raise AssertionError(
            "Expected missing ASR evidence to fail"
        )


def test_project_curated_turn_rejects_non_contiguous_words(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    storage.add_source(
        SourceRecord(
            id="source_001",
            media_path="/tmp/source.wav",
            metadata={
                "continuous_asr": {
                    "qwen3": {
                        "words": [
                            {
                                "text": "first",
                                "start": 1.0,
                                "end": 1.4,
                            },
                            {
                                "text": "stranded",
                                "start": 5.0,
                                "end": 5.4,
                            },
                            {
                                "text": "third",
                                "start": 1.5,
                                "end": 1.9,
                            },
                        ],
                        "utterances": [],
                    }
                }
            },
        )
    )

    try:
        project_curated_turn(
            storage,
            source_id="source_001",
            source_start=0.9,
            source_end=2.0,
        )
    except ValueError as error:
        assert (
            "non-contiguous continuous ASR words"
            in str(error)
        )
        assert "(0, 2)" in str(error)
    else:
        raise AssertionError(
            "Expected non-contiguous projection to fail"
        )


def test_prepare_curated_source_turns_prepares_review_and_speaker_evidence(
    tmp_path,
    monkeypatch,
):
    storage = DatasetStorage(tmp_path)
    calls: list[tuple[str, str]] = []

    def fake_prepare_review(
        actual_storage,
        source_id,
    ):
        assert actual_storage is storage
        calls.append(("review", source_id))

    def fake_prepare_speaker(
        actual_storage,
        source_id,
        *,
        capture_output=False,
    ):
        assert actual_storage is storage
        assert capture_output is False
        calls.append(("speaker", source_id))

    monkeypatch.setattr(
        "voice_dataset.turn_curation.prepare_source_review_audio",
        fake_prepare_review,
    )
    monkeypatch.setattr(
        "voice_dataset.turn_curation.prepare_source_speaker_evidence",
        fake_prepare_speaker,
    )

    prepare_curated_source_turns(
        storage,
        "source_001",
    )

    assert calls == [
        ("review", "source_001"),
        ("speaker", "source_001"),
    ]


def test_merge_and_prepare_turns_adds_asr_provenance_and_prepares(
    tmp_path,
    monkeypatch,
):
    storage = _storage_with_qwen(tmp_path)

    merged = {
        "schema_version": 1,
        "record_type": "turn",
        "id": "turn_000001",
        "source_id": "source_001",
        "source_start": 0.9,
        "source_end": 1.9,
        "source_regions": [
            "region_000001",
            "region_000002",
        ],
        "language": "en",
        "transcript": "Manually corrected transcript",
        "assignment": {
            "status": "unknown",
            "voice_id": None,
            "method": None,
        },
        "representations": {},
        "embeddings": {},
        "metadata": {
            "creation": {
                "method": "manual_merge",
                "source_turn_ids": [
                    "turn_000001",
                    "turn_000002",
                ],
            },
        },
    }

    storage.turns.append(merged)

    storage.turns.append(
        {
            "schema_version": 1,
            "record_type": "turn",
            "id": "turn_000002",
            "source_id": "source_001",
            "source_start": 1.4,
            "source_end": 1.9,
            "source_regions": [
                "region_000002",
            ],
            "language": "en",
            "transcript": "second",
            "assignment": {
                "status": "unknown",
                "voice_id": None,
                "method": None,
            },
            "representations": {},
            "embeddings": {},
            "metadata": {},
        }
    )

    def fake_merge(
        actual_storage,
        turn_ids,
    ):
        assert actual_storage is storage
        assert turn_ids == [
            "turn_000001",
            "turn_000002",
        ]
        return actual_storage.get_turn(
            "turn_000001"
        )

    prepared: list[str] = []

    def fake_prepare(
        actual_storage,
        source_id,
        *,
        capture_output=False,
    ):
        assert actual_storage is storage
        assert capture_output is False
        prepared.append(source_id)

    monkeypatch.setattr(
        "voice_dataset.turn_curation.merge_turns",
        fake_merge,
    )
    monkeypatch.setattr(
        "voice_dataset.turn_curation.prepare_curated_source_turns",
        fake_prepare,
    )

    result = merge_and_prepare_turns(
        storage,
        [
            "turn_000001",
            "turn_000002",
        ],
    )

    assert (
        result["transcript"]
        == "Manually corrected transcript"
    )
    assert result["language"] == "en"

    assert result["metadata"]["word_range"] == {
        "start": 0,
        "end": 1,
    }
    assert result["metadata"]["continuous_asr"] == {
        "evidence": "qwen3",
        "word_indices": [0, 1],
    }

    assert result["metadata"]["creation"] == {
        "method": "manual_merge",
        "source_turn_ids": [
            "turn_000001",
            "turn_000002",
        ],
    }

    assert prepared == ["source_001"]

    source = storage.get_source("source_001")
    
    assert source is not None
    assert source["metadata"][
        "utterance_reconciliation"
    ]["mode"] == "curated"


def test_merge_and_prepare_turns_does_not_merge_when_projection_fails(
    tmp_path,
    monkeypatch,
):
    storage = _storage_with_qwen(tmp_path)

    first = {
        "schema_version": 1,
        "record_type": "turn",
        "id": "turn_000001",
        "source_id": "source_001",
        "source_start": 2.0,
        "source_end": 2.2,
        "source_regions": ["region_000001"],
        "language": "en",
        "transcript": "first",
        "assignment": {
            "status": "unknown",
            "voice_id": None,
            "method": None,
        },
        "representations": {},
        "embeddings": {},
        "metadata": {},
    }

    second = {
        "schema_version": 1,
        "record_type": "turn",
        "id": "turn_000002",
        "source_id": "source_001",
        "source_start": 2.3,
        "source_end": 2.5,
        "source_regions": ["region_000002"],
        "language": "en",
        "transcript": "second",
        "assignment": {
            "status": "unknown",
            "voice_id": None,
            "method": None,
        },
        "representations": {},
        "embeddings": {},
        "metadata": {},
    }

    storage.turns.append(first)
    storage.turns.append(second)

    merge_called = False

    def fake_merge(
        actual_storage,
        turn_ids,
    ):
        nonlocal merge_called
        merge_called = True
        raise AssertionError(
            "merge_turns must not run after "
            "a failed projection preflight"
        )

    monkeypatch.setattr(
        "voice_dataset.turn_curation.merge_turns",
        fake_merge,
    )

    before = storage.turns.load()

    with pytest.raises(
        ValueError,
        match="does not project to continuous ASR words",
    ):
        merge_and_prepare_turns(
            storage,
            [
                "turn_000001",
                "turn_000002",
            ],
        )

    after = storage.turns.load()

    assert merge_called is False
    assert after == before


def test_project_split_projects_both_child_turns(
    tmp_path,
):
    storage = _storage_with_qwen(tmp_path)

    storage.regions.append(
        {
            "schema_version": 1,
            "record_type": "region",
            "id": "region_000001",
            "source_id": "source_001",
            "source_start": 1.0,
            "source_end": 1.4,
        }
    )
    storage.regions.append(
        {
            "schema_version": 1,
            "record_type": "region",
            "id": "region_000002",
            "source_id": "source_001",
            "source_start": 1.4,
            "source_end": 1.8,
        }
    )

    storage.turns.append(
        {
            "schema_version": 1,
            "record_type": "turn",
            "id": "turn_000001",
            "source_id": "source_001",
            "source_start": 1.0,
            "source_end": 1.8,
            "source_regions": [
                "region_000001",
                "region_000002",
            ],
            "language": "en",
            "transcript": "manually corrected text",
            "assignment": {
                "status": "unknown",
                "voice_id": None,
                "method": None,
            },
            "representations": {},
            "embeddings": {},
            "metadata": {},
        }
    )

    source_id, left, right = _project_split(
        storage,
        "turn_000001",
        after_region_id="region_000001",
        evidence_name="qwen3",
    )

    assert source_id == "source_001"

    assert left.projection.word_indices == (0,)
    assert left.transcript == "hello"
    assert left.start_word_index == 0
    assert left.end_word_index == 0
    assert left.language == "en"

    assert right.projection.word_indices == (1,)
    assert right.transcript == "there"
    assert right.start_word_index == 1
    assert right.end_word_index == 1
    assert right.language == "en"


def test_split_and_prepare_turn_projects_children_and_prepares(
    tmp_path,
    monkeypatch,
):
    storage = _storage_with_qwen(tmp_path)

    storage.regions.append(
        {
            "schema_version": 1,
            "record_type": "region",
            "id": "region_000001",
            "source_id": "source_001",
            "source_start": 1.0,
            "source_end": 1.4,
        }
    )
    storage.regions.append(
        {
            "schema_version": 1,
            "record_type": "region",
            "id": "region_000002",
            "source_id": "source_001",
            "source_start": 1.4,
            "source_end": 1.8,
        }
    )

    storage.turns.append(
        {
            "schema_version": 1,
            "record_type": "turn",
            "id": "turn_000001",
            "source_id": "source_001",
            "source_start": 1.0,
            "source_end": 1.8,
            "source_regions": [
                "region_000001",
                "region_000002",
            ],
            "language": "en",
            "transcript":
                "manually corrected combined text",
            "assignment": {
                "status": "unknown",
                "voice_id": None,
                "method": None,
            },
            "representations": {},
            "embeddings": {},
            "metadata": {},
        }
    )

    prepared = []

    monkeypatch.setattr(
        "voice_dataset.turn_curation."
        "prepare_curated_source_turns",
        lambda storage, source_id, capture_output=False:
            prepared.append(source_id),
    )

    left, right = split_and_prepare_turn(
        storage,
        "turn_000001",
        after_region_id="region_000001",
    )

    assert left["id"] == "turn_000001"
    assert right["id"] != left["id"]

    assert left["transcript"] == "hello"
    assert right["transcript"] == "there"

    assert left["language"] == "en"
    assert right["language"] == "en"

    assert left["metadata"]["word_range"] == {
        "start": 0,
        "end": 0,
    }
    assert right["metadata"]["word_range"] == {
        "start": 1,
        "end": 1,
    }

    assert left["metadata"]["continuous_asr"] == {
        "evidence": "qwen3",
        "word_indices": [0],
    }
    assert right["metadata"]["continuous_asr"] == {
        "evidence": "qwen3",
        "word_indices": [1],
    }

    assert (
        left["metadata"]["creation"]["method"]
        == "manual_split"
    )
    assert (
        right["metadata"]["creation"]["method"]
        == "manual_split"
    )

    assert prepared == ["source_001"]

    source = storage.get_source("source_001")
    
    assert source is not None
    assert source["metadata"][
        "utterance_reconciliation"
    ]["mode"] == "curated"


def test_split_and_prepare_turn_does_not_split_when_projection_fails(
    tmp_path,
    monkeypatch,
):
    storage = _storage_with_qwen(tmp_path)

    storage.regions.append(
        {
            "schema_version": 1,
            "record_type": "region",
            "id": "region_000001",
            "source_id": "source_001",
            "source_start": 1.0,
            "source_end": 1.4,
        }
    )
    storage.regions.append(
        {
            "schema_version": 1,
            "record_type": "region",
            "id": "region_000002",
            "source_id": "source_001",
            "source_start": 2.0,
            "source_end": 2.5,
        }
    )

    storage.turns.append(
        {
            "schema_version": 1,
            "record_type": "turn",
            "id": "turn_000001",
            "source_id": "source_001",
            "source_start": 1.0,
            "source_end": 2.5,
            "source_regions": [
                "region_000001",
                "region_000002",
            ],
            "language": "en",
            "transcript": "combined text",
            "assignment": {
                "status": "unknown",
                "voice_id": None,
                "method": None,
            },
            "representations": {},
            "embeddings": {},
            "metadata": {},
        }
    )

    split_called = False

    def fail_split(*args, **kwargs):
        nonlocal split_called
        split_called = True
        raise AssertionError(
            "split_turn must not be called"
        )

    monkeypatch.setattr(
        "voice_dataset.turn_curation.split_turn",
        fail_split,
    )

    before = storage.turns.load()

    try:
        split_and_prepare_turn(
            storage,
            "turn_000001",
            after_region_id="region_000001",
        )
    except ValueError as exc:
        assert (
            "Right split turn does not project "
            "to continuous ASR words"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected split projection failure"
        )

    assert split_called is False
    assert storage.turns.load() == before


def test_merge_and_prepare_turns_requires_reconciliation_before_merge(
    tmp_path,
    monkeypatch,
):
    storage = _storage_with_qwen(tmp_path)

    def remove_reconciliation(record):
        del record["metadata"]["utterance_reconciliation"]
        return record

    storage.update_source(
        "source_001",
        remove_reconciliation,
    )

    for turn_id, start, end in (
        ("turn_000001", 0.9, 1.4),
        ("turn_000002", 1.4, 1.9),
    ):
        storage.turns.append(
            {
                "schema_version": 1,
                "record_type": "turn",
                "id": turn_id,
                "source_id": "source_001",
                "source_start": start,
                "source_end": end,
                "source_regions": [],
                "language": "en",
                "transcript": "text",
                "assignment": {
                    "status": "unknown",
                    "voice_id": None,
                    "method": None,
                },
                "representations": {},
                "embeddings": {},
                "metadata": {},
            }
        )

    merge_called = False

    def fail_merge(*args, **kwargs):
        nonlocal merge_called
        merge_called = True
        raise AssertionError(
            "merge_turns must not be called"
        )

    monkeypatch.setattr(
        "voice_dataset.turn_curation.merge_turns",
        fail_merge,
    )

    before = storage.turns.load()

    with pytest.raises(
        ValueError,
        match="no completed utterance reconciliation",
    ):
        merge_and_prepare_turns(
            storage,
            [
                "turn_000001",
                "turn_000002",
            ],
        )

    assert merge_called is False
    assert storage.turns.load() == before


def test_split_and_prepare_turn_requires_reconciliation_before_split(
    tmp_path,
    monkeypatch,
):
    storage = _storage_with_qwen(tmp_path)

    def remove_reconciliation(record):
        del record["metadata"]["utterance_reconciliation"]
        return record

    storage.update_source(
        "source_001",
        remove_reconciliation,
    )

    for region_id, start, end in (
        ("region_000001", 1.0, 1.4),
        ("region_000002", 1.4, 1.8),
    ):
        storage.regions.append(
            {
                "schema_version": 1,
                "record_type": "region",
                "id": region_id,
                "source_id": "source_001",
                "source_start": start,
                "source_end": end,
            }
        )

    storage.turns.append(
        {
            "schema_version": 1,
            "record_type": "turn",
            "id": "turn_000001",
            "source_id": "source_001",
            "source_start": 1.0,
            "source_end": 1.8,
            "source_regions": [
                "region_000001",
                "region_000002",
            ],
            "language": "en",
            "transcript": "combined text",
            "assignment": {
                "status": "unknown",
                "voice_id": None,
                "method": None,
            },
            "representations": {},
            "embeddings": {},
            "metadata": {},
        }
    )

    split_called = False

    def fail_split(*args, **kwargs):
        nonlocal split_called
        split_called = True
        raise AssertionError(
            "split_turn must not be called"
        )

    monkeypatch.setattr(
        "voice_dataset.turn_curation.split_turn",
        fail_split,
    )

    before = storage.turns.load()

    with pytest.raises(
        ValueError,
        match="no completed utterance reconciliation",
    ):
        split_and_prepare_turn(
            storage,
            "turn_000001",
            after_region_id="region_000001",
        )

    assert split_called is False
    assert storage.turns.load() == before


def test_accept_alignment_recovery_and_prepare_reprepares_source(
    tmp_path,
    monkeypatch,
):
    storage = DatasetStorage(tmp_path)

    storage.add_turn(
        TurnRecord(
            id="turn_000001",
            source_id="source_001",
            source_start=1.0,
            source_end=2.0,
        )
    )

    prepared = []

    def fake_accept(
        storage_arg,
        turn_id,
    ):
        assert storage_arg is storage
        assert turn_id == "turn_000001"

        def update(record):
            record["source_start"] = 1.25
            return record

        return storage.update_turn(
            turn_id,
            update,
        )

    def fake_prepare(
        storage_arg,
        source_id,
    ):
        assert storage_arg is storage
        prepared.append(source_id)

        def update(record):
            record["representations"] = {
                "review": {
                    "path": "prepared.wav",
                },
            }
            return record

        storage.update_turn(
            "turn_000001",
            update,
        )

    monkeypatch.setattr(
        "voice_dataset.turn_curation."
        "accept_alignment_recovery",
        fake_accept,
    )
    monkeypatch.setattr(
        "voice_dataset.turn_curation."
        "prepare_curated_source_turns",
        fake_prepare,
    )

    from voice_dataset.turn_curation import (
        accept_alignment_recovery_and_prepare,
    )

    turn = accept_alignment_recovery_and_prepare(
        storage,
        "turn_000001",
    )

    assert prepared == ["source_001"]
    assert turn["source_start"] == 1.25
    assert turn["representations"]["review"][
        "path"
    ] == "prepared.wav"


def test_accept_edge_recovery_and_prepare_reprepares_source(
    tmp_path,
    monkeypatch,
):
    storage = DatasetStorage(tmp_path)

    storage.add_turn(
        TurnRecord(
            id="turn_000001",
            source_id="source_001",
            source_start=1.0,
            source_end=2.0,
        )
    )

    prepared = []

    def fake_accept(
        storage_arg,
        turn_id,
    ):
        assert storage_arg is storage
        assert turn_id == "turn_000001"

        def update(record):
            record["source_end"] = 2.25
            record["transcript"] = "recovered text"
            return record

        return storage.update_turn(
            turn_id,
            update,
        )

    def fake_prepare(
        storage_arg,
        source_id,
    ):
        assert storage_arg is storage
        prepared.append(source_id)

        def update(record):
            record["embeddings"] = {
                "ecapa": {
                    "path": "prepared.npy",
                },
            }
            return record

        storage.update_turn(
            "turn_000001",
            update,
        )

    monkeypatch.setattr(
        "voice_dataset.turn_curation."
        "accept_edge_recovery",
        fake_accept,
    )
    monkeypatch.setattr(
        "voice_dataset.turn_curation."
        "prepare_curated_source_turns",
        fake_prepare,
    )

    from voice_dataset.turn_curation import (
        accept_edge_recovery_and_prepare,
    )

    turn = accept_edge_recovery_and_prepare(
        storage,
        "turn_000001",
    )

    assert prepared == ["source_001"]
    assert turn["source_end"] == 2.25
    assert turn["transcript"] == "recovered text"
    assert turn["embeddings"]["ecapa"][
        "path"
    ] == "prepared.npy"


def test_trim_and_prepare_turn_updates_end_and_reprepares_source(
    tmp_path,
    monkeypatch,
):
    storage = _storage_with_qwen(tmp_path)

    storage.turns.append(
        {
            "schema_version": 1,
            "record_type": "turn",
            "id": "turn_000001",
            "source_id": "source_001",
            "source_start": 0.9,
            "source_end": 1.9,
            "source_regions": [
                "region_000001",
            ],
            "language": "en",
            "transcript": "Manually reviewed transcript",
            "assignment": {
                "status": "unknown",
                "voice_id": None,
                "method": None,
            },
            "representations": {},
            "embeddings": {},
            "metadata": {},
        }
    )

    prepared = []

    def fake_prepare(
        storage_arg,
        source_id,
        *,
        capture_output=False,
    ):
        assert storage_arg is storage
        assert capture_output is True
        prepared.append(source_id)

    monkeypatch.setattr(
        "voice_dataset.turn_curation."
        "prepare_curated_source_turns",
        fake_prepare,
    )

    from voice_dataset.turn_curation import (
        trim_and_prepare_turn,
    )

    result = trim_and_prepare_turn(
        storage,
        "turn_000001",
        source_end=1.6,
    )

    assert result["source_start"] == 0.9
    assert result["source_end"] == 1.6

    assert (
        result["transcript"]
        == "Manually reviewed transcript"
    )
    assert result["language"] == "en"

    assert result["metadata"]["word_range"] == {
        "start": 0,
        "end": 1,
    }

    assert result["metadata"]["continuous_asr"] == {
        "evidence": "qwen3",
        "word_indices": [0, 1],
    }

    assert prepared == ["source_001"]

    assert result["metadata"][
        "boundary_curation"
    ] == {
        "method": "manual",
    }


def test_trim_and_prepare_turn_rejects_invalid_range_without_mutation(
    tmp_path,
    monkeypatch,
):
    storage = _storage_with_qwen(tmp_path)

    storage.turns.append(
        {
            "schema_version": 1,
            "record_type": "turn",
            "id": "turn_000001",
            "source_id": "source_001",
            "source_start": 0.9,
            "source_end": 1.9,
            "source_regions": [
                "region_000001",
            ],
            "language": "en",
            "transcript": "hello there",
            "assignment": {
                "status": "unknown",
                "voice_id": None,
                "method": None,
            },
            "representations": {},
            "embeddings": {},
            "metadata": {},
        }
    )

    prepared = []

    monkeypatch.setattr(
        "voice_dataset.turn_curation."
        "prepare_curated_source_turns",
        lambda storage_arg, source_id:
            prepared.append(source_id),
    )

    from voice_dataset.turn_curation import (
        trim_and_prepare_turn,
    )

    before = storage.get_turn(
        "turn_000001"
    )

    with pytest.raises(
        ValueError,
        match="source_start must be before source_end",
    ):
        trim_and_prepare_turn(
            storage,
            "turn_000001",
            source_start=1.7,
            source_end=1.6,
        )

    after = storage.get_turn(
        "turn_000001"
    )

    assert after == before
    assert prepared == []


def test_trim_and_prepare_turn_removes_boundary_contamination(
    tmp_path,
    monkeypatch,
):
    storage = _storage_with_qwen(tmp_path)

    storage.turns.append(
        {
            "schema_version": 1,
            "record_type": "turn",
            "id": "turn_000001",
            "source_id": "source_001",
            "source_start": 0.9,
            "source_end": 1.9,
            "source_regions": [
                "region_000001",
            ],
            "language": "en",
            "transcript": "there",
            "assignment": {
                "status": "unknown",
                "voice_id": None,
                "method": None,
            },
            "representations": {},
            "embeddings": {},
            "metadata": {
                "word_range": {
                    "start": 1,
                    "end": 1,
                },
                "continuous_asr": {
                    "evidence": "qwen3",
                    "word_indices": [1],
                },
            },
        }
    )

    prepared = []

    def fake_prepare(
        storage_arg,
        source_id,
        *,
        capture_output=False,
    ):
        assert storage_arg is storage
        assert capture_output is True
        prepared.append(source_id)

    monkeypatch.setattr(
        "voice_dataset.turn_curation."
        "prepare_curated_source_turns",
        fake_prepare,
    )

    result = trim_and_prepare_turn(
        storage,
        "turn_000001",
        source_start=1.4,
    )

    assert result["source_start"] == 1.4

    assert result["metadata"]["word_range"] == {
        "start": 1,
        "end": 1,
    }
    assert result["metadata"]["continuous_asr"] == {
        "evidence": "qwen3",
        "word_indices": [1],
    }

    assert prepared == ["source_001"]


def test_trim_and_prepare_turn_rejects_assigned_word_loss(
    tmp_path,
    monkeypatch,
):
    storage = _storage_with_qwen(tmp_path)

    storage.turns.append(
        {
            "schema_version": 1,
            "record_type": "turn",
            "id": "turn_000001",
            "source_id": "source_001",
            "source_start": 0.9,
            "source_end": 1.9,
            "source_regions": [
                "region_000001",
            ],
            "language": "en",
            "transcript": "hello there",
            "assignment": {
                "status": "unknown",
                "voice_id": None,
                "method": None,
            },
            "representations": {},
            "embeddings": {},
            "metadata": {
                "word_range": {
                    "start": 0,
                    "end": 1,
                },
                "continuous_asr": {
                    "evidence": "qwen3",
                    "word_indices": [0, 1],
                },
            },
        }
    )

    before = storage.get_turn("turn_000001")

    with pytest.raises(
        ValueError,
        match=(
            "Boundary edit would change assigned "
            "continuous ASR words"
        ),
    ):
        trim_and_prepare_turn(
            storage,
            "turn_000001",
            source_end=1.35,
        )

    assert storage.get_turn("turn_000001") == before

def test_trim_and_prepare_turn_rejects_added_words_without_mutation(
    tmp_path,
    monkeypatch,
):
    storage = _storage_with_qwen(tmp_path)

    storage.turns.append(
        {
            "schema_version": 1,
            "record_type": "turn",
            "id": "turn_000001",
            "source_id": "source_001",
            "source_start": 0.9,
            "source_end": 1.9,
            "source_regions": [
                "region_000001",
            ],
            "language": "en",
            "transcript": "hello there",
            "assignment": {
                "status": "unknown",
                "voice_id": None,
                "method": None,
            },
            "representations": {},
            "embeddings": {},
            "metadata": {},
        }
    )

    prepared = []

    monkeypatch.setattr(
        "voice_dataset.turn_curation."
        "prepare_curated_source_turns",
        lambda *args, **kwargs:
            prepared.append("called"),
    )

    before = storage.get_turn("turn_000001")

    with pytest.raises(
        ValueError,
        match=(
            "Boundary edit would change assigned "
            "continuous ASR words"
        ),
    ):
        trim_and_prepare_turn(
            storage,
            "turn_000001",
            source_end=3.6,
        )

    assert storage.get_turn("turn_000001") == before
    assert prepared == []

def test_trim_and_prepare_turn_requires_reconciliation_before_trim(
    tmp_path,
    monkeypatch,
):
    storage = _storage_with_qwen(tmp_path)

    def remove_reconciliation(record):
        del record["metadata"][
            "utterance_reconciliation"
        ]
        return record

    storage.update_source(
        "source_001",
        remove_reconciliation,
    )

    storage.turns.append(
        {
            "schema_version": 1,
            "record_type": "turn",
            "id": "turn_000001",
            "source_id": "source_001",
            "source_start": 0.9,
            "source_end": 1.9,
            "source_regions": [
                "region_000001",
            ],
            "language": "en",
            "transcript": "hello there",
            "assignment": {
                "status": "unknown",
                "voice_id": None,
                "method": None,
            },
            "representations": {},
            "embeddings": {},
            "metadata": {},
        }
    )

    prepared = []

    monkeypatch.setattr(
        "voice_dataset.turn_curation."
        "prepare_curated_source_turns",
        lambda storage_arg, source_id:
            prepared.append(source_id),
    )

    from voice_dataset.turn_curation import (
        trim_and_prepare_turn,
    )

    before = storage.get_turn(
        "turn_000001"
    )

    with pytest.raises(
        ValueError,
        match="no completed utterance reconciliation",
    ):
        trim_and_prepare_turn(
            storage,
            "turn_000001",
            source_end=1.6,
        )

    after = storage.get_turn(
        "turn_000001"
    )

    assert after == before
    assert prepared == []


def test_trim_and_prepare_turn_allows_safe_expansion(
    tmp_path,
    monkeypatch,
):
    storage = _storage_with_qwen(tmp_path)

    storage.turns.append(
        {
            "schema_version": 1,
            "record_type": "turn",
            "id": "turn_000001",
            "source_id": "source_001",
            "source_start": 0.9,
            "source_end": 1.9,
            "source_regions": [
                "region_000001",
            ],
            "language": "en",
            "transcript": "hello there",
            "assignment": {
                "status": "unknown",
                "voice_id": None,
                "method": None,
            },
            "representations": {},
            "embeddings": {},
            "metadata": {},
        }
    )

    storage.turns.append(
        {
            "schema_version": 1,
            "record_type": "turn",
            "id": "turn_000002",
            "source_id": "source_001",
            "source_start": 2.5,
            "source_end": 3.0,
            "source_regions": [],
            "language": "en",
            "transcript": "later",
            "assignment": {
                "status": "unknown",
                "voice_id": None,
                "method": None,
            },
            "representations": {},
            "embeddings": {},
            "metadata": {},
        }
    )

    prepared = []

    def fake_prepare(
        storage_arg,
        source_id,
        *,
        capture_output=False,
    ):
        assert storage_arg is storage
        assert capture_output is True
        prepared.append(source_id)

    monkeypatch.setattr(
        "voice_dataset.turn_curation."
        "prepare_curated_source_turns",
        fake_prepare,
    )

    result = trim_and_prepare_turn(
        storage,
        "turn_000001",
        source_start=0.8,
        source_end=2.0,
    )

    assert result["source_start"] == 0.8
    assert result["source_end"] == 2.0

    assert result["metadata"]["word_range"] == {
        "start": 0,
        "end": 1,
    }

    assert prepared == ["source_001"]


def test_trim_and_prepare_turn_rejects_overlap_with_next_turn(
    tmp_path,
    monkeypatch,
):
    storage = _storage_with_qwen(tmp_path)

    storage.turns.append(
        {
            "schema_version": 1,
            "record_type": "turn",
            "id": "turn_000001",
            "source_id": "source_001",
            "source_start": 0.9,
            "source_end": 1.9,
            "source_regions": [],
            "language": "en",
            "transcript": "hello there",
            "assignment": {
                "status": "unknown",
                "voice_id": None,
                "method": None,
            },
            "representations": {},
            "embeddings": {},
            "metadata": {},
        }
    )

    storage.turns.append(
        {
            "schema_version": 1,
            "record_type": "turn",
            "id": "turn_000002",
            "source_id": "source_001",
            "source_start": 2.0,
            "source_end": 2.5,
            "source_regions": [],
            "language": "en",
            "transcript": "later",
            "assignment": {
                "status": "unknown",
                "voice_id": None,
                "method": None,
            },
            "representations": {},
            "embeddings": {},
            "metadata": {},
        }
    )

    prepared = []

    monkeypatch.setattr(
        "voice_dataset.turn_curation."
        "prepare_curated_source_turns",
        lambda storage_arg, source_id:
            prepared.append(source_id),
    )

    before = storage.get_turn(
        "turn_000001"
    )

    with pytest.raises(
        ValueError,
        match="overlap",
    ):
        trim_and_prepare_turn(
            storage,
            "turn_000001",
            source_end=2.1,
        )

    after = storage.get_turn(
        "turn_000001"
    )

    assert after == before
    assert prepared == []


def test_trim_and_prepare_turn_rejects_overlap_with_previous_turn(
    tmp_path,
    monkeypatch,
):
    storage = _storage_with_qwen(tmp_path)

    storage.turns.append(
        {
            "schema_version": 1,
            "record_type": "turn",
            "id": "turn_000001",
            "source_id": "source_001",
            "source_start": 0.4,
            "source_end": 0.8,
            "source_regions": [],
            "language": "en",
            "transcript": "earlier",
            "assignment": {
                "status": "unknown",
                "voice_id": None,
                "method": None,
            },
            "representations": {},
            "embeddings": {},
            "metadata": {},
        }
    )

    storage.turns.append(
        {
            "schema_version": 1,
            "record_type": "turn",
            "id": "turn_000002",
            "source_id": "source_001",
            "source_start": 0.9,
            "source_end": 1.9,
            "source_regions": [],
            "language": "en",
            "transcript": "hello there",
            "assignment": {
                "status": "unknown",
                "voice_id": None,
                "method": None,
            },
            "representations": {},
            "embeddings": {},
            "metadata": {},
        }
    )

    prepared = []

    monkeypatch.setattr(
        "voice_dataset.turn_curation."
        "prepare_curated_source_turns",
        lambda storage_arg, source_id:
            prepared.append(source_id),
    )

    before = storage.get_turn(
        "turn_000002"
    )

    with pytest.raises(
        ValueError,
        match="overlap",
    ):
        trim_and_prepare_turn(
            storage,
            "turn_000002",
            source_start=0.7,
        )

    after = storage.get_turn(
        "turn_000002"
    )

    assert after == before
    assert prepared == []


def test_prepare_curated_source_turns_can_capture_worker_output(
    tmp_path,
    monkeypatch,
):
    storage = DatasetStorage(tmp_path)

    calls = []

    monkeypatch.setattr(
        "voice_dataset.turn_curation."
        "prepare_source_review_audio",
        lambda storage_arg, source_id:
            calls.append(("review", source_id)),
    )

    def fake_prepare_speaker(
        storage_arg,
        source_id,
        *,
        capture_output=False,
    ):
        calls.append(
            (
                "speaker",
                source_id,
                capture_output,
            )
        )

    monkeypatch.setattr(
        "voice_dataset.turn_curation."
        "prepare_source_speaker_evidence",
        fake_prepare_speaker,
    )

    prepare_curated_source_turns(
        storage,
        "source_001",
        capture_output=True,
    )

    assert calls == [
        ("review", "source_001"),
        (
            "speaker",
            "source_001",
            True,
        ),
    ]


def test_migrate_legacy_ignored_turns(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    storage.turns.append(
        {
            "schema_version": 2,
            "record_type": "turn",
            "id": "turn_legacy_ignore",
            "source_id": "source_001",
            "source_start": 0.0,
            "source_end": 1.0,
            "source_regions": [],
            "language": "en",
            "transcript": "legacy ignored",
            "representations": {},
            "embeddings": {},
            "assignment": {
                "status": "ignore",
                "voice_id": None,
                "method": "manual",
                "confidence": None,
            },
            "review": {
                "status": "reviewed",
                "speaker_calibration": {
                    "schema_version": 1,
                    "confirmed_voice_id": "voice_001",
                },
            },
            "metadata": {},
        }
    )

    storage.turns.append(
        {
            "schema_version": 2,
            "record_type": "turn",
            "id": "turn_normal",
            "source_id": "source_001",
            "source_start": 1.0,
            "source_end": 2.0,
            "source_regions": [],
            "language": "en",
            "transcript": "normal",
            "representations": {},
            "embeddings": {},
            "assignment": {
                "status": "unknown",
                "voice_id": None,
                "method": None,
                "confidence": None,
            },
            "curation": {
                "status": "pending",
            },
            "review": {
                "status": "pending",
            },
            "metadata": {},
        }
    )

    migrated = migrate_legacy_ignored_turns(
        storage
    )

    assert migrated == 1

    legacy = storage.get_turn(
        "turn_legacy_ignore"
    )

    assert legacy is not None

    assert legacy["assignment"] == {
        "status": "unknown",
        "voice_id": None,
        "method": None,
        "confidence": None,
    }

    assert legacy["curation"] == {
        "status": "rejected",
    }

    assert legacy["review"]["status"] == "reviewed"
    assert (
        "speaker_calibration"
        not in legacy["review"]
    )

    normal = storage.get_turn(
        "turn_normal"
    )

    assert normal is not None
    assert normal["curation"] == {
        "status": "pending",
    }


def test_migrate_legacy_ignored_turns_is_idempotent(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    storage.turns.append(
        {
            "schema_version": 2,
            "record_type": "turn",
            "id": "turn_legacy_ignore",
            "source_id": "source_001",
            "source_start": 0.0,
            "source_end": 1.0,
            "source_regions": [],
            "language": None,
            "transcript": None,
            "representations": {},
            "embeddings": {},
            "assignment": {
                "status": "ignore",
                "voice_id": None,
                "method": "manual",
                "confidence": None,
            },
            "review": {
                "status": "pending",
            },
            "metadata": {},
        }
    )

    first = migrate_legacy_ignored_turns(
        storage
    )
    after_first = storage.turns.path.read_bytes()

    second = migrate_legacy_ignored_turns(
        storage
    )
    after_second = storage.turns.path.read_bytes()

    assert first == 1
    assert second == 0
    assert after_second == after_first


def test_project_curated_turn_includes_zero_duration_word(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    storage.add_source(
        SourceRecord(
            id="source_001",
            media_path="/tmp/source.wav",
            metadata={
                "continuous_asr": {
                    "qwen3": {
                        "words": [
                            {
                                "text": "care",
                                "start": 10.0,
                                "end": 10.2,
                            },
                            {
                                "text": "of",
                                "start": 10.2,
                                "end": 10.2,
                            },
                            {
                                "text": "the",
                                "start": 10.2,
                                "end": 10.4,
                            },
                        ],
                        "utterances": [],
                    }
                }
            },
        )
    )

    result = project_curated_turn(
        storage,
        source_id="source_001",
        source_start=9.9,
        source_end=10.5,
    )

    assert (
        result.projection.word_indices
        == (0, 1, 2)
    )
    assert result.transcript == "care of the"
    assert result.start_word_index == 0
    assert result.end_word_index == 2

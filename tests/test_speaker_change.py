import numpy as np
import pytest

from voice_dataset.speaker_change import (
    SpeakerEmbeddingChange,
    SpeakerChangeBoundary,
    cosine_similarity,
    is_suspected_speaker_change,
    speaker_change_window_kind,
    plan_turn_speaker_change_boundaries,
    score_speaker_change,
)


def test_plan_turn_speaker_change_boundaries_uses_non_overlapping_flanks():
    turn = {
        "id": "turn_001",
        "source_start": 10.0,
        "source_end": 13.0,
    }

    boundaries = (
        plan_turn_speaker_change_boundaries(
            turn,
            window=1.0,
            step=0.25,
        )
    )

    assert [
        boundary.boundary
        for boundary in boundaries
    ] == pytest.approx([
        11.0,
        11.25,
        11.5,
        11.75,
        12.0,
    ])

    first = boundaries[0]

    assert first.left_start == pytest.approx(
        10.0
    )
    assert first.left_end == pytest.approx(
        11.0
    )
    assert first.right_start == pytest.approx(
        11.0
    )
    assert first.right_end == pytest.approx(
        12.0
    )


def test_plan_turn_speaker_change_boundaries_skips_short_turn():
    boundaries = (
        plan_turn_speaker_change_boundaries(
            {
                "id": "turn_001",
                "source_start": 10.0,
                "source_end": 11.5,
            },
            window=1.0,
            step=0.25,
        )
    )

    assert boundaries == ()


def test_plan_turn_speaker_change_boundaries_handles_exact_minimum_duration():
    boundaries = (
        plan_turn_speaker_change_boundaries(
            {
                "id": "turn_001",
                "source_start": 10.0,
                "source_end": 12.0,
            },
            window=1.0,
            step=0.25,
        )
    )

    assert len(boundaries) == 1
    assert boundaries[0].boundary == pytest.approx(
        11.0
    )


def test_cosine_similarity_normalizes_inputs():
    similarity = cosine_similarity(
        np.array([2.0, 0.0]),
        np.array([3.0, 0.0]),
    )

    assert similarity == pytest.approx(1.0)


def test_score_speaker_change_combines_encoders():
    boundary = SpeakerChangeBoundary(
        turn_id="turn_001",
        boundary=11.0,
        left_start=10.0,
        left_end=11.0,
        right_start=11.0,
        right_end=12.0,
    )

    change = score_speaker_change(
        boundary,
        ecapa_left=np.array([1.0, 0.0]),
        ecapa_right=np.array([0.0, 1.0]),
        wespeaker_left=np.array([1.0, 0.0]),
        wespeaker_right=np.array([0.8, 0.6]),
    )

    assert change.ecapa_similarity == pytest.approx(
        0.0
    )

    assert (
        change.wespeaker_similarity
        == pytest.approx(0.8)
    )

    assert change.mean_similarity == pytest.approx(
        0.4
    )

    assert (
        change.encoder_disagreement
        == pytest.approx(0.8)
    )

    assert change.divergence == pytest.approx(
        0.6
    )


def test_analyze_speaker_change_boundaries_handles_empty_input():
    from voice_dataset.speaker_change import (
        analyze_speaker_change_boundaries,
    )

    assert (
        analyze_speaker_change_boundaries(
            storage=None,
            source_id="source",
            boundaries=[],
        )
        == ()
    )


def test_adjacent_boundaries_reuse_shared_window_geometry():
    first = SpeakerChangeBoundary(
        turn_id="turn_a",
        boundary=11.0,
        left_start=10.0,
        left_end=11.0,
        right_start=11.0,
        right_end=12.0,
    )

    second = SpeakerChangeBoundary(
        turn_id="turn_b",
        boundary=12.0,
        left_start=11.0,
        left_end=12.0,
        right_start=12.0,
        right_end=13.0,
    )

    assert (
        first.right_start,
        first.right_end,
    ) == (
        second.left_start,
        second.left_end,
    )


def test_speech_aware_planning_rejects_pause_only_side():
    words = [
        {
            "text": "right",
            "start": 11.5,
            "end": 12.0,
        },
    ]

    boundaries = (
        plan_turn_speaker_change_boundaries(
            {
                "id": "turn_001",
                "source_start": 9.0,
                "source_end": 12.0,
            },
            words=words,
            window=1.0,
            step=0.25,
            minimum_window=0.5,
            minimum_speech_ratio=0.5,
        )
    )

    assert boundaries == ()


def test_speech_aware_planning_accepts_short_valid_side():
    words = [
        {
            "text": "left",
            "start": 10.0,
            "end": 11.0,
        },
        {
            "text": "right",
            "start": 11.0,
            "end": 11.64,
        },
    ]

    boundaries = (
        plan_turn_speaker_change_boundaries(
            {
                "id": "turn_001",
                "source_start": 10.0,
                "source_end": 11.64,
            },
            words=words,
            window=1.0,
            step=0.25,
            minimum_window=0.5,
            minimum_speech_ratio=0.5,
        )
    )

    assert len(boundaries) >= 1

    assert (
        boundaries[0].left_end
        - boundaries[0].left_start
    ) >= 0.5

    assert (
        boundaries[0].right_end
        - boundaries[0].right_start
    ) >= 0.5


def test_speech_aware_planning_shifts_away_from_pause():
    words = [
        {
            "text": "speech",
            "start": 10.0,
            "end": 10.8,
        },
        {
            "text": "speech",
            "start": 11.2,
            "end": 12.0,
        },
    ]

    boundaries = (
        plan_turn_speaker_change_boundaries(
            {
                "id": "turn_001",
                "source_start": 9.5,
                "source_end": 12.5,
            },
            words=words,
            window=1.0,
            step=1.0,
            minimum_window=0.5,
            max_shift=0.75,
            shift_step=0.05,
            minimum_speech_ratio=0.5,
        )
    )

    assert boundaries

    boundary = boundaries[0]

    assert boundary.left_start >= 9.5
    assert boundary.right_end <= 12.5


def test_long_turn_uses_only_full_speaker_change_windows():
    words = [
        {
            "text": "speech",
            "start": 10.0,
            "end": 16.0,
        },
    ]

    boundaries = (
        plan_turn_speaker_change_boundaries(
            {
                "id": "turn_001",
                "source_start": 10.0,
                "source_end": 16.0,
            },
            words=words,
            window=1.0,
            step=0.5,
            minimum_window=0.5,
            minimum_speech_ratio=0.5,
        )
    )

    assert boundaries

    assert boundaries[0].boundary == pytest.approx(
        11.0
    )
    assert boundaries[-1].boundary == pytest.approx(
        15.0
    )

    for boundary in boundaries:
        assert (
            boundary.left_end
            - boundary.left_start
        ) == pytest.approx(1.0)

        assert (
            boundary.right_end
            - boundary.right_start
        ) == pytest.approx(1.0)


def test_short_turn_uses_single_adaptive_center_boundary():
    words = [
        {
            "text": "speech",
            "start": 10.0,
            "end": 11.6,
        },
    ]

    boundaries = (
        plan_turn_speaker_change_boundaries(
            {
                "id": "turn_001",
                "source_start": 10.0,
                "source_end": 11.6,
            },
            words=words,
            window=1.0,
            step=0.25,
            minimum_window=0.5,
            minimum_speech_ratio=0.5,
        )
    )

    assert len(boundaries) == 1
    assert boundaries[0].boundary == pytest.approx(
        10.8
    )


def test_speaker_change_window_kind_full():
    boundary = SpeakerChangeBoundary(
        turn_id="turn_001",
        boundary=11.0,
        left_start=10.0,
        left_end=11.0,
        right_start=11.0,
        right_end=12.0,
    )

    assert (
        speaker_change_window_kind(
            boundary
        )
        == "full"
    )


def test_speaker_change_window_kind_adaptive():
    boundary = SpeakerChangeBoundary(
        turn_id="turn_001",
        boundary=11.0,
        left_start=10.0,
        left_end=11.0,
        right_start=11.0,
        right_end=11.7,
    )

    assert (
        speaker_change_window_kind(
            boundary
        )
        == "adaptive"
    )


def test_suspected_speaker_change_uses_full_threshold():
    boundary = SpeakerChangeBoundary(
        turn_id="turn_001",
        boundary=11.0,
        left_start=10.0,
        left_end=11.0,
        right_start=11.0,
        right_end=12.0,
    )

    change = SpeakerEmbeddingChange(
        turn_id="turn_001",
        boundary=11.0,
        ecapa_similarity=0.08,
        wespeaker_similarity=0.06,
        mean_similarity=0.07,
        encoder_disagreement=0.02,
        divergence=0.93,
    )

    assert is_suspected_speaker_change(
        change,
        boundary,
    )


def test_suspected_speaker_change_uses_adaptive_threshold():
    boundary = SpeakerChangeBoundary(
        turn_id="turn_001",
        boundary=11.0,
        left_start=10.0,
        left_end=11.0,
        right_start=11.0,
        right_end=11.7,
    )

    change = SpeakerEmbeddingChange(
        turn_id="turn_001",
        boundary=11.0,
        ecapa_similarity=0.08,
        wespeaker_similarity=0.06,
        mean_similarity=0.07,
        encoder_disagreement=0.02,
        divergence=0.93,
    )

    assert is_suspected_speaker_change(
        change,
        boundary,
    )


def _speaker_change_test_storage(
    tmp_path,
    *,
    automatic_pipeline=None,
):
    from voice_dataset.schema import (
        TurnRecord,
    )
    from voice_dataset.storage import (
        DatasetStorage,
    )

    storage = DatasetStorage(
        tmp_path
    )

    metadata = {}

    if automatic_pipeline is not None:
        metadata["automatic_pipeline"] = (
            automatic_pipeline
        )

    storage.add_turn(
        TurnRecord(
            id="turn_000001",
            source_id="source_001",
            source_start=10.0,
            source_end=12.0,
            metadata=metadata,
        )
    )

    return storage


def test_apply_speaker_change_evidence_adds_review_reason(
    tmp_path,
):
    from voice_dataset.speaker_change import (
        apply_source_speaker_change_evidence,
    )

    storage = _speaker_change_test_storage(
        tmp_path,
        automatic_pipeline={
            "status": "ready",
            "review_reasons": [],
        },
    )

    boundary = SpeakerChangeBoundary(
        turn_id="turn_000001",
        boundary=11.0,
        left_start=10.0,
        left_end=11.0,
        right_start=11.0,
        right_end=12.0,
    )

    change = SpeakerEmbeddingChange(
        turn_id="turn_000001",
        boundary=11.0,
        ecapa_similarity=0.05,
        wespeaker_similarity=0.07,
        mean_similarity=0.06,
        encoder_disagreement=0.02,
        divergence=0.94,
    )

    result = (
        apply_source_speaker_change_evidence(
            storage,
            "source_001",
            [boundary],
            (change,),
            step=0.5,
        )
    )

    turn = storage.get_turn(
        "turn_000001"
    )

    metadata = turn["metadata"]

    evidence = metadata[
        "speaker_change_evidence"
    ]

    assert evidence["status"] == "suspected"
    assert evidence["strongest"][
        "divergence"
    ] == pytest.approx(0.94)

    assert (
        evidence["strongest"][
            "window_kind"
        ]
        == "full"
    )

    assert (
        metadata[
            "automatic_pipeline"
        ]["status"]
        == "review"
    )

    assert (
        metadata[
            "automatic_pipeline"
        ]["review_reasons"]
        == ["suspected_speaker_change"]
    )

    assert result.suspected_turns == 1
    assert result.comparisons == 1


def test_apply_speaker_change_evidence_preserves_other_review_reasons(
    tmp_path,
):
    from voice_dataset.speaker_change import (
        apply_source_speaker_change_evidence,
    )

    storage = _speaker_change_test_storage(
        tmp_path,
        automatic_pipeline={
            "status": "review",
            "review_reasons": [
                "invalid_alignment",
            ],
        },
    )

    boundary = SpeakerChangeBoundary(
        turn_id="turn_000001",
        boundary=11.0,
        left_start=10.0,
        left_end=11.0,
        right_start=11.0,
        right_end=12.0,
    )

    change = SpeakerEmbeddingChange(
        turn_id="turn_000001",
        boundary=11.0,
        ecapa_similarity=0.10,
        wespeaker_similarity=0.12,
        mean_similarity=0.11,
        encoder_disagreement=0.02,
        divergence=0.89,
    )

    apply_source_speaker_change_evidence(
        storage,
        "source_001",
        [boundary],
        (change,),
        step=0.5,
    )

    turn = storage.get_turn(
        "turn_000001"
    )

    pipeline = (
        turn["metadata"]
        ["automatic_pipeline"]
    )

    assert pipeline["status"] == "review"
    assert pipeline["review_reasons"] == [
        "invalid_alignment",
    ]


def test_apply_speaker_change_evidence_removes_stale_reason(
    tmp_path,
):
    from voice_dataset.speaker_change import (
        apply_source_speaker_change_evidence,
    )

    storage = _speaker_change_test_storage(
        tmp_path,
        automatic_pipeline={
            "status": "review",
            "review_reasons": [
                "suspected_speaker_change",
            ],
        },
    )

    boundary = SpeakerChangeBoundary(
        turn_id="turn_000001",
        boundary=11.0,
        left_start=10.0,
        left_end=11.0,
        right_start=11.0,
        right_end=12.0,
    )

    change = SpeakerEmbeddingChange(
        turn_id="turn_000001",
        boundary=11.0,
        ecapa_similarity=0.15,
        wespeaker_similarity=0.15,
        mean_similarity=0.15,
        encoder_disagreement=0.0,
        divergence=0.85,
    )

    apply_source_speaker_change_evidence(
        storage,
        "source_001",
        [boundary],
        (change,),
        step=0.5,
    )

    turn = storage.get_turn(
        "turn_000001"
    )

    pipeline = (
        turn["metadata"]
        ["automatic_pipeline"]
    )

    assert pipeline["status"] == "ready"
    assert pipeline["review_reasons"] == []

    assert (
        turn["metadata"]
        ["speaker_change_evidence"]
        ["status"]
        == "clear"
    )


def test_prepare_speaker_change_evidence_skips_reviewed_turns(
    tmp_path,
    monkeypatch,
):
    from voice_dataset.schema import (
        SourceRecord,
        TurnRecord,
    )
    from voice_dataset.storage import (
        DatasetStorage,
    )
    from voice_dataset.speaker_change import (
        prepare_source_speaker_change_evidence,
    )

    storage = DatasetStorage(tmp_path)

    storage.add_source(
        SourceRecord(
            id="source_001",
            metadata={
                "continuous_asr": {
                    "qwen3": {
                        "words": [
                            {
                                "text": "hello",
                                "start": 10.0,
                                "end": 12.0,
                            },
                        ],
                    },
                },
            },
        )
    )

    storage.add_turn(
        TurnRecord(
            id="turn_000001",
            source_id="source_001",
            source_start=10.0,
            source_end=12.0,
            review={
                "status": "reviewed",
            },
        )
    )

    monkeypatch.setattr(
        "voice_dataset.speaker_change."
        "analyze_speaker_change_boundaries",
        lambda *args, **kwargs: (),
    )

    result = (
        prepare_source_speaker_change_evidence(
            storage,
            "source_001",
        )
    )

    turn = storage.get_turn(
        "turn_000001"
    )

    assert (
        "speaker_change_evidence"
        not in turn["metadata"]
    )

    assert result.analyzed_turns == 0
    assert result.comparisons == 0
    assert result.suspected_turns == 0


def test_prepare_speaker_change_evidence_skips_curated_turns(
    tmp_path,
    monkeypatch,
):
    from voice_dataset.schema import (
        SourceRecord,
        TurnRecord,
        TurnCuration,
    )
    from voice_dataset.storage import (
        DatasetStorage,
    )
    from voice_dataset.speaker_change import (
        prepare_source_speaker_change_evidence,
    )

    storage = DatasetStorage(tmp_path)

    storage.add_source(
        SourceRecord(
            id="source_001",
            metadata={
                "continuous_asr": {
                    "qwen3": {
                        "words": [
                            {
                                "text": "hello",
                                "start": 10.0,
                                "end": 12.0,
                            },
                        ],
                    },
                },
            },
        )
    )

    storage.add_turn(
        TurnRecord(
            id="turn_000001",
            source_id="source_001",
            source_start=10.0,
            source_end=12.0,
            curation=TurnCuration(
                status="accepted",
            ),
        )
    )

    monkeypatch.setattr(
        "voice_dataset.speaker_change."
        "analyze_speaker_change_boundaries",
        lambda *args, **kwargs: (),
    )

    result = (
        prepare_source_speaker_change_evidence(
            storage,
            "source_001",
        )
    )

    turn = storage.get_turn(
        "turn_000001"
    )

    assert (
        "speaker_change_evidence"
        not in turn["metadata"]
    )

    assert result.analyzed_turns == 0
    assert result.comparisons == 0
    assert result.suspected_turns == 0


def test_prepare_speaker_change_evidence_reuses_current_evidence(
    tmp_path,
    monkeypatch,
):
    from voice_dataset.schema import (
        SourceRecord,
        TurnRecord,
    )
    from voice_dataset.speaker_change import (
        SpeakerEmbeddingChange,
        apply_source_speaker_change_evidence,
        prepare_source_speaker_change_evidence,
        plan_turn_speaker_change_boundaries,
    )
    from voice_dataset.storage import (
        DatasetStorage,
    )

    storage = DatasetStorage(tmp_path)

    words = [
        {
            "text": "hello",
            "start": 10.0,
            "end": 11.0,
        },
        {
            "text": "there",
            "start": 11.0,
            "end": 12.0,
        },
    ]

    storage.add_source(
        SourceRecord(
            id="source_001",
            metadata={
                "continuous_asr": {
                    "qwen3": {
                        "words": words,
                    },
                },
            },
        )
    )

    storage.add_turn(
        TurnRecord(
            id="turn_000001",
            source_id="source_001",
            source_start=10.0,
            source_end=12.0,
        )
    )

    turn = storage.get_turn(
        "turn_000001"
    )

    boundaries = list(
        plan_turn_speaker_change_boundaries(
            turn,
            words=words,
            step=0.5,
        )
    )

    assert len(boundaries) == 1

    boundary = boundaries[0]

    apply_source_speaker_change_evidence(
        storage,
        "source_001",
        boundaries,
        (
            SpeakerEmbeddingChange(
                turn_id="turn_000001",
                boundary=boundary.boundary,
                ecapa_similarity=0.2,
                wespeaker_similarity=0.2,
                mean_similarity=0.2,
                encoder_disagreement=0.0,
                divergence=0.8,
            ),
        ),
        step=0.5,
    )

    def fail_analysis(*args, **kwargs):
        raise AssertionError(
            "current evidence must skip GPU analysis"
        )

    monkeypatch.setattr(
        "voice_dataset.speaker_change."
        "analyze_speaker_change_boundaries",
        fail_analysis,
    )

    result = (
        prepare_source_speaker_change_evidence(
            storage,
            "source_001",
            step=0.5,
        )
    )

    assert result.analyzed_turns == 1
    assert result.comparisons == 1
    assert result.suspected_turns == 0


def test_prepare_speaker_change_evidence_reruns_when_step_changes(
    tmp_path,
    monkeypatch,
):
    from voice_dataset.schema import (
        SourceRecord,
        TurnRecord,
    )
    from voice_dataset.speaker_change import (
        SpeakerEmbeddingChange,
        apply_source_speaker_change_evidence,
        prepare_source_speaker_change_evidence,
        plan_turn_speaker_change_boundaries,
    )
    from voice_dataset.storage import (
        DatasetStorage,
    )

    storage = DatasetStorage(tmp_path)

    words = [
        {
            "text": "one",
            "start": 10.0,
            "end": 11.0,
        },
        {
            "text": "two",
            "start": 11.0,
            "end": 12.0,
        },
        {
            "text": "three",
            "start": 12.0,
            "end": 13.0,
        },
        {
            "text": "four",
            "start": 13.0,
            "end": 14.0,
        },
    ]

    storage.add_source(
        SourceRecord(
            id="source_001",
            metadata={
                "continuous_asr": {
                    "qwen3": {
                        "words": words,
                    },
                },
            },
        )
    )

    storage.add_turn(
        TurnRecord(
            id="turn_000001",
            source_id="source_001",
            source_start=10.0,
            source_end=14.0,
        )
    )

    turn = storage.get_turn(
        "turn_000001"
    )

    old_boundaries = list(
        plan_turn_speaker_change_boundaries(
            turn,
            words=words,
            step=0.5,
        )
    )

    old_changes = tuple(
        SpeakerEmbeddingChange(
            turn_id=boundary.turn_id,
            boundary=boundary.boundary,
            ecapa_similarity=0.2,
            wespeaker_similarity=0.2,
            mean_similarity=0.2,
            encoder_disagreement=0.0,
            divergence=0.8,
        )
        for boundary in old_boundaries
    )

    apply_source_speaker_change_evidence(
        storage,
        "source_001",
        old_boundaries,
        old_changes,
        step=0.5,
    )

    calls = []

    def fake_analysis(
        storage_arg,
        source_id,
        boundaries,
        *,
        capture_output=True,
    ):
        calls.append(
            list(boundaries)
        )

        return tuple(
            SpeakerEmbeddingChange(
                turn_id=boundary.turn_id,
                boundary=boundary.boundary,
                ecapa_similarity=0.2,
                wespeaker_similarity=0.2,
                mean_similarity=0.2,
                encoder_disagreement=0.0,
                divergence=0.8,
            )
            for boundary in boundaries
        )

    monkeypatch.setattr(
        "voice_dataset.speaker_change."
        "analyze_speaker_change_boundaries",
        fake_analysis,
    )

    prepare_source_speaker_change_evidence(
        storage,
        "source_001",
        step=0.25,
    )

    assert len(calls) == 1
    assert calls[0]

    turn = storage.get_turn(
        "turn_000001"
    )

    assert (
        turn["metadata"]
        ["speaker_change_evidence"]
        ["step"]
        == 0.25
    )

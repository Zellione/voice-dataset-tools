import pytest
import numpy as np
from voice_dataset.reviewer_session import ReviewerSession
from voice_dataset.storage import DatasetStorage
from voice_dataset.schema import TurnRecord
from voice_dataset.voices import create_voice
from voice_dataset.speaker_candidates import (
    EmbeddingVoiceCandidate,
    SpeakerCandidate,
)
from voice_dataset.speaker_similarity import (
    VoiceTurnMatch,
)


def add_turn(
    storage: DatasetStorage,
    turn_id: str,
    *,
    source_id: str = "source_001",
    source_start: float,
    auto_status: str = "accepted",
) -> None:
    storage.add_turn(
        TurnRecord(
            id=turn_id,
            source_id=source_id,
            source_start=source_start,
            source_end=source_start + 1.0,
        )
    )

    def update(record):
        record["transcript"] = f"text {turn_id}"
        record["metadata"] = {
            "automatic_pipeline": {
                "status": auto_status,
            },
        }
        return record

    storage.update_turn(turn_id, update)


def test_session_sorts_and_navigates_turns(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000002",
        source_start=2.0,
    )
    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )
    add_turn(
        storage,
        "turn_000003",
        source_start=3.0,
    )

    session = ReviewerSession(storage)

    assert session.turn_ids == (
        "turn_000001",
        "turn_000002",
        "turn_000003",
    )
    assert session.total == 3
    assert session.position == 1
    assert session.current_turn_id == "turn_000001"

    assert session.previous()["id"] == "turn_000001"
    assert session.position == 1

    assert session.next()["id"] == "turn_000002"
    assert session.position == 2

    assert session.next()["id"] == "turn_000003"
    assert session.position == 3

    assert session.next()["id"] == "turn_000003"
    assert session.position == 3


def test_session_filters_by_source(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_id="source_001",
        source_start=1.0,
    )
    add_turn(
        storage,
        "turn_000002",
        source_id="source_002",
        source_start=2.0,
    )

    session = ReviewerSession(
        storage,
        source_id="source_002",
    )

    assert session.turn_ids == ("turn_000002",)
    assert session.current()["id"] == "turn_000002"


def test_session_filters_auto_review_turns(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
        auto_status="accepted",
    )
    add_turn(
        storage,
        "turn_000002",
        source_start=2.0,
        auto_status="review",
    )

    session = ReviewerSession(
        storage,
        auto_review_only=True,
    )

    assert session.turn_ids == ("turn_000002",)
    assert session.current()["id"] == "turn_000002"


def test_current_reads_fresh_turn_from_storage(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    session = ReviewerSession(storage)

    storage.update_turn(
        "turn_000001",
        lambda turn: {
            **turn,
            "transcript": "changed externally",
        },
    )

    assert (
        session.current()["transcript"]
        == "changed externally"
    )


def test_empty_session_has_no_current_turn(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    session = ReviewerSession(storage)

    assert session.turn_ids == ()
    assert session.total == 0
    assert session.position is None
    assert session.current_turn_id is None
    assert session.current() is None
    assert session.next() is None
    assert session.previous() is None


def test_session_edits_current_turn(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )
    add_turn(
        storage,
        "turn_000002",
        source_start=2.0,
    )

    session = ReviewerSession(storage)
    session.next()

    result = session.edit_transcript(
        "corrected transcript"
    )

    assert result["id"] == "turn_000002"
    assert result["transcript"] == "corrected transcript"

    stored = storage.get_turn("turn_000002")
    assert stored["transcript"] == "corrected transcript"

    untouched = storage.get_turn("turn_000001")
    assert untouched["transcript"] == "text turn_000001"


def test_session_sets_language_on_current_turn(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    session = ReviewerSession(storage)

    result = session.set_language("en")

    assert result["language"] == "en"
    assert session.current()["language"] == "en"


def test_session_updates_review_status(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    session = ReviewerSession(storage)

    result = session.mark_reviewed()

    assert result["review"]["status"] == "reviewed"

    result = session.mark_pending()

    assert result["review"]["status"] == "pending"


def test_session_persists_speaker_calibration_when_reviewed(
    tmp_path,
    monkeypatch,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_id="episode_01",
        source_start=1.0,
    )

    voice = create_voice(
        storage,
        character="Test Character",
        language="en",
    )

    session = ReviewerSession(
        storage,
        embedding_names=(
            "ecapa_speaker",
            "wespeaker_speaker",
        ),
    )

    session.assign_voice(
        voice["id"]
    )

    candidates = [
        SpeakerCandidate(
            voice_id=voice["id"],
            score=0.81,
            encoder_count=2,
            embedding_scores=(
                EmbeddingVoiceCandidate(
                    voice_id=voice["id"],
                    embedding_name="ecapa_speaker",
                    score=0.78,
                    support=2,
                    matches=(
                        VoiceTurnMatch(
                            turn_id="turn_ref_001",
                            similarity=0.90,
                            source_id="episode_02",
                        ),
                    ),
                ),
                EmbeddingVoiceCandidate(
                    voice_id=voice["id"],
                    embedding_name="wespeaker_speaker",
                    score=0.84,
                    support=2,
                    matches=(
                        VoiceTurnMatch(
                            turn_id="turn_ref_002",
                            similarity=0.91,
                            source_id="episode_03",
                        ),
                    ),
                ),
            ),
        ),
        SpeakerCandidate(
            voice_id="voice_other",
            score=0.43,
            encoder_count=2,
            embedding_scores=(),
        ),
    ]

    monkeypatch.setattr(
        session,
        "speaker_candidates",
        lambda **kwargs: candidates,
    )

    result = session.mark_reviewed()

    assert result["review"]["status"] == "reviewed"

    calibration = result["review"][
        "speaker_calibration"
    ]

    assert calibration["schema_version"] == 1
    assert calibration["turn_id"] == "turn_000001"
    assert calibration["source_id"] == "episode_01"

    assert (
        calibration["confirmed_voice_id"]
        == voice["id"]
    )
    assert (
        calibration["predicted_voice_id"]
        == voice["id"]
    )

    assert calibration["correct"] is True

    assert calibration[
        "top_score"
    ] == pytest.approx(0.81)

    assert calibration[
        "runner_up_score"
    ] == pytest.approx(0.43)

    assert calibration[
        "margin"
    ] == pytest.approx(0.38)

    assert calibration["encoder_count"] == 2

    assert (
        calibration["total_source_support"]
        == 2
    )


def test_session_pending_clears_speaker_calibration(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    def add_calibration(record):
        review = dict(
            record.get("review") or {}
        )

        review["speaker_calibration"] = {
            "schema_version": 1,
        }

        record["review"] = review
        return record

    storage.update_turn(
        "turn_000001",
        add_calibration,
    )

    session = ReviewerSession(storage)

    result = session.mark_pending()

    assert result["review"]["status"] == "pending"

    assert (
        "speaker_calibration"
        not in result["review"]
    )


def test_session_updates_assignment_status(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    session = ReviewerSession(storage)

    result = session.mark_unknown()

    assert result["assignment"]["status"] == "unknown"


def test_session_rejects_turn_without_changing_assignment(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    voice = create_voice(
        storage,
        character="Test Character",
        language="en",
    )

    session = ReviewerSession(storage)

    session.assign_voice(
        voice["id"]
    )

    result = session.reject()

    assert result["curation"] == {
        "status": "rejected",
    }

    assert result["assignment"] == {
        "status": "assigned",
        "voice_id": voice["id"],
        "method": "manual",
        "confidence": None,
    }


def test_session_assigns_voice_to_current_turn(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    voice = create_voice(
        storage,
        character="Test Character",
        language="en",
    )

    session = ReviewerSession(storage)

    result = session.assign_voice(voice["id"])

    assert result["assignment"] == {
        "status": "assigned",
        "voice_id": voice["id"],
        "method": "manual",
        "confidence": None,
    }

    assert (
        session.current()["assignment"]["voice_id"]
        == voice["id"]
    )


def test_session_updates_boundary_status(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    def add_boundary_evidence(record):
        metadata = dict(
            record.get("metadata") or {}
        )
        metadata["boundary_evidence"] = {
            "representation": "center",
            "margin": 0.1,
            "near_source_start": True,
            "near_source_end": False,
        }
        record["metadata"] = metadata
        return record

    storage.update_turn(
        "turn_000001",
        add_boundary_evidence,
    )

    session = ReviewerSession(storage)

    result = session.mark_boundary_clipped()

    assert (
        result["review"]["boundary"]["status"]
        == "clipped"
    )

    result = session.mark_boundary_complete()

    assert (
        result["review"]["boundary"]["status"]
        == "complete"
    )


def test_session_merge_with_next_refreshes_and_reanchors(
    tmp_path,
    monkeypatch,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )
    add_turn(
        storage,
        "turn_000002",
        source_start=2.0,
    )
    add_turn(
        storage,
        "turn_000003",
        source_start=3.0,
    )

    session = ReviewerSession(storage)

    def fake_merge(
        actual_storage,
        turn_ids,
    ):
        assert actual_storage is storage
        assert turn_ids == [
            "turn_000001",
            "turn_000002",
        ]

        storage.turns.replace([
            turn
            for turn in storage.turns.load()
            if turn["id"] != "turn_000002"
        ])

        merged = storage.get_turn("turn_000001")
        assert merged is not None
        return merged

    monkeypatch.setattr(
        "voice_dataset.reviewer_session."
        "merge_and_prepare_turns",
        fake_merge,
    )

    result = session.merge_with_next()

    assert result["id"] == "turn_000001"
    assert session.turn_ids == (
        "turn_000001",
        "turn_000003",
    )
    assert session.current_turn_id == "turn_000001"
    assert session.position == 1


def test_session_split_refreshes_and_reanchors_left(
    tmp_path,
    monkeypatch,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )
    add_turn(
        storage,
        "turn_000002",
        source_start=3.0,
    )

    session = ReviewerSession(storage)

    def fake_split(
        actual_storage,
        turn_id,
        *,
        after_region_id,
    ):
        assert actual_storage is storage
        assert turn_id == "turn_000001"
        assert after_region_id == "region_000001"

        def shorten_left(record):
            record["source_end"] = 1.5
            return record

        storage.update_turn(
            "turn_000001",
            shorten_left,
        )

        add_turn(
            storage,
            "turn_000003",
            source_start=1.5,
        )

        left = storage.get_turn("turn_000001")
        right = storage.get_turn("turn_000003")

        assert left is not None
        assert right is not None

        return left, right

    monkeypatch.setattr(
        "voice_dataset.reviewer_session."
        "split_and_prepare_turn",
        fake_split,
    )

    left, right = session.split(
        after_region_id="region_000001",
    )

    assert left["id"] == "turn_000001"
    assert right["id"] == "turn_000003"

    assert session.turn_ids == (
        "turn_000001",
        "turn_000003",
        "turn_000002",
    )
    assert session.current_turn_id == left["id"]
    assert session.position == 1


def test_session_merge_with_next_rejects_last_turn(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    session = ReviewerSession(storage)

    with pytest.raises(
        ValueError,
        match="Current turn has no next turn",
    ):
        session.merge_with_next()


def test_refresh_moves_to_next_filtered_turn_when_current_disappears(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
        auto_status="review",
    )
    add_turn(
        storage,
        "turn_000002",
        source_start=2.0,
        auto_status="review",
    )
    add_turn(
        storage,
        "turn_000003",
        source_start=3.0,
        auto_status="review",
    )

    session = ReviewerSession(
        storage,
        auto_review_only=True,
    )

    session.next()

    assert session.current_turn_id == "turn_000002"

    def update(record):
        record["metadata"]["automatic_pipeline"][
            "status"
        ] = "accepted"
        return record

    storage.update_turn(
        "turn_000002",
        update,
    )

    session.refresh()

    assert session.turn_ids == (
        "turn_000001",
        "turn_000003",
    )
    assert session.current_turn_id == "turn_000003"
    assert session.position == 2


def test_refresh_moves_to_previous_filtered_turn_when_last_disappears(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
        auto_status="review",
    )
    add_turn(
        storage,
        "turn_000002",
        source_start=2.0,
        auto_status="review",
    )

    session = ReviewerSession(
        storage,
        auto_review_only=True,
    )

    session.next()

    assert session.current_turn_id == "turn_000002"

    def update(record):
        record["metadata"]["automatic_pipeline"][
            "status"
        ] = "accepted"
        return record

    storage.update_turn(
        "turn_000002",
        update,
    )

    session.refresh()

    assert session.turn_ids == (
        "turn_000001",
    )
    assert session.current_turn_id == "turn_000001"
    assert session.position == 1


def test_session_accept_alignment_recovery_uses_prepared_operation(
    tmp_path,
    monkeypatch,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )
    add_turn(
        storage,
        "turn_000002",
        source_start=2.0,
    )

    session = ReviewerSession(storage)

    session.next()

    calls = []

    def fake_accept(
        storage_arg,
        turn_id,
    ):
        assert storage_arg is storage
        calls.append(turn_id)

        return storage.get_turn(turn_id)

    monkeypatch.setattr(
        "voice_dataset.reviewer_session."
        "accept_alignment_recovery_and_prepare",
        fake_accept,
    )

    updated = session.accept_alignment_recovery()

    assert calls == ["turn_000002"]
    assert updated["id"] == "turn_000002"
    assert session.current_turn_id == "turn_000002"


def test_session_accept_edge_recovery_uses_prepared_operation(
    tmp_path,
    monkeypatch,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )
    add_turn(
        storage,
        "turn_000002",
        source_start=2.0,
    )

    session = ReviewerSession(storage)

    session.next()

    calls = []

    def fake_accept(
        storage_arg,
        turn_id,
    ):
        assert storage_arg is storage
        calls.append(turn_id)

        return storage.get_turn(turn_id)

    monkeypatch.setattr(
        "voice_dataset.reviewer_session."
        "accept_edge_recovery_and_prepare",
        fake_accept,
    )

    updated = session.accept_edge_recovery()

    assert calls == ["turn_000002"]
    assert updated["id"] == "turn_000002"
    assert session.current_turn_id == "turn_000002"


def test_merge_with_next_uses_canonical_next_turn_when_filtered(
    tmp_path,
    monkeypatch,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
        auto_status="review",
    )
    add_turn(
        storage,
        "turn_000002",
        source_start=2.0,
        auto_status="accepted",
    )
    add_turn(
        storage,
        "turn_000003",
        source_start=3.0,
        auto_status="review",
    )

    session = ReviewerSession(
        storage,
        auto_review_only=True,
    )

    assert session.turn_ids == (
        "turn_000001",
        "turn_000003",
    )
    assert session.current_turn_id == "turn_000001"

    merged_ids = []

    def fake_merge(
        storage_arg,
        turn_ids,
    ):
        assert storage_arg is storage
        merged_ids.extend(turn_ids)

        return storage.get_turn(turn_ids[0])

    monkeypatch.setattr(
        "voice_dataset.reviewer_session."
        "merge_and_prepare_turns",
        fake_merge,
    )

    session.merge_with_next()

    assert merged_ids == [
        "turn_000001",
        "turn_000002",
    ]


def test_merge_with_next_does_not_cross_sources(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_id="source_001",
        source_start=1.0,
    )
    add_turn(
        storage,
        "turn_000002",
        source_id="source_002",
        source_start=1.0,
    )

    session = ReviewerSession(storage)

    assert session.current_turn_id == "turn_000001"

    with pytest.raises(
        ValueError,
        match="Current turn has no next turn",
    ):
        session.merge_with_next()


def test_session_speaker_candidates_combines_available_embeddings(
    tmp_path,
    monkeypatch,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    def add_embeddings(record):
        record["embeddings"] = {
            "ecapa_speaker": {},
            "wespeaker_speaker": {},
        }
        return record

    storage.update_turn(
        "turn_000001",
        add_embeddings,
    )

    session = ReviewerSession(storage)

    rank_calls = []

    def fake_rank(
        storage_arg,
        turn_id,
        embedding_name,
    ):
        assert storage_arg is storage
        assert turn_id == "turn_000001"
        rank_calls.append(embedding_name)

        return []

    def fake_aggregate(
        matches,
        *,
        embedding_name,
    ):
        assert matches == []

        score = {
            "ecapa_speaker": 0.80,
            "wespeaker_speaker": 0.70,
        }[embedding_name]

        return [
            EmbeddingVoiceCandidate(
                voice_id="voice_001",
                embedding_name=embedding_name,
                score=score,
                support=2,
                matches=(),
            )
        ]

    def fake_combine(*groups):
        assert len(groups) == 2

        return [
            SpeakerCandidate(
                voice_id="voice_001",
                score=0.75,
                encoder_count=2,
                embedding_scores=tuple(
                    group[0]
                    for group in groups
                ),
            )
        ]

    monkeypatch.setattr(
        "voice_dataset.reviewer_session."
        "rank_voice_matches",
        fake_rank,
    )
    monkeypatch.setattr(
        "voice_dataset.reviewer_session."
        "aggregate_voice_matches",
        fake_aggregate,
    )
    monkeypatch.setattr(
        "voice_dataset.reviewer_session."
        "combine_embedding_candidates",
        fake_combine,
    )

    candidates = session.speaker_candidates(
        embedding_names=(
            "ecapa_speaker",
            "wespeaker_speaker",
        ),
    )

    assert rank_calls == [
        "ecapa_speaker",
        "wespeaker_speaker",
    ]
    assert len(candidates) == 1
    assert candidates[0].voice_id == "voice_001"
    assert candidates[0].score == pytest.approx(0.75)


def test_session_speaker_candidates_skips_missing_embedding(
    tmp_path,
    monkeypatch,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    def add_embeddings(record):
        record["embeddings"] = {
            "ecapa_speaker": {},
        }
        return record

    storage.update_turn(
        "turn_000001",
        add_embeddings,
    )

    session = ReviewerSession(storage)

    rank_calls = []

    def fake_rank(
        storage_arg,
        turn_id,
        embedding_name,
    ):
        rank_calls.append(embedding_name)
        return []

    monkeypatch.setattr(
        "voice_dataset.reviewer_session."
        "rank_voice_matches",
        fake_rank,
    )

    candidates = session.speaker_candidates(
        embedding_names=(
            "ecapa_speaker",
            "wespeaker_speaker",
        ),
    )

    assert rank_calls == [
        "ecapa_speaker",
    ]
    assert candidates == []


def test_session_speaker_candidates_applies_limit(
    tmp_path,
    monkeypatch,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    def add_embeddings(record):
        record["embeddings"] = {
            "ecapa_speaker": {},
        }
        return record

    storage.update_turn(
        "turn_000001",
        add_embeddings,
    )

    session = ReviewerSession(storage)

    combined = [
        SpeakerCandidate(
            voice_id=f"voice_{index:03d}",
            score=1.0 - index / 10,
            encoder_count=1,
            embedding_scores=(),
        )
        for index in range(1, 6)
    ]

    monkeypatch.setattr(
        "voice_dataset.reviewer_session."
        "rank_voice_matches",
        lambda *args: [],
    )
    monkeypatch.setattr(
        "voice_dataset.reviewer_session."
        "aggregate_voice_matches",
        lambda matches, *, embedding_name: [],
    )
    monkeypatch.setattr(
        "voice_dataset.reviewer_session."
        "combine_embedding_candidates",
        lambda *groups: combined,
    )

    candidates = session.speaker_candidates(
        embedding_names=("ecapa_speaker",),
        limit=3,
    )

    assert [
        candidate.voice_id
        for candidate in candidates
    ] == [
        "voice_001",
        "voice_002",
        "voice_003",
    ]


def test_session_speaker_candidates_rejects_invalid_limit(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    session = ReviewerSession(storage)

    with pytest.raises(
        ValueError,
        match="limit must be positive",
    ):
        session.speaker_candidates(
            embedding_names=("ecapa_speaker",),
            limit=0,
        )


def test_session_speaker_candidates_integrates_real_embeddings(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    voice_1 = create_voice(
        storage,
        character="Voice One",
        language="en",
    )
    voice_2 = create_voice(
        storage,
        character="Voice Two",
        language="en",
    )

    add_turn(
        storage,
        "turn_query",
        source_start=1.0,
    )
    add_turn(
        storage,
        "turn_voice_1",
        source_start=2.0,
    )
    add_turn(
        storage,
        "turn_voice_2",
        source_start=3.0,
    )

    def add_embedding(
        turn_id,
        embedding_name,
        vector,
    ):
        relative_path = (
            f"turns/{turn_id}/embeddings/"
            f"{embedding_name}.npy"
        )

        path = storage.root / relative_path
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(path, vector)

        def update(record):
            embeddings = dict(
                record.get("embeddings") or {}
            )
            embeddings[embedding_name] = {
                "encoder": f"test-{embedding_name}",
                "representation": "speech",
                "path": relative_path,
                "dimension": int(vector.shape[0]),
                "metadata": {
                    "model": f"test-{embedding_name}",
                },
            }
            record["embeddings"] = embeddings
            return record

        storage.update_turn(
            turn_id,
            update,
        )

    query = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )
    same = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )
    different = np.array(
        [0.0, 1.0],
        dtype=np.float32,
    )

    for embedding_name in (
        "ecapa_speaker",
        "wespeaker_speaker",
    ):
        add_embedding(
            "turn_query",
            embedding_name,
            query,
        )
        add_embedding(
            "turn_voice_1",
            embedding_name,
            same,
        )
        add_embedding(
            "turn_voice_2",
            embedding_name,
            different,
        )

    def assign(
        turn_id,
        voice_id,
    ):
        def update(record):
            record["assignment"] = {
                "status": "assigned",
                "voice_id": voice_id,
                "method": "manual",
                "confidence": None,
            }
            return record

        storage.update_turn(
            turn_id,
            update,
        )

    assign(
        "turn_voice_1",
        voice_1["id"],
    )
    assign(
        "turn_voice_2",
        voice_2["id"],
    )

    session = ReviewerSession(storage)

    assert session.current_turn_id == "turn_query"

    candidates = session.speaker_candidates(
        embedding_names=(
            "ecapa_speaker",
            "wespeaker_speaker",
        ),
    )

    assert [
        candidate.voice_id
        for candidate in candidates
    ] == [
        voice_1["id"],
        voice_2["id"],
    ]

    assert candidates[0].score == pytest.approx(1.0)
    assert candidates[0].encoder_count == 2

    assert [
        score.embedding_name
        for score in candidates[0].embedding_scores
    ] == [
        "ecapa_speaker",
        "wespeaker_speaker",
    ]

    assert [
        score.score
        for score in candidates[0].embedding_scores
    ] == pytest.approx([
        1.0,
        1.0,
    ])

    assert [
        score.support
        for score in candidates[0].embedding_scores
    ] == [
        1,
        1,
    ]

    assert candidates[1].score == pytest.approx(0.0)
    assert candidates[1].encoder_count == 2


def test_session_current_view_builds_view(
    tmp_path,
    monkeypatch,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    session = ReviewerSession(storage)

    monkeypatch.setattr(
        session,
        "speaker_candidates",
        lambda **kwargs: [],
    )

    view = session.current_view(
        embedding_names=(
            "ecapa_speaker",
            "wespeaker_speaker",
        ),
    )

    assert view is not None
    assert view.turn_id == "turn_000001"
    assert view.position == 1
    assert view.total == 1


def test_session_current_view_returns_none_when_empty(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)
    session = ReviewerSession(storage)

    view = session.current_view(
        embedding_names=(
            "ecapa_speaker",
            "wespeaker_speaker",
        ),
    )

    assert view is None


def test_session_create_and_assign_voice(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    session = ReviewerSession(storage)

    voice = session.create_and_assign_voice(
        character="Vander",
        language="en",
    )

    assert voice["character"] == "Vander"
    assert voice["language"] == "en"

    persisted_voice = storage.get_voice(
        voice["id"]
    )

    assert persisted_voice is not None

    turn = storage.get_turn("turn_000001")

    assert turn is not None
    assert turn["assignment"] == {
        "status": "assigned",
        "voice_id": voice["id"],
        "method": "manual",
        "confidence": None,
    }


def test_session_create_and_assign_voice_requires_current_turn(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)
    session = ReviewerSession(storage)

    with pytest.raises(
        ValueError,
        match="Reviewer session has no current turn",
    ):
        session.create_and_assign_voice(
            character="Vander",
            language="en",
        )

    assert storage.voices.load() == []


def test_session_trim_uses_prepared_operation_and_keeps_current_turn(
    tmp_path,
    monkeypatch,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )
    add_turn(
        storage,
        "turn_000002",
        source_start=2.0,
    )

    session = ReviewerSession(storage)

    calls = []

    def fake_trim(
        storage_arg,
        turn_id,
        *,
        source_start=None,
        source_end=None,
    ):
        assert storage_arg is storage
        assert turn_id == "turn_000001"

        calls.append(
            (
                source_start,
                source_end,
            )
        )

        def update(record):
            record["source_end"] = source_end
            return record

        return storage.update_turn(
            turn_id,
            update,
        )

    monkeypatch.setattr(
        "voice_dataset.reviewer_session."
        "trim_and_prepare_turn",
        fake_trim,
    )

    updated = session.trim(
        source_end=1.75,
    )

    assert calls == [
        (
            None,
            1.75,
        )
    ]

    assert updated["source_end"] == 1.75

    assert (
        session.current_turn_id
        == "turn_000001"
    )
    assert session.position == 1


def test_session_trim_requires_current_turn(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)
    session = ReviewerSession(storage)

    with pytest.raises(
        ValueError,
        match="Reviewer session has no current turn",
    ):
        session.trim(
            source_end=1.0,
        )


def test_turn_rejection_preserves_speaker_calibration(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    voice = create_voice(
        storage,
        character="Voice One",
        language="en",
    )

    def add_existing_calibration(record):
        record["assignment"] = {
            "status": "assigned",
            "voice_id": voice["id"],
            "method": "manual",
            "confidence": None,
        }

        review = dict(
            record.get("review") or {}
        )
        review["speaker_calibration"] = {
            "schema_version": 1,
            "confirmed_voice_id": voice["id"],
        }

        record["review"] = review
        return record

    storage.update_turn(
        "turn_000001",
        add_existing_calibration,
    )

    session = ReviewerSession(storage)

    result = session.reject()

    assert result["curation"]["status"] == "rejected"

    assert (
        result["review"]["speaker_calibration"][
            "confirmed_voice_id"
        ]
        == voice["id"]
    )


@pytest.mark.parametrize(
    "action",
    [
        "assign",
        "unknown",
    ],
)
def test_assignment_change_invalidates_speaker_calibration(
    tmp_path,
    action,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    voice = create_voice(
        storage,
        character="Voice One",
        language="en",
    )

    other_voice = create_voice(
        storage,
        character="Voice Two",
        language="en",
    )

    def add_existing_calibration(record):
        record["assignment"] = {
            "status": "assigned",
            "voice_id": voice["id"],
            "method": "manual",
            "confidence": None,
        }

        review = dict(
            record.get("review") or {}
        )
        review["status"] = "reviewed"
        review["speaker_calibration"] = {
            "schema_version": 1,
            "confirmed_voice_id": voice["id"],
        }

        record["review"] = review
        return record

    storage.update_turn(
        "turn_000001",
        add_existing_calibration,
    )

    session = ReviewerSession(storage)

    if action == "assign":
        result = session.assign_voice(
            other_voice["id"]
        )
    else:
        result = session.mark_unknown()

    assert (
        "speaker_calibration"
        not in result["review"]
    )


def test_mark_reviewed_accepts_pending_turn(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    session = ReviewerSession(storage)

    result = session.mark_reviewed()

    assert result["review"]["status"] == "reviewed"
    assert result["curation"] == {
        "status": "accepted",
    }


def test_mark_reviewed_preserves_rejected_turn(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    session = ReviewerSession(storage)

    rejected = session.reject()

    assert rejected["curation"] == {
        "status": "rejected",
    }

    result = session.mark_reviewed()

    assert result["review"]["status"] == "reviewed"
    assert result["curation"] == {
        "status": "rejected",
    }


def test_mark_pending_reopens_rejected_turn_curation(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    session = ReviewerSession(storage)

    session.reject()
    session.mark_reviewed()

    result = session.mark_pending()

    assert result["review"]["status"] == "pending"
    assert result["curation"] == {
        "status": "pending",
    }

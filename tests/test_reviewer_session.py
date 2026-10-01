import pytest
from voice_dataset.reviewer_session import ReviewerSession
from voice_dataset.storage import DatasetStorage
from voice_dataset.schema import TurnRecord
from voice_dataset.voices import create_voice


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

    result = session.ignore()

    assert result["assignment"]["status"] == "ignore"


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

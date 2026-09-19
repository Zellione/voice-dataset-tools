from typing import Any

from .storage import DatasetStorage


REVIEW_STATUSES = frozenset({
    "pending",
    "reviewed",
})

BOUNDARY_REVIEW_STATUSES = frozenset({
    "unknown",
    "complete",
    "clipped",
})


def set_turn_review_status(
    storage: DatasetStorage,
    turn_id: str,
    status: str,
) -> dict[str, Any]:
    if status not in REVIEW_STATUSES:
        raise ValueError(
            f"Invalid review status: {status}"
        )

    if storage.get_turn(turn_id) is None:
        raise KeyError(
            f"Turn does not exist: {turn_id}"
        )

    def update(
        record: dict[str, Any],
    ) -> dict[str, Any]:
        review = dict(
            record.get("review") or {}
        )

        review["status"] = status
        record["review"] = review

        return record

    return storage.update_turn(
        turn_id,
        update,
    )


def mark_turn_reviewed(
    storage: DatasetStorage,
    turn_id: str,
) -> dict[str, Any]:
    return set_turn_review_status(
        storage,
        turn_id,
        "reviewed",
    )


def mark_turn_pending(
    storage: DatasetStorage,
    turn_id: str,
) -> dict[str, Any]:
    return set_turn_review_status(
        storage,
        turn_id,
        "pending",
    )


def set_turn_boundary_review_status(
    storage: DatasetStorage,
    turn_id: str,
    status: str,
) -> dict[str, Any]:
    if status not in BOUNDARY_REVIEW_STATUSES:
        raise ValueError(
            f"Invalid boundary review status: {status}"
        )

    turn = storage.get_turn(turn_id)

    if turn is None:
        raise KeyError(
            f"Turn does not exist: {turn_id}"
        )

    metadata = turn.get("metadata") or {}
    evidence = (
        metadata.get("boundary_evidence") or {}
    )

    if not (
        evidence.get("near_source_start")
        or evidence.get("near_source_end")
    ):
        raise ValueError(
            "Turn has no source-boundary evidence"
        )

    def update(
        record: dict[str, Any],
    ) -> dict[str, Any]:
        review = dict(
            record.get("review") or {}
        )

        boundary = dict(
            review.get("boundary") or {}
        )

        boundary["status"] = status
        review["boundary"] = boundary
        record["review"] = review

        return record

    return storage.update_turn(
        turn_id,
        update,
    )


def mark_turn_boundary_complete(
    storage: DatasetStorage,
    turn_id: str,
) -> dict[str, Any]:
    return set_turn_boundary_review_status(
        storage,
        turn_id,
        "complete",
    )


def mark_turn_boundary_clipped(
    storage: DatasetStorage,
    turn_id: str,
) -> dict[str, Any]:
    return set_turn_boundary_review_status(
        storage,
        turn_id,
        "clipped",
    )


def mark_turn_boundary_unknown(
    storage: DatasetStorage,
    turn_id: str,
) -> dict[str, Any]:
    return set_turn_boundary_review_status(
        storage,
        turn_id,
        "unknown",
    )

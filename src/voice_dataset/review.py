from typing import Any

from .storage import DatasetStorage


REVIEW_STATUSES = frozenset({
    "pending",
    "reviewed",
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

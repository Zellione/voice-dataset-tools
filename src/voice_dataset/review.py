import shutil
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


def accept_alignment_recovery(
    storage: DatasetStorage,
    turn_id: str,
) -> dict[str, Any]:
    turn = storage.get_turn(turn_id)

    if turn is None:
        raise KeyError(
            f"Turn does not exist: {turn_id}"
        )

    metadata = turn.get("metadata") or {}
    alignment = (
        metadata.get("alignment_evidence") or {}
    )
    recovery = alignment.get("recovery") or {}

    if recovery.get("status") != "suggested":
        raise ValueError(
            "Turn has no suggested alignment recovery"
        )

    source_start = recovery.get("source_start")
    source_end = recovery.get("source_end")
    region_ids = recovery.get("region_ids")

    if not isinstance(
        source_start,
        (int, float),
    ):
        raise ValueError(
            "Alignment recovery has invalid source_start"
        )

    if not isinstance(
        source_end,
        (int, float),
    ):
        raise ValueError(
            "Alignment recovery has invalid source_end"
        )

    source_start = float(source_start)
    source_end = float(source_end)

    if source_end <= source_start:
        raise ValueError(
            "Alignment recovery has invalid geometry"
        )

    if not isinstance(region_ids, list) or not all(
        isinstance(region_id, str)
        for region_id in region_ids
    ):
        raise ValueError(
            "Alignment recovery has invalid region_ids"
        )

    if not region_ids:
        raise ValueError(
            "Alignment recovery has no regions"
        )

    def update(
        record: dict[str, Any],
    ) -> dict[str, Any]:
        record["source_start"] = source_start
        record["source_end"] = source_end
        record["source_regions"] = list(region_ids)

        # These were generated from the old geometry.
        record["representations"] = {}
        record["embeddings"] = {}

        record_metadata = dict(
            record.get("metadata") or {}
        )
        record_alignment = dict(
            record_metadata.get(
                "alignment_evidence"
            )
            or {}
        )
        record_recovery = dict(
            record_alignment.get("recovery")
            or {}
        )

        record_recovery["status"] = "accepted"
        record_recovery["accepted_by"] = "human"

        record_alignment["recovery"] = (
            record_recovery
        )
        record_alignment["status"] = "recovered"

        record_metadata["alignment_evidence"] = (
            record_alignment
        )
        record["metadata"] = record_metadata

        return record

    return storage.update_turn(
        turn_id,
        update,
    )


def accept_edge_recovery(
    storage: DatasetStorage,
    turn_id: str,
) -> dict[str, Any]:
    turn = storage.get_turn(turn_id)

    if turn is None:
        raise KeyError(
            f"Turn does not exist: {turn_id}"
        )

    metadata = turn.get("metadata") or {}
    edge = metadata.get("edge_evidence") or {}

    if edge.get("status") != "suggested":
        raise ValueError(
            "Turn has no suggested edge recovery"
        )

    edge_kind = edge.get("edge")

    if edge_kind not in {"start", "end"}:
        raise ValueError(
            "Edge recovery has invalid edge"
        )

    region_id = edge.get("region_id")

    if not isinstance(region_id, str) or not region_id:
        raise ValueError(
            "Edge recovery has invalid region_id"
        )

    source_start = float(
        turn["source_start"]
    )
    source_end = float(
        turn["source_end"]
    )
    transcript = turn.get("transcript")

    if edge_kind == "end":
        recovered_end = edge.get(
            "source_end"
        )
        candidate_token = edge.get(
            "candidate_token"
        )
        whisper_token = edge.get(
            "whisper_token"
        )
        candidate_text = edge.get(
            "candidate_text"
        )
        whisper_text = edge.get(
            "whisper_text"
        )

        if not isinstance(
            recovered_end,
            (int, float),
        ):
            raise ValueError(
                "Edge recovery has invalid "
                "source_end"
            )

        source_end = float(
            recovered_end
        )

        if source_end <= source_start:
            raise ValueError(
                "Edge recovery has invalid geometry"
            )

        if not all(
            isinstance(value, str) and value
            for value in (
                candidate_token,
                whisper_token,
                candidate_text,
                whisper_text,
            )
        ):
            raise ValueError(
                "Edge recovery has invalid "
                "transcript evidence"
            )

        if transcript != candidate_text:
            raise ValueError(
                "Turn transcript no longer matches "
                "edge recovery candidate"
            )

        # For end extension the human acceptance
        # confirms both the extended geometry and
        # Whisper's lexical extension.
        transcript = whisper_text

    else:
        recovered_start = edge.get(
            "source_start"
        )
        candidate_text = edge.get(
            "candidate_text"
        )

        if not isinstance(
            recovered_start,
            (int, float),
        ):
            raise ValueError(
                "Edge recovery has invalid "
                "source_start"
            )

        source_start = float(
            recovered_start
        )

        if source_start >= source_end:
            raise ValueError(
                "Edge recovery has invalid geometry"
            )

        if not isinstance(
            candidate_text,
            str,
        ) or not candidate_text:
            raise ValueError(
                "Edge recovery has invalid "
                "transcript evidence"
            )

        if transcript != candidate_text:
            raise ValueError(
                "Turn transcript no longer matches "
                "edge recovery candidate"
            )

        # Start recovery only fixes geometry.
        # Whisper is evidence for the boundary, not
        # canonical lexical truth.

    source_regions = list(
        turn.get("source_regions") or []
    )

    if edge_kind == "start":
        conflicting_region_id = edge.get(
            "conflicting_region_id"
        )

        if (
            isinstance(
                conflicting_region_id,
                str,
            )
            and conflicting_region_id
            in source_regions
        ):
            source_regions.remove(
                conflicting_region_id
            )

    if region_id not in source_regions:
        source_regions.append(region_id)

    artifact_dir = (
        storage.root
        / "turns"
        / turn_id
    )

    def update(
        record: dict[str, Any],
    ) -> dict[str, Any]:
        record["source_start"] = source_start
        record["source_end"] = source_end
        record["source_regions"] = list(
            source_regions
        )
        record["transcript"] = transcript

        # These were generated from the old geometry.
        record["representations"] = {}
        record["embeddings"] = {}

        record_metadata = dict(
            record.get("metadata") or {}
        )
        record_edge = dict(
            record_metadata.get(
                "edge_evidence"
            )
            or {}
        )

        record_edge["status"] = "accepted"
        record_edge["accepted_by"] = "human"

        record_metadata["edge_evidence"] = (
            record_edge
        )
        record["metadata"] = record_metadata

        return record

    updated = storage.update_turn(
        turn_id,
        update,
    )

    # Physical files are derived from the old geometry.
    # Remove them only after the canonical update
    # succeeded.
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)

    return updated

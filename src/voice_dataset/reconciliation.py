from __future__ import annotations

from typing import Any

from .schema import (
    REJECTION_REASONS,
    TurnRecord,
)
from .storage import DatasetStorage


def region_reconciliation(
    region: dict[str, Any],
) -> dict[str, Any]:
    reconciliation = region.get(
        "reconciliation"
    )

    if reconciliation is None:
        return {
            "status": "pending",
            "reason": None,
            "notes": None,
        }

    if not isinstance(reconciliation, dict):
        raise ValueError(
            f"{region.get('id', '<unknown>')}: "
            "reconciliation must be an object"
        )

    status = reconciliation.get("status")

    if status not in (
        "pending",
        "rejected",
    ):
        raise ValueError(
            f"{region.get('id', '<unknown>')}: "
            "invalid reconciliation status: "
            f"{status!r}"
        )

    reason = reconciliation.get("reason")
    notes = reconciliation.get("notes")

    if status == "rejected":
        if reason not in REJECTION_REASONS:
            raise ValueError(
                f"{region.get('id', '<unknown>')}: "
                "rejected reconciliation requires "
                "a valid reason"
            )

    elif reason is not None:
        raise ValueError(
            f"{region.get('id', '<unknown>')}: "
            "pending reconciliation must not "
            "contain reason"
        )

    if (
        notes is not None
        and not isinstance(notes, str)
    ):
        raise ValueError(
            f"{region.get('id', '<unknown>')}: "
            "reconciliation notes must be "
            "a string or null"
        )

    return {
        "status": status,
        "reason": reason,
        "notes": notes,
    }


def _turns_using_region(
    storage: DatasetStorage,
    region_id: str,
) -> list[dict[str, Any]]:
    matches = []

    for turn in storage.turns.load():
        source_regions = turn.get(
            "source_regions"
        )

        if not isinstance(
            source_regions,
            list,
        ):
            raise ValueError(
                f"{turn.get('id', '<unknown>')}: "
                "source_regions must be a list"
            )

        if region_id in source_regions:
            matches.append(turn)

    return matches


def effective_region_reconciliation(
    storage: DatasetStorage,
    region: dict[str, Any],
) -> dict[str, Any]:
    region_id = region.get("id")

    if not isinstance(region_id, str) or not region_id:
        raise ValueError(
            "Candidate region has invalid id"
        )

    explicit = region_reconciliation(region)

    turns = _turns_using_region(
        storage,
        region_id,
    )

    if len(turns) > 1:
        raise RuntimeError(
            "Candidate region belongs to multiple "
            f"turns: {region_id}"
        )

    if turns:
        if explicit["status"] == "rejected":
            raise RuntimeError(
                "Rejected candidate region belongs "
                f"to a turn: {region_id}"
            )

        return {
            "status": "reconciled",
            "turn_id": turns[0]["id"],
            "reason": None,
            "notes": explicit["notes"],
        }

    return {
        "status": explicit["status"],
        "turn_id": None,
        "reason": explicit["reason"],
        "notes": explicit["notes"],
    }


def _find_existing_turn(
    storage: DatasetStorage,
    *,
    source_id: str,
    region_ids: list[str],
) -> dict[str, Any] | None:
    matches = [
        turn
        for turn in storage.turns.load()
        if (
            turn.get("source_id") == source_id
            and turn.get("source_regions") == region_ids
        )
    ]

    if not matches:
        return None

    if len(matches) > 1:
        raise RuntimeError(
            "Duplicate reconciled turns exist for "
            f"candidate regions: {region_ids}"
        )

    return matches[0]


def _check_existing_turn(
    turn: dict[str, Any],
    *,
    transcript: str | None,
    language: str | None,
) -> None:
    if turn.get("transcript") != transcript:
        raise ValueError(
            "Reconciliation conflict: existing "
            f"{turn['id']} has transcript "
            f"{turn.get('transcript')!r}, requested "
            f"{transcript!r}"
        )

    if turn.get("language") != language:
        raise ValueError(
            "Reconciliation conflict: existing "
            f"{turn['id']} has language "
            f"{turn.get('language')!r}, requested "
            f"{language!r}"
        )


def reject_region(
    storage: DatasetStorage,
    region_id: str,
    *,
    reason: str,
    notes: str | None = None,
) -> dict[str, Any]:
    if reason not in REJECTION_REASONS:
        raise ValueError(
            "Invalid rejection reason: "
            f"{reason!r}"
        )

    region = storage.get_region(region_id)

    if region is None:
        raise KeyError(
            "Candidate region does not exist: "
            f"{region_id}"
        )

    effective = effective_region_reconciliation(
        storage,
        region,
    )

    if effective["status"] == "reconciled":
        raise ValueError(
            "Candidate region already belongs "
            "to a turn: "
            f"{region_id} -> "
            f"{effective['turn_id']}"
        )

    if effective["status"] == "rejected":
        if (
            effective["reason"] == reason
            and effective["notes"] == notes
        ):
            return region

        raise ValueError(
            "Candidate region already has a "
            "different rejection decision: "
            f"{region_id}"
        )

    def update(
        record: dict[str, Any],
    ) -> dict[str, Any]:
        record["schema_version"] = 2
        record["reconciliation"] = {
            "status": "rejected",
            "reason": reason,
            "notes": notes,
        }

        return record

    return storage.update_region(
        region_id,
        update,
    )


def create_turn_from_regions(
    storage: DatasetStorage,
    region_ids: list[str],
    *,
    transcript: str | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    if not region_ids:
        raise ValueError(
            "At least one candidate region is required"
        )

    if len(set(region_ids)) != len(region_ids):
        raise ValueError(
            "Candidate region ids must be unique"
        )

    regions = []

    for region_id in region_ids:
        region = storage.get_region(region_id)

        if region is None:
            raise KeyError(
                f"Candidate region does not exist: "
                f"{region_id}"
            )

        regions.append(region)

    source_ids = {
        region.get("source_id")
        for region in regions
    }

    if len(source_ids) != 1:
        raise ValueError(
            "All candidate regions must belong "
            "to the same source"
        )

    source_id = next(iter(source_ids))

    if not isinstance(source_id, str):
        raise ValueError(
            "Candidate regions contain an invalid "
            "source_id"
        )

    ordered = sorted(
        regions,
        key=lambda region: (
            region["source_start"],
            region["source_end"],
            region["id"],
        ),
    )

    ordered_ids = [
        region["id"]
        for region in ordered
    ]

    if ordered_ids != region_ids:
        raise ValueError(
            "Candidate regions must be supplied "
            "in source timeline order"
        )

    previous_end = None

    for region in ordered:
        start = region.get("source_start")
        end = region.get("source_end")

        if not isinstance(
            start,
            (int, float),
        ):
            raise ValueError(
                f"{region['id']}: invalid source_start"
            )

        if not isinstance(
            end,
            (int, float),
        ):
            raise ValueError(
                f"{region['id']}: invalid source_end"
            )

        if end <= start:
            raise ValueError(
                f"{region['id']}: invalid source range"
            )

        if (
            previous_end is not None
            and start < previous_end
        ):
            raise ValueError(
                "Candidate regions for one turn "
                "must not overlap"
            )

        previous_end = end

    for region in ordered:
        effective = (
            effective_region_reconciliation(
                storage,
                region,
            )
        )

        if effective["status"] == "rejected":
            raise ValueError(
                "Candidate region is rejected: "
                f"{region['id']} "
                f"({effective['reason']})"
            )

        if effective["status"] == "reconciled":
            existing_turn_id = effective[
                "turn_id"
            ]

            existing_turn = storage.get_turn(
                existing_turn_id
            )

            if existing_turn is None:
                raise RuntimeError(
                    "Reconciled turn disappeared: "
                    f"{existing_turn_id}"
                )

            if (
                existing_turn.get(
                    "source_regions"
                )
                != ordered_ids
            ):
                raise ValueError(
                    "Candidate region already belongs "
                    "to another turn: "
                    f"{region['id']} -> "
                    f"{existing_turn_id}"
                )

    existing = _find_existing_turn(
        storage,
        source_id=source_id,
        region_ids=ordered_ids,
    )

    if existing is not None:
        _check_existing_turn(
            existing,
            transcript=transcript,
            language=language,
        )

        return existing

    turn = TurnRecord(
        id=storage.next_turn_id(),
        source_id=source_id,
        source_start=float(
            ordered[0]["source_start"]
        ),
        source_end=float(
            ordered[-1]["source_end"]
        ),
        source_regions=ordered_ids,
        language=language,
        transcript=transcript,
        metadata={
            "creation": {
                "method": "manual_reconciliation",
            }
        },
    )

    storage.add_turn(turn)

    stored = storage.get_turn(turn.id)

    if stored is None:
        raise RuntimeError(
            f"Failed to read back created turn: "
            f"{turn.id}"
        )

    return stored

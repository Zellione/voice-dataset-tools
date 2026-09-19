from __future__ import annotations
from pathlib import Path

from typing import Any

from .schema import (
    REJECTION_REASONS,
    TurnRecord,
)
from .storage import DatasetStorage
from .boundary_evidence import (
    calculate_boundary_evidence,
    find_boundary_representation,
)

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


def _boundary_evidence_for_turn(
    storage: DatasetStorage,
    *,
    source_id: str,
    start: float,
    end: float,
) -> dict[str, Any] | None:
    source = storage.get_source(source_id)

    if source is None:
        return None

    boundary_representation = (
        find_boundary_representation(source)
    )

    if boundary_representation is None:
        return None

    (
        boundary_representation_name,
        representation,
    ) = boundary_representation

    return calculate_boundary_evidence(
        representation_name=(
            boundary_representation_name
        ),
        representation=representation,
        start=start,
        end=end,
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

    boundary_evidence = _boundary_evidence_for_turn(
        storage,
        source_id=source_id,
        start=float(
            ordered[0]["source_start"]
        ),
        end=float(
            ordered[-1]["source_end"]
        ),
    )

    metadata = {
        "creation": {
            "method": "manual_reconciliation",
        }
    }

    if boundary_evidence is not None:
        metadata["boundary_evidence"] = (
            boundary_evidence
        )

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
        metadata=metadata,
    )

    storage.add_turn(turn)

    stored = storage.get_turn(turn.id)

    if stored is None:
        raise RuntimeError(
            f"Failed to read back created turn: "
            f"{turn.id}"
        )

    return stored


def edit_turn(
    storage: DatasetStorage,
    turn_id: str,
    *,
    transcript: str | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    turn = storage.get_turn(turn_id)

    if turn is None:
        raise KeyError(
            f"Unknown turn: {turn_id}"
        )

    if transcript is not None:
        transcript = transcript.strip()

        if not transcript:
            raise ValueError(
                "Transcript must not be empty"
            )

    if language is not None:
        language = language.strip()

        if not language:
            raise ValueError(
                "Language must not be empty"
            )

    def update(
        record: dict[str, Any],
    ) -> dict[str, Any]:
        if transcript is not None:
            record["transcript"] = transcript

        if language is not None:
            record["language"] = language

        return record

    return storage.update_turn(
        turn_id,
        update,
    )


def split_turn(
    storage: DatasetStorage,
    turn_id: str,
    *,
    after_region_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    turns = storage.turns.load()

    matches = [
        (index, turn)
        for index, turn in enumerate(turns)
        if turn.get("id") == turn_id
    ]

    if not matches:
        raise KeyError(
            f"Turn does not exist: {turn_id}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Duplicate turn id: {turn_id}"
        )

    turn_index, original = matches[0]

    source_regions = original.get("source_regions")

    if not isinstance(source_regions, list):
        raise ValueError(
            f"{turn_id}: source_regions must be a list"
        )

    if len(source_regions) < 2:
        raise ValueError(
            "A turn must contain at least two regions "
            "to be split"
        )

    if after_region_id not in source_regions:
        raise ValueError(
            f"{after_region_id} does not belong to "
            f"{turn_id}"
        )

    split_index = (
        source_regions.index(after_region_id) + 1
    )

    if split_index >= len(source_regions):
        raise ValueError(
            "Cannot split after the final region "
            f"of {turn_id}"
        )

    left_ids = source_regions[:split_index]
    right_ids = source_regions[split_index:]

    regions: dict[str, dict[str, Any]] = {}

    for region_id in source_regions:
        region = storage.get_region(region_id)

        if region is None:
            raise KeyError(
                "Candidate region does not exist: "
                f"{region_id}"
            )

        regions[region_id] = region

    source_id = original.get("source_id")

    if not isinstance(source_id, str):
        raise ValueError(
            f"{turn_id}: invalid source_id"
        )

    for region_id in source_regions:
        region = regions[region_id]

        if region.get("source_id") != source_id:
            raise ValueError(
                f"{region_id}: source does not match "
                f"{turn_id}"
            )

        effective = effective_region_reconciliation(
            storage,
            region,
        )

        if (
            effective["status"] != "reconciled"
            or effective["turn_id"] != turn_id
        ):
            raise ValueError(
                f"{region_id} is not exclusively "
                f"reconciled to {turn_id}"
            )

    def bounds(
        region_ids: list[str],
    ) -> tuple[float, float]:
        first = regions[region_ids[0]]
        last = regions[region_ids[-1]]

        start = first.get("source_start")
        end = last.get("source_end")

        if not isinstance(start, (int, float)):
            raise ValueError(
                f"{first['id']}: invalid source_start"
            )

        if not isinstance(end, (int, float)):
            raise ValueError(
                f"{last['id']}: invalid source_end"
            )

        if end <= start:
            raise ValueError(
                "Invalid split turn source range"
            )

        return float(start), float(end)

    left_start, left_end = bounds(left_ids)
    right_start, right_end = bounds(right_ids)

    new_turn_id = storage.next_turn_id()

    def fresh_turn(
        *,
        record_id: str,
        region_ids: list[str],
        start: float,
        end: float,
        split_side: str,
    ) -> dict[str, Any]:
        metadata = {
            "creation": {
                "method": "manual_split",
                "source_turn_id": turn_id,
                "split_after_region":
                    after_region_id,
                "split_side": split_side,
            }
        }

        boundary_evidence = (
            _boundary_evidence_for_turn(
                storage,
                source_id=source_id,
                start=start,
                end=end,
            )
        )

        if boundary_evidence is not None:
            metadata["boundary_evidence"] = (
                boundary_evidence
            )

        return TurnRecord(
            id=record_id,
            source_id=source_id,
            source_start=start,
            source_end=end,
            source_regions=region_ids,
            metadata=metadata,
        ).to_dict()

    left = fresh_turn(
        record_id=turn_id,
        region_ids=left_ids,
        start=left_start,
        end=left_end,
        split_side="left",
    )

    right = fresh_turn(
        record_id=new_turn_id,
        region_ids=right_ids,
        start=right_start,
        end=right_end,
        split_side="right",
    )

    final_turns = turns.copy()
    final_turns[turn_index] = left
    final_turns.append(right)

    seen_regions: dict[str, str] = {}

    for turn in final_turns:
        candidate_ids = turn.get("source_regions")

        if not isinstance(candidate_ids, list):
            raise ValueError(
                f"{turn.get('id', '<unknown>')}: "
                "source_regions must be a list"
            )

        for region_id in candidate_ids:
            previous = seen_regions.get(region_id)

            if previous is not None:
                raise ValueError(
                    "Candidate region would belong "
                    "to multiple turns: "
                    f"{region_id} -> "
                    f"{previous}, {turn.get('id')}"
                )

            seen_regions[region_id] = turn.get("id")

    old_representations = original.get(
        "representations",
        {},
    )

    if not isinstance(old_representations, dict):
        raise ValueError(
            f"{turn_id}: representations must be a dict"
        )

    stale_paths: list[Path] = []

    for name, representation in (
        old_representations.items()
    ):
        if not isinstance(representation, dict):
            raise ValueError(
                f"{turn_id}/{name}: invalid representation"
            )

        relative_path = representation.get("path")

        if not isinstance(relative_path, str):
            raise ValueError(
                f"{turn_id}/{name}: "
                "representation has invalid path"
            )

        candidate = (
            storage.root / relative_path
        ).resolve()

        turn_directory = (
            storage.root
            / "turns"
            / turn_id
        ).resolve()

        try:
            candidate.relative_to(turn_directory)
        except ValueError as exc:
            raise ValueError(
                f"{turn_id}/{name}: "
                "representation path escapes "
                "turn directory"
            ) from exc

        stale_paths.append(candidate)

    storage.turns.replace(final_turns)

    for stale_path in stale_paths:
        stale_path.unlink(
            missing_ok=True
        )

    stored_left = storage.get_turn(turn_id)
    stored_right = storage.get_turn(new_turn_id)

    if stored_left is None or stored_right is None:
        raise RuntimeError(
            "Failed to read back split turns"
        )

    return stored_left, stored_right

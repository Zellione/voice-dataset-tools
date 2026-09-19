from __future__ import annotations

from typing import Any

from .schema import TurnRecord
from .storage import DatasetStorage


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

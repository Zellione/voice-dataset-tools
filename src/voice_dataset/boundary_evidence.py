from __future__ import annotations

from typing import Any
from .storage import DatasetStorage

SOURCE_BOUNDARY_MARGIN = 0.1


def select_boundary_representation(
    source: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    representations = source.get(
        "representations",
        {},
    )

    if not isinstance(representations, dict):
        raise ValueError(
            "Source has invalid representations"
        )

    matches = []

    for name, representation in representations.items():
        if not isinstance(representation, dict):
            continue

        purposes = representation.get(
            "purposes",
            [],
        )

        if "boundary_analysis" in purposes:
            matches.append(
                (
                    name,
                    representation,
                )
            )

    if not matches:
        raise ValueError(
            "Source has no boundary_analysis representation"
        )

    if len(matches) > 1:
        names = ", ".join(
            name
            for name, _ in matches
        )

        raise ValueError(
            "Source has multiple boundary_analysis "
            f"representations: {names}"
        )

    return matches[0]


def find_boundary_representation(
    source: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    representations = source.get(
        "representations",
        {},
    )

    if not isinstance(representations, dict):
        raise ValueError(
            "Source has invalid representations"
        )

    matches = []

    for name, representation in representations.items():
        if not isinstance(representation, dict):
            continue

        purposes = representation.get(
            "purposes",
            [],
        )

        if "boundary_analysis" in purposes:
            matches.append(
                (
                    name,
                    representation,
                )
            )

    if not matches:
        return None

    if len(matches) > 1:
        names = ", ".join(
            name
            for name, _ in matches
        )

        raise ValueError(
            "Source has multiple boundary_analysis "
            f"representations: {names}"
        )

    return matches[0]


def calculate_boundary_evidence(
    *,
    representation_name: str,
    representation: dict[str, Any],
    start: float,
    end: float,
    margin: float = SOURCE_BOUNDARY_MARGIN,
) -> dict[str, Any]:
    start = float(start)
    end = float(end)
    margin = float(margin)

    if start < 0:
        raise ValueError(
            "Turn start must not be negative"
        )

    if end <= start:
        raise ValueError(
            "Turn end must be after turn start"
        )

    if margin < 0:
        raise ValueError(
            "Boundary margin must not be negative"
        )

    duration = representation.get("duration")

    if duration is None:
        raise ValueError(
            "Source representation has no known duration"
        )

    duration = float(duration)

    if duration <= 0:
        raise ValueError(
            "Source representation duration "
            "must be positive"
        )

    if end > duration:
        raise ValueError(
            "Turn ends after source representation"
        )

    return {
        "representation": representation_name,
        "margin": margin,
        "near_source_start": start <= margin,
        "near_source_end": (
            duration - end <= margin
        ),
    }


def refresh_turn_boundary_evidence(
    storage: DatasetStorage,
    turn_id: str,
) -> dict[str, Any]:
    turn = storage.get_turn(turn_id)

    if turn is None:
        raise KeyError(
            f"Unknown turn: {turn_id}"
        )

    source_id = turn.get("source_id")

    if not isinstance(source_id, str):
        raise ValueError(
            f"{turn_id}: invalid source_id"
        )

    source = storage.get_source(source_id)

    if source is None:
        raise KeyError(
            f"Unknown source: {source_id}"
        )

    boundary_representation = (
        find_boundary_representation(source)
    )

    start = turn.get("source_start")
    end = turn.get("source_end")

    if not isinstance(start, (int, float)):
        raise ValueError(
            f"{turn_id}: invalid source_start"
        )

    if not isinstance(end, (int, float)):
        raise ValueError(
            f"{turn_id}: invalid source_end"
        )

    evidence = None

    if boundary_representation is not None:
        (
            representation_name,
            representation,
        ) = boundary_representation

        evidence = calculate_boundary_evidence(
            representation_name=representation_name,
            representation=representation,
            start=float(start),
            end=float(end),
        )

    def update(
        record: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = record.setdefault(
            "metadata",
            {},
        )

        previous_evidence = metadata.get(
            "boundary_evidence"
        )

        evidence_changed = (
            previous_evidence != evidence
        )

        if evidence is None:
            metadata.pop(
                "boundary_evidence",
                None,
            )
        else:
            metadata["boundary_evidence"] = (
                evidence
            )

        if evidence_changed:
            review = record.get("review")

            if isinstance(review, dict):
                review.pop("boundary", None)

        return record

    return storage.update_turn(
        turn_id,
        update,
    )


def refresh_source_boundary_evidence(
    storage: DatasetStorage,
    source_id: str,
) -> list[dict[str, Any]]:
    source = storage.get_source(source_id)

    if source is None:
        raise KeyError(
            f"Unknown source: {source_id}"
        )

    turns = [
        turn
        for turn in storage.turns.load()
        if turn.get("source_id") == source_id
    ]

    turns.sort(
        key=lambda turn: (
            turn.get("source_start", 0.0),
            turn.get("source_end", 0.0),
            turn.get("id", ""),
        )
    )

    updated = []

    for turn in turns:
        turn_id = turn.get("id")

        if not isinstance(turn_id, str):
            raise ValueError(
                "Turn has invalid id"
            )

        updated.append(
            refresh_turn_boundary_evidence(
                storage,
                turn_id,
            )
        )

    return updated

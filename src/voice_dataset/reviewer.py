from __future__ import annotations

from typing import Any

from .storage import DatasetStorage


def sorted_turns(
    storage: DatasetStorage,
    *,
    source_id: str | None = None,
) -> list[dict[str, Any]]:
    turns = storage.turns.load()

    if source_id is not None:
        turns = [
            turn
            for turn in turns
            if turn.get("source_id") == source_id
        ]

    return sorted(
        turns,
        key=lambda turn: (
            str(turn.get("source_id", "")),
            float(turn.get("source_start", 0.0)),
            float(turn.get("source_end", 0.0)),
            str(turn.get("id", "")),
        ),
    )


def turn_review_status(
    turn: dict[str, Any],
) -> str:
    review = turn.get("review") or {}

    return str(
        review.get("status") or "pending"
    )


def turn_boundary_review_status(
    turn: dict[str, Any],
) -> str:
    review = turn.get("review") or {}
    boundary = review.get("boundary") or {}

    return str(
        boundary.get("status") or "unknown"
    )


def turn_boundary_text(
    turn: dict[str, Any],
) -> str:
    metadata = turn.get("metadata") or {}
    evidence = (
        metadata.get("boundary_evidence") or {}
    )

    near_start = bool(
        evidence.get("near_source_start")
    )
    near_end = bool(
        evidence.get("near_source_end")
    )

    if not near_start and not near_end:
        return "-"

    if near_start and near_end:
        location = "near start/end"
    elif near_start:
        location = "near start"
    else:
        location = "near end"

    return (
        f"{location} -> "
        f"{turn_boundary_review_status(turn)}"
    )


def turn_assignment_text(
    turn: dict[str, Any],
) -> str:
    assignment = turn.get("assignment") or {}
    status = assignment.get("status") or "unknown"

    if status == "assigned":
        voice_id = assignment.get("voice_id")

        if voice_id:
            return f"assigned -> {voice_id}"

    return str(status)


def turn_representation_names(
    turn: dict[str, Any],
) -> list[str]:
    representations = (
        turn.get("representations") or {}
    )

    return list(representations)


def format_turn(
    turn: dict[str, Any],
    *,
    position: int,
    total: int,
) -> str:
    start = float(turn["source_start"])
    end = float(turn["source_end"])
    duration = end - start

    language = turn.get("language") or "-"
    transcript = turn.get("transcript") or "-"

    regions = turn.get("source_regions") or []
    representations = turn_representation_names(
        turn
    )

    region_text = (
        ", ".join(regions)
        if regions
        else "-"
    )

    representation_text = (
        ", ".join(representations)
        if representations
        else "-"
    )

    lines = [
        (
            f"[{position}/{total}] "
            f"{turn['id']}"
        ),
        f"source:          {turn['source_id']}",
        (
            f"range:           "
            f"{start:.3f}-{end:.3f} "
            f"({duration:.3f}s)"
        ),
        f"regions:         {region_text}",
        f"language:        {language}",
        (
            "assignment:      "
            f"{turn_assignment_text(turn)}"
        ),
        (
            "review:          "
            f"{turn_review_status(turn)}"
        ),
        (
            "boundary:        "
            f"{turn_boundary_text(turn)}"
        ),
        (
            "representations: "
            f"{representation_text}"
        ),
        f"transcript:      {transcript}",
    ]

    return "\n".join(lines)

def reviewer_help() -> str:
    return "\n".join([
        "Commands:",
        "  p  play preferred review audio",
        "  r  play raw representation",
        "  c  play source context",
        "  s  stop playback",
        "  t  edit transcript",
        "  l  set language",
        "  v  assign voice",
        "  u  mark voice unknown",
        "  i  ignore turn",
        "  a  mark reviewed",
        "  x  mark review pending",
        "  k  mark boundary complete",
        "  d  mark boundary clipped",
        "  n  next turn",
        "  b  previous turn",
        "  h  show help",
        "  q  quit",
    ])


def raw_representation(
    turn: dict[str, Any],
) -> dict[str, Any]:
    representations = (
        turn.get("representations") or {}
    )

    representation = representations.get("raw")

    if representation is None:
        raise ValueError(
            f"Turn {turn.get('id', '<unknown>')} "
            "has no raw representation"
        )

    return representation

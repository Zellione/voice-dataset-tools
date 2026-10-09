from __future__ import annotations

from typing import Any

from .storage import DatasetStorage
from .speaker_similarity import VoiceMatch


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


def format_voice_matches(
    matches: list[VoiceMatch],
    voices: dict[str, dict[str, Any]],
    *,
    limit_per_voice: int = 3,
) -> str:
    if limit_per_voice <= 0:
        raise ValueError(
            "limit_per_voice must be positive"
        )

    if not matches:
        return "No speaker evidence."

    lines: list[str] = []

    for voice_match in matches:
        voice = voices.get(voice_match.voice_id)
        character = (
            voice.get("character")
            if voice is not None
            else None
        )

        if character:
            heading = (
                f"{voice_match.voice_id} "
                f"({character})"
            )
        else:
            heading = voice_match.voice_id

        lines.append(heading)

        for match in voice_match.matches[
            :limit_per_voice
        ]:
            lines.append(
                f"  {match.turn_id}: "
                f"{match.similarity * 100:.1f}% similarity"
            )

    return "\n".join(lines)


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

    metadata = turn.get("metadata") or {}

    automatic_pipeline = metadata.get(
        "automatic_pipeline"
    )
    speaker_evidence = metadata.get(
        "speaker_evidence"
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
    ]

    if isinstance(
        automatic_pipeline,
        dict,
    ):
        status = (
            automatic_pipeline.get("status")
            or "-"
        )

        reasons = (
            automatic_pipeline.get(
                "review_reasons"
            )
            or []
        )

        reason_text = (
            ", ".join(reasons)
            if reasons
            else "-"
        )

        lines.extend([
            f"auto:            {status}",
            f"auto reasons:    {reason_text}",
        ])

    if isinstance(
        speaker_evidence,
        dict,
    ):
        speaker = (
            speaker_evidence.get("speaker")
            or "-"
        )

        known_speakers = (
            speaker_evidence.get(
                "known_speakers"
            )
            or []
        )

        unresolved_words = (
            speaker_evidence.get(
                "unresolved_word_indices"
            )
            or []
        )

        known_text = (
            ", ".join(known_speakers)
            if known_speakers
            else "-"
        )

        unresolved_text = (
            ", ".join(
                str(index)
                for index in unresolved_words
            )
            if unresolved_words
            else "-"
        )

        lines.extend([
            f"auto speaker:    {speaker}",
            f"speaker evidence:{known_text}",
            f"unresolved words:{unresolved_text}",
        ])

    alignment_evidence = metadata.get(
        "alignment_evidence"
    )

    if isinstance(alignment_evidence, dict):
        alignment_status = (
            alignment_evidence.get("status")
            or "-"
        )

        alignment_words = (
            alignment_evidence.get(
                "issue_word_indices"
            )
            or []
        )

        alignment_words_text = (
            ", ".join(
                str(index)
                for index in alignment_words
            )
            if alignment_words
            else "-"
        )

        lines.extend([
            f"alignment:       {alignment_status}",
            f"alignment words: {alignment_words_text}",
        ])

        recovery = alignment_evidence.get(
            "recovery"
        )

        if isinstance(recovery, dict):
            recovery_start = recovery.get(
                "source_start"
            )
            recovery_end = recovery.get(
                "source_end"
            )
            recovery_regions = (
                recovery.get("region_ids")
                or []
            )
            matched_count = recovery.get(
                "matched_token_count"
            )
            candidate_count = recovery.get(
                "candidate_token_count"
            )

            if (
                isinstance(
                    recovery_start,
                    (int, float),
                )
                and isinstance(
                    recovery_end,
                    (int, float),
                )
            ):
                recovery_range = (
                    f"{recovery_start:.3f}-"
                    f"{recovery_end:.3f} "
                    f"({recovery_end - recovery_start:.3f}s)"
                )
            else:
                recovery_range = "-"

            recovery_match = (
                f"{matched_count}/{candidate_count} words"
                if (
                    isinstance(matched_count, int)
                    and isinstance(
                        candidate_count,
                        int,
                    )
                )
                else "-"
            )

            lines.extend([
                (
                    "recovery:        "
                    f"{recovery.get('status') or '-'}"
                ),
                (
                    "recovery range:  "
                    f"{recovery_range}"
                ),
                (
                    "recovery speaker:"
                    f"{recovery.get('speaker') or '-'}"
                ),
                (
                    "recovery regions:"
                    + (
                        ", ".join(recovery_regions)
                        if recovery_regions
                        else "-"
                    )
                ),
                (
                    "recovery match:  "
                    f"{recovery_match}"
                ),
            ])

    lines.append(
        f"transcript:      {transcript}"
    )

    return "\n".join(lines)

def reviewer_help() -> str:
    return "\n".join([
        "Commands:",
        "  p  play preferred review audio",
        "  r  play raw representation",
        "  c  play source context",
        "  g  play alignment recovery suggestion",
        "  f  accept alignment recovery",
        "  e  accept turn edge recovery",
        "  s  stop playback",
        "  t  edit transcript",
        "  l  set language",
        "  v  assign voice",
        "  u  mark voice unknown",
        "  i  reject turn",
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

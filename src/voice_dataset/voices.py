from __future__ import annotations

from typing import Any

from .schema import VoiceProfile
from .storage import DatasetStorage


def create_voice(
    storage: DatasetStorage,
    *,
    character: str | None = None,
    language: str | None = None,
    aliases: list[str] | None = None,
    ignored: bool = False,
    notes: str | None = None,
) -> dict[str, Any]:
    voice = VoiceProfile(
        id=storage.next_voice_id(),
        character=character,
        language=language,
        aliases=aliases or [],
        ignored=ignored,
        notes=notes,
    )

    storage.add_voice(voice)

    result = storage.get_voice(voice.id)

    if result is None:
        raise RuntimeError(
            f"Failed to create voice: {voice.id}"
        )

    return result


def assign_turn(
    storage: DatasetStorage,
    turn_id: str,
    voice_id: str,
    *,
    method: str = "manual",
    confidence: float | None = None,
) -> dict[str, Any]:
    voice = storage.get_voice(voice_id)

    if voice is None:
        raise KeyError(
            f"Voice does not exist: {voice_id}"
        )

    if storage.get_turn(turn_id) is None:
        raise KeyError(
            f"Turn does not exist: {turn_id}"
        )

    def update(
        record: dict[str, Any],
    ) -> dict[str, Any]:
        record["assignment"] = {
            "status": "assigned",
            "voice_id": voice_id,
            "method": method,
            "confidence": confidence,
        }

        review = dict(
            record.get("review") or {}
        )
        review.pop(
            "speaker_calibration",
            None,
        )
        record["review"] = review

        return record

    updated = storage.turns.update(
        turn_id,
        update,
    )

    if bool(voice.get("ignored")):
        def reject(
            record: dict[str, Any],
        ) -> dict[str, Any]:
            curation = dict(
                record.get("curation")
                or {}
            )

            curation["status"] = "rejected"
            record["curation"] = curation

            return record

        updated = storage.update_turn(
            turn_id,
            reject,
        )

    return updated


def mark_turn_unknown(
    storage: DatasetStorage,
    turn_id: str,
) -> dict[str, Any]:
    if storage.get_turn(turn_id) is None:
        raise KeyError(
            f"Turn does not exist: {turn_id}"
        )

    def update(
        record: dict[str, Any],
    ) -> dict[str, Any]:
        record["assignment"] = {
            "status": "unknown",
            "voice_id": None,
            "method": None,
            "confidence": None,
        }

        review = dict(
            record.get("review") or {}
        )
        review.pop(
            "speaker_calibration",
            None,
        )
        record["review"] = review

        return record

    return storage.turns.update(
        turn_id,
        update,
    )



def set_voice_ignored(
    storage: DatasetStorage,
    voice_id: str,
    ignored: bool = True,
) -> dict[str, Any]:
    if storage.get_voice(voice_id) is None:
        raise KeyError(
            f"Voice does not exist: {voice_id}"
        )

    def update(
        record: dict[str, Any],
    ) -> dict[str, Any]:
        record["ignored"] = ignored
        return record

    voice = storage.voices.update(
        voice_id,
        update,
    )

    if ignored:
        for turn in storage.turns.load():
            assignment = turn.get(
                "assignment"
            )

            if not isinstance(
                assignment,
                dict,
            ):
                continue

            if (
                assignment.get("status")
                != "assigned"
                or assignment.get("voice_id")
                != voice_id
            ):
                continue

            turn_id = turn.get("id")

            if not isinstance(turn_id, str):
                continue

            def reject(
                record: dict[str, Any],
            ) -> dict[str, Any]:
                curation = dict(
                    record.get("curation")
                    or {}
                )

                curation["status"] = "rejected"
                record["curation"] = curation

                return record

            storage.update_turn(
                turn_id,
                reject,
            )

    return voice

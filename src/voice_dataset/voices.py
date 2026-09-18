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

        return record

    return storage.turns.update(
        turn_id,
        update,
    )


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

        return record

    return storage.turns.update(
        turn_id,
        update,
    )


def ignore_turn(
    storage: DatasetStorage,
    turn_id: str,
    *,
    method: str = "manual",
) -> dict[str, Any]:
    if storage.get_turn(turn_id) is None:
        raise KeyError(
            f"Turn does not exist: {turn_id}"
        )

    def update(
        record: dict[str, Any],
    ) -> dict[str, Any]:
        record["assignment"] = {
            "status": "ignore",
            "voice_id": None,
            "method": method,
            "confidence": None,
        }

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

    return storage.voices.update(
        voice_id,
        update,
    )

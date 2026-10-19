from voice_dataset.storage import DatasetStorage
from voice_dataset.voices import (
    assign_turn,
    create_voice,
    set_voice_ignored,
)



def test_assigning_ignored_voice_does_not_change_curation(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    storage.turns.append(
        {
            "schema_version": 2,
            "record_type": "turn",
            "id": "turn_001",
            "source_id": "source_001",
            "source_start": 1.0,
            "source_end": 2.0,
            "assignment": {
                "status": "unknown",
                "voice_id": None,
                "method": None,
                "confidence": None,
            },
            "curation": {
                "status": "pending",
            },
        }
    )

    voice = create_voice(
        storage,
        character="Minor Character",
        ignored=True,
    )

    assign_turn(
        storage,
        "turn_001",
        voice["id"],
    )

    turn = storage.get_turn(
        "turn_001"
    )

    assert turn["curation"] == {
        "status": "pending",
    }


def test_ignoring_voice_does_not_change_turn_curation(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    voice = create_voice(
        storage,
        character="Minor Character",
    )

    for index in (1, 2):
        turn_id = f"turn_{index:03d}"

        storage.turns.append(
            {
                "schema_version": 2,
                "record_type": "turn",
                "id": turn_id,
                "source_id": "source_001",
                "source_start": float(index),
                "source_end": float(index + 1),
                "assignment": {
                    "status": "assigned",
                    "voice_id": voice["id"],
                    "method": "manual",
                    "confidence": None,
                },
                "curation": {
                    "status": "accepted",
                },
            }
        )

    set_voice_ignored(
        storage,
        voice["id"],
        True,
    )

    for index in (1, 2):
        turn = storage.get_turn(
            f"turn_{index:03d}"
        )

        assert turn["curation"] == {
            "status": "accepted",
        }

def test_rejected_turn_does_not_ignore_voice(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    voice = create_voice(
        storage,
        character="Main Character",
    )

    storage.turns.append(
        {
            "schema_version": 2,
            "record_type": "turn",
            "id": "turn_001",
            "source_id": "source_001",
            "source_start": 1.0,
            "source_end": 2.0,
            "assignment": {
                "status": "assigned",
                "voice_id": voice["id"],
                "method": "manual",
                "confidence": None,
            },
            "curation": {
                "status": "rejected",
            },
        }
    )

    turn = storage.get_turn(
        "turn_001"
    )
    stored_voice = storage.get_voice(
        voice["id"]
    )

    assert (
        turn["curation"]["status"]
        == "rejected"
    )
    assert stored_voice["ignored"] is False


def test_rename_voice_preserves_voice_id_and_assignments(
    tmp_path,
):
    from voice_dataset.voices import rename_voice

    storage = DatasetStorage(tmp_path)

    voice = create_voice(
        storage,
        character="Vi",
    )

    storage.turns.append(
        {
            "schema_version": 2,
            "record_type": "turn",
            "id": "turn_001",
            "source_id": "source_001",
            "source_start": 1.0,
            "source_end": 2.0,
            "assignment": {
                "status": "assigned",
                "voice_id": voice["id"],
                "method": "manual",
                "confidence": None,
            },
        }
    )

    updated = rename_voice(
        storage,
        voice["id"],
        "Vi (young)",
    )

    assert updated["id"] == voice["id"]
    assert updated["character"] == "Vi (young)"

    turn = storage.get_turn("turn_001")

    assert turn is not None
    assert (
        turn["assignment"]["voice_id"]
        == voice["id"]
    )

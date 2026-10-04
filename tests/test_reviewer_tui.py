import pytest

from voice_dataset.reviewer_session import ReviewerSession
from voice_dataset.reviewer_tui import ReviewerTUI
from voice_dataset.storage import DatasetStorage

from textual.widgets import Input, Static


def add_turn(
    storage: DatasetStorage,
    turn_id: str,
    *,
    source_start: float,
) -> None:
    storage.turns.append(
        {
            "schema_version": 2,
            "record_type": "turn",
            "id": turn_id,
            "source_id": "source_001",
            "source_start": source_start,
            "source_end": source_start + 1.0,
            "source_regions": [],
            "language": "en",
            "transcript": turn_id,
            "representations": {},
            "embeddings": {},
            "assignment": {
                "status": "unknown",
                "voice_id": None,
                "method": None,
                "confidence": None,
            },
            "review": {
                "status": "pending",
            },
            "metadata": {},
        }
    )


@pytest.mark.asyncio
async def test_reviewer_tui_navigates_turns(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )
    add_turn(
        storage,
        "turn_000002",
        source_start=2.0,
    )

    session = ReviewerSession(storage)
    app = ReviewerTUI(
        session,
        embedding_names=(
            "ecapa_speaker",
            "wespeaker_speaker",
        ),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        assert session.current_turn_id == "turn_000001"

        await pilot.press("right")
        await pilot.pause()

        assert session.current_turn_id == "turn_000002"

        await pilot.press("left")
        await pilot.pause()

        assert session.current_turn_id == "turn_000001"


@pytest.mark.asyncio
async def test_reviewer_tui_opens_help(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    session = ReviewerSession(storage)
    app = ReviewerTUI(
        session,
        embedding_names=(
            "ecapa_speaker",
            "wespeaker_speaker",
        ),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        assert len(app.screen_stack) == 1

        await pilot.press("?")
        await pilot.pause()

        assert len(app.screen_stack) == 2


@pytest.mark.asyncio
async def test_reviewer_tui_uses_configured_embeddings(
    tmp_path,
    monkeypatch,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    session = ReviewerSession(storage)

    calls = []

    original_current_view = session.current_view

    def current_view(
        *,
        embedding_names,
        speaker_limit=None,
    ):
        calls.append(
            (
                embedding_names,
                speaker_limit,
            )
        )

        return original_current_view(
            embedding_names=(),
            speaker_limit=speaker_limit,
        )

    monkeypatch.setattr(
        session,
        "current_view",
        current_view,
    )

    app = ReviewerTUI(
        session,
        embedding_names=("custom_speaker",),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        await pilot.pause()

        assert calls
        assert calls[-1] == (
            ("custom_speaker",),
            3,
        )


@pytest.mark.asyncio
async def test_reviewer_tui_plays_preferred_audio(
    tmp_path,
    monkeypatch,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    session = ReviewerSession(storage)

    calls = []

    def fake_play(
        dataset,
        turn,
        *,
        blocking,
    ):
        calls.append(
            (
                dataset,
                turn["id"],
                blocking,
            )
        )

        return (
            "review",
            dataset / "review.wav",
        )

    monkeypatch.setattr(
        "voice_dataset.reviewer_tui."
        "play_preferred_review_audio",
        fake_play,
    )

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        await pilot.press("space")
        await pilot.pause()

        assert calls == [
            (
                storage.root,
                "turn_000001",
                False,
            )
        ]


@pytest.mark.asyncio
async def test_reviewer_tui_stops_playback_before_navigation(
    tmp_path,
    monkeypatch,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )
    add_turn(
        storage,
        "turn_000002",
        source_start=2.0,
    )

    session = ReviewerSession(storage)

    calls = []

    monkeypatch.setattr(
        "voice_dataset.reviewer_tui.stop",
        lambda: calls.append("stop"),
    )

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        await pilot.press("right")
        await pilot.pause()

        assert calls == ["stop"]
        assert (
            session.current_turn_id
            == "turn_000002"
        )


@pytest.mark.asyncio
async def test_reviewer_tui_plays_raw_audio(
    tmp_path,
    monkeypatch,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    session = ReviewerSession(storage)

    raw = {
        "path": "turns/turn_000001/raw.wav",
    }

    monkeypatch.setattr(
        "voice_dataset.reviewer_tui."
        "raw_representation",
        lambda turn: raw,
    )

    calls = []

    def fake_play_representation(
        dataset,
        representation,
        *,
        blocking,
    ):
        calls.append(
            (
                dataset,
                representation,
                blocking,
            )
        )

        return dataset / "raw.wav"

    monkeypatch.setattr(
        "voice_dataset.reviewer_tui."
        "play_representation",
        fake_play_representation,
    )

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        await pilot.press("r")
        await pilot.pause()

        assert calls == [
            (
                storage.root,
                raw,
                False,
            )
        ]


@pytest.mark.asyncio
async def test_reviewer_tui_plays_context(
    tmp_path,
    monkeypatch,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    storage.sources.append(
        {
            "schema_version": 1,
            "record_type": "source",
            "id": "source_001",
            "metadata": {},
        }
    )

    session = ReviewerSession(storage)

    calls = []

    def fake_play_context(
        source,
        turn,
        *,
        padding,
        blocking,
    ):
        calls.append(
            (
                source["id"],
                turn["id"],
                padding,
                blocking,
            )
        )

        return (
            "source",
            storage.root / "source.wav",
            0.5,
            3.5,
        )

    monkeypatch.setattr(
        "voice_dataset.reviewer_tui."
        "play_turn_context",
        fake_play_context,
    )

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.5,
    )

    async with app.run_test() as pilot:
        await pilot.press("c")
        await pilot.pause()

        assert calls == [
            (
                "source_001",
                "turn_000001",
                2.5,
                False,
            )
        ]


@pytest.mark.asyncio
async def test_reviewer_tui_opens_voice_picker(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    session = ReviewerSession(storage)

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        assert len(app.screen_stack) == 1

        await pilot.press("v")
        await pilot.pause()

        assert len(app.screen_stack) == 2


@pytest.mark.asyncio
async def test_voice_picker_shows_candidates_and_voices(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    storage.voices.append(
        {
            "schema_version": 1,
            "record_type": "voice_profile",
            "id": "voice_001",
            "character": "Silco",
            "language": "en",
            "aliases": [],
            "ignored": False,
            "notes": None,
            "metadata": {},
        }
    )

    session = ReviewerSession(storage)

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        await pilot.press("v")
        await pilot.pause()

        screen = app.screen

        content = screen.query_one(
            "#voice-content",
            Static,
        )

        rendered = str(content.render())

        assert "All voices" in rendered
        assert "voice_001" in rendered
        assert "Silco" in rendered


@pytest.mark.asyncio
async def test_voice_picker_assigns_selected_voice(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    storage.voices.append(
        {
            "schema_version": 1,
            "record_type": "voice_profile",
            "id": "voice_001",
            "character": "Silco",
            "language": "en",
            "aliases": [],
            "ignored": False,
            "notes": None,
            "metadata": {},
        }
    )

    session = ReviewerSession(storage)

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        await pilot.press("v")
        await pilot.pause()

        await pilot.press("enter")
        await pilot.pause()

        turn = storage.get_turn(
            "turn_000001"
        )

        assert turn is not None
        assert turn["assignment"] == {
            "status": "assigned",
            "voice_id": "voice_001",
            "method": "manual",
            "confidence": None,
        }

        assert len(app.screen_stack) == 1


@pytest.mark.asyncio
async def test_voice_picker_creates_and_assigns_new_voice(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    session = ReviewerSession(storage)

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        await pilot.press("v")
        await pilot.pause()

        await pilot.press("n")
        await pilot.pause()

        assert len(app.screen_stack) == 3

        character = app.screen.query_one(
            "#new-voice-character",
            Input,
        )
        language = app.screen.query_one(
            "#new-voice-language",
            Input,
        )

        character.value = "Vander"
        language.value = "en"

        character.focus()

        await pilot.press("enter")
        await pilot.pause()

        assert len(app.screen_stack) == 3
        assert language.has_focus
        assert storage.voices.load() == []

        await pilot.press("enter")
        await pilot.pause()

        await pilot.press("enter")
        await pilot.pause()

        voices = storage.voices.load()

        assert len(voices) == 1
        assert voices[0]["character"] == "Vander"
        assert voices[0]["language"] == "en"

        turn = storage.get_turn(
            "turn_000001"
        )

        assert turn is not None
        assert turn["assignment"]["status"] == "assigned"
        assert (
            turn["assignment"]["voice_id"]
            == voices[0]["id"]
        )

        assert len(app.screen_stack) == 1


@pytest.mark.asyncio
async def test_reviewer_tui_marks_turn_reviewed(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    session = ReviewerSession(storage)

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        await pilot.press("a")
        await pilot.pause()

        turn = storage.get_turn(
            "turn_000001"
        )

        assert turn is not None
        assert turn["review"]["status"] == "reviewed"


@pytest.mark.asyncio
async def test_reviewer_tui_marks_turn_pending(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    session = ReviewerSession(storage)

    session.mark_reviewed()

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        await pilot.press("x")
        await pilot.pause()

        turn = storage.get_turn(
            "turn_000001"
        )

        assert turn is not None
        assert turn["review"]["status"] == "pending"


@pytest.mark.asyncio
async def test_reviewer_tui_marks_voice_unknown(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    storage.voices.append(
        {
            "schema_version": 1,
            "record_type": "voice_profile",
            "id": "voice_001",
            "character": "Silco",
            "language": "en",
            "aliases": [],
            "ignored": False,
            "notes": None,
            "metadata": {},
        }
    )

    session = ReviewerSession(storage)
    session.assign_voice("voice_001")

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        await pilot.press("u")
        await pilot.pause()

        turn = storage.get_turn(
            "turn_000001"
        )

        assert turn is not None
        assert turn["assignment"]["status"] == "unknown"


@pytest.mark.asyncio
async def test_reviewer_tui_ignores_turn(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    session = ReviewerSession(storage)

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        await pilot.press("i")
        await pilot.pause()

        turn = storage.get_turn(
            "turn_000001"
        )

        assert turn is not None
        assert turn["assignment"]["status"] == "ignore"


@pytest.mark.asyncio
async def test_reviewer_tui_edits_transcript(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    session = ReviewerSession(storage)

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        await pilot.press("t")
        await pilot.pause()

        assert len(app.screen_stack) == 2

        field = app.screen.query_one(
            "#edit-value",
            Input,
        )

        assert field.value == "turn_000001"

        field.value = "Corrected transcript"

        await pilot.press("enter")
        await pilot.pause()

        turn = storage.get_turn(
            "turn_000001"
        )

        assert turn is not None
        assert (
            turn["transcript"]
            == "Corrected transcript"
        )

        assert len(app.screen_stack) == 1


@pytest.mark.asyncio
async def test_reviewer_tui_edits_language(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    session = ReviewerSession(storage)

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        await pilot.press("l")
        await pilot.pause()

        field = app.screen.query_one(
            "#edit-value",
            Input,
        )

        assert field.value == "en"

        field.value = "de"

        await pilot.press("enter")
        await pilot.pause()

        turn = storage.get_turn(
            "turn_000001"
        )

        assert turn is not None
        assert turn["language"] == "de"

        assert len(app.screen_stack) == 1


@pytest.mark.asyncio
async def test_reviewer_tui_marks_boundary_complete(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    storage.turns.update(
        "turn_000001",
        lambda turn: {
            **turn,
            "metadata": {
                **(turn.get("metadata") or {}),
                "boundary_evidence": {
                    "near_source_start": True,
                    "near_source_end": False,
                },
            },
        },
    )

    session = ReviewerSession(storage)

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        await pilot.press("k")
        await pilot.pause()

        turn = storage.get_turn(
            "turn_000001"
        )

        assert turn is not None
        assert (
            turn["review"]["boundary"]["status"]
            == "complete"
        )


@pytest.mark.asyncio
async def test_reviewer_tui_marks_boundary_clipped(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    storage.turns.update(
        "turn_000001",
        lambda turn: {
            **turn,
            "metadata": {
                **(turn.get("metadata") or {}),
                "boundary_evidence": {
                    "near_source_start": True,
                    "near_source_end": False,
                },
            },
        },
    )

    session = ReviewerSession(storage)

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        await pilot.press("d")
        await pilot.pause()

        turn = storage.get_turn(
            "turn_000001"
        )

        assert turn is not None
        assert (
            turn["review"]["boundary"]["status"]
            == "clipped"
        )

import asyncio

import pytest

import threading

from rich.text import Text

from voice_dataset.reviewer_session import ReviewerSession
from voice_dataset.reviewer_tui import (
    build_split_turn_candidates,
    ConfirmMergeScreen,
    ConfirmRecoveryScreen,
    EditValueScreen,
    NewVoiceScreen,
    ReviewerTUI,
    SplitTurnCandidate,
    SplitTurnScreen,
    VoicePickerScreen,
    TrimTurnRequest,
    TrimTurnScreen,
    UpdatingTurnScreen,
)
from voice_dataset.storage import DatasetStorage
from voice_dataset.schema import VoiceProfile

from textual.widgets import (
    Footer,
    Input,
    OptionList,
    Static,
)


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

    storage.update_turn(
        "turn_000001",
        lambda turn: {
            **turn,
            "representations": {
                "review": {
                    "path": (
                        "turns/turn_000001/"
                        "review.wav"
                    ),
                    "purposes": ["review"],
                },
            },
        },
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
            "representations": {
                "center": {
                    "path": "audio/source_001/center.wav",
                    "purposes": ["context"],
                },
            },
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

        options = screen.query_one(
            "#voice-options",
            OptionList,
        )

        assert options.option_count == 1

        option = options.get_option_at_index(0)

        rendered = str(option.prompt)

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

        options = app.screen.query_one(
            "#voice-options",
            OptionList,
        )

        assert options.highlighted is None

        options.highlighted = 0

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
async def test_reviewer_tui_rejects_turn(
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

        assert turn["curation"] == {
            "status": "rejected",
        }

        assert turn["assignment"]["status"] == "unknown"


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


@pytest.mark.asyncio
async def test_reviewer_tui_does_not_open_alignment_recovery_without_suggestion(
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
        await pilot.press("f")
        await pilot.pause()

        assert len(app.screen_stack) == 1


@pytest.mark.asyncio
async def test_reviewer_tui_accepts_alignment_recovery(
    tmp_path,
    monkeypatch,
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
                "alignment_evidence": {
                    "status": "review",
                    "issue_word_indices": [
                        2,
                        3,
                    ],
                    "recovery": {
                        "status": "suggested",
                        "source_start": 1.1,
                        "source_end": 1.9,
                        "speaker": "SPEAKER_04",
                        "region_ids": [
                            "region_000020",
                        ],
                        "matched_token_count": 4,
                        "candidate_token_count": 5,
                    },
                },
            },
        },
    )

    session = ReviewerSession(storage)

    calls = []

    def accept_alignment_recovery():
        calls.append(
            session.current_turn_id
        )

        return storage.get_turn(
            "turn_000001"
        )

    monkeypatch.setattr(
        session,
        "accept_alignment_recovery",
        accept_alignment_recovery,
    )

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        await pilot.press("f")
        await pilot.pause()

        assert len(app.screen_stack) == 2

        content = app.screen.query_one(
            "#recovery-content",
            Static,
        )

        rendered = str(content.render())

        assert "Alignment Recovery" in rendered
        assert "1.100" in rendered
        assert "1.900" in rendered
        assert "SPEAKER_04" in rendered
        assert "region_000020" in rendered
        assert "4 / 5" in rendered

        await pilot.press("enter")
        await pilot.pause()

        assert calls == [
            "turn_000001",
        ]

        assert len(app.screen_stack) == 1


@pytest.mark.asyncio
async def test_reviewer_tui_does_not_open_edge_recovery_without_suggestion(
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
        await pilot.press("e")
        await pilot.pause()

        assert len(app.screen_stack) == 1


@pytest.mark.asyncio
async def test_reviewer_tui_accepts_edge_recovery(
    tmp_path,
    monkeypatch,
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
                "edge_evidence": {
                    "status": "suggested",
                    "method": (
                        "community_whisper_"
                        "edge_extension"
                    ),
                    "edge": "end",
                    "source_end": 2.4,
                    "region_id": "region_000020",
                    "speaker": "SPEAKER_04",
                    "candidate_token": "kid",
                    "whisper_token": "kiddo",
                    "candidate_text": (
                        "It gets easier kid"
                    ),
                    "whisper_text": (
                        "It gets easier kiddo"
                    ),
                },
            },
        },
    )

    session = ReviewerSession(storage)

    calls = []

    def accept_edge_recovery():
        calls.append(
            session.current_turn_id
        )

        return storage.get_turn(
            "turn_000001"
        )

    monkeypatch.setattr(
        session,
        "accept_edge_recovery",
        accept_edge_recovery,
    )

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        await pilot.press("e")
        await pilot.pause()

        assert len(app.screen_stack) == 2

        content = app.screen.query_one(
            "#recovery-content",
            Static,
        )

        rendered = str(content.render())

        assert "Edge Recovery" in rendered
        assert "end" in rendered
        assert "2.400" in rendered
        assert "SPEAKER_04" in rendered
        assert "region_000020" in rendered
        assert "kid" in rendered
        assert "kiddo" in rendered

        await pilot.press("enter")
        await pilot.pause()

        assert calls == [
            "turn_000001",
        ]

        assert len(app.screen_stack) == 1


@pytest.mark.asyncio
async def test_reviewer_tui_does_not_open_merge_without_next_turn(
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
        await pilot.press("m")
        await pilot.pause()

        assert len(app.screen_stack) == 1


@pytest.mark.asyncio
async def test_reviewer_tui_merges_with_next_turn(
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

    storage.update_turn(
        "turn_000001",
        lambda turn: {
            **turn,
            "transcript": "First line",
        },
    )

    storage.update_turn(
        "turn_000002",
        lambda turn: {
            **turn,
            "transcript": "Second line",
        },
    )

    session = ReviewerSession(storage)

    calls = []

    def merge(turn_ids):
        calls.append(
            list(turn_ids)
        )

        return storage.get_turn(
            "turn_000001"
        )

    monkeypatch.setattr(
        session,
        "merge",
        merge,
    )

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        await pilot.press("m")
        await pilot.pause()

        assert len(app.screen_stack) == 2

        content = app.screen.query_one(
            "#merge-content",
            Static,
        )

        rendered = str(content.render())

        title = app.screen.query_one(
            "#merge-title",
            Static,
        )

        assert "Merge Turns" in str(
            title.render()
        )
        assert "turn_000001" in rendered
        assert "turn_000002" in rendered
        assert "First line" in rendered
        assert "Second line" in rendered

        await pilot.press("enter")
        await pilot.pause()

        assert calls == [
            [
                "turn_000001",
                "turn_000002",
            ],
        ]

        assert len(app.screen_stack) == 1


@pytest.mark.asyncio
async def test_reviewer_tui_merge_uses_canonical_next_turn_when_filtered(
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
    add_turn(
        storage,
        "turn_000003",
        source_start=3.0,
    )

    def set_auto_status(
        turn_id: str,
        status: str,
    ) -> None:
        def update(turn):
            metadata = dict(
                turn.get("metadata") or {}
            )
            automatic_pipeline = dict(
                metadata.get(
                    "automatic_pipeline"
                )
                or {}
            )

            automatic_pipeline["status"] = status
            metadata["automatic_pipeline"] = (
                automatic_pipeline
            )
            turn["metadata"] = metadata

            return turn

        storage.update_turn(
            turn_id,
            update,
        )

    set_auto_status(
        "turn_000001",
        "review",
    )
    set_auto_status(
        "turn_000002",
        "accepted",
    )
    set_auto_status(
        "turn_000003",
        "review",
    )

    storage.update_turn(
        "turn_000001",
        lambda turn: {
            **turn,
            "transcript": "First visible turn",
        },
    )

    storage.update_turn(
        "turn_000002",
        lambda turn: {
            **turn,
            "transcript": "Hidden canonical next turn",
        },
    )

    session = ReviewerSession(
        storage,
        auto_review_only=True,
    )

    assert session.turn_ids == (
        "turn_000001",
        "turn_000003",
    )

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        await pilot.press("m")
        await pilot.pause()

        assert len(app.screen_stack) == 2

        content = app.screen.query_one(
            "#merge-content",
            Static,
        )

        rendered = str(content.render())

        assert "turn_000001" in rendered
        assert "turn_000002" in rendered
        assert "Hidden canonical next turn" in rendered

        assert "turn_000003" not in rendered


@pytest.mark.asyncio
async def test_reviewer_tui_opens_manual_split_without_valid_region(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    storage.update_turn(
        "turn_000001",
        lambda turn: {
            **turn,
            "source_regions": [
                "region_000001",
            ],
        },
    )

    session = ReviewerSession(storage)

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        await pilot.press("/")
        await pilot.pause()

        assert len(app.screen_stack) == 2
        assert isinstance(
            app.screen,
            SplitTurnScreen,
        )

        content = app.screen.query_one(
            "#split-candidate-info",
            Static,
        )

        rendered = str(
            content.render()
        )

        assert "Manual split" in rendered


@pytest.mark.asyncio
async def test_reviewer_tui_splits_after_selected_region(
    tmp_path,
    monkeypatch,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    storage.update_turn(
        "turn_000001",
        lambda turn: {
            **turn,
            "source_regions": [
                "region_000001",
                "region_000002",
                "region_000003",
            ],
        },
    )

    session = ReviewerSession(storage)

    calls = []

    def split(
        *,
        after_region_id,
        split_time=None,
    ):
        calls.append(
            (
                after_region_id,
                split_time,
            )
        )

        turn = storage.get_turn(
            "turn_000001"
        )

        assert turn is not None

        return turn, turn

    monkeypatch.setattr(
        session,
        "split",
        split,
    )

    def fake_build_split_turn_candidates(
        storage_arg,
        turn,
        *,
        minimum_stable_duration=0.2,
    ):
        assert storage_arg is storage
        assert turn["id"] == "turn_000001"

        return (
            SplitTurnCandidate(
                region_id="region_000001",
                left_region_id="region_000001",
                left_start=1.0,
                left_end=1.3,
                left_speaker="SPEAKER_01",
                left_text="one",
                right_region_id="region_000002",
                right_start=1.3,
                right_end=1.6,
                right_speaker="SPEAKER_02",
                right_text="two",
            ),
            SplitTurnCandidate(
                region_id="region_000002",
                left_region_id="region_000002",
                left_start=1.3,
                left_end=1.6,
                left_speaker="SPEAKER_02",
                left_text="two",
                right_region_id="region_000003",
                right_start=1.6,
                right_end=2.0,
                right_speaker="SPEAKER_03",
                right_text="three",
            ),
        )

    monkeypatch.setattr(
        "voice_dataset.reviewer_tui."
        "build_split_turn_candidates",
        fake_build_split_turn_candidates,
    )

    monkeypatch.setattr(
        "voice_dataset.reviewer_tui."
        "build_split_word_hints",
        lambda storage_arg, turn, **kwargs: (),
    )

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        await pilot.press("/")
        await pilot.pause()

        assert len(app.screen_stack) == 2

        title = app.screen.query_one(
            "#split-title",
            Static,
        )

        assert "Split Turn" in str(
            title.render()
        )

        options = app.screen.query_one(
            "#split-options",
            OptionList,
        )

        option_ids = [
            str(option.id)
            for option in options.options
        ]

        assert option_ids == [
            "region_000001",
            "region_000002",
        ]

        await pilot.press("enter")
        await pilot.pause()

        assert calls == [
            (
                "region_000001",
                pytest.approx(1.3),
            ),
        ]

        assert len(app.screen_stack) == 1


@pytest.mark.asyncio
async def test_reviewer_tui_hides_previous_on_first_turn(
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
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test():
        assert (
            app.check_action(
                "previous_turn",
                (),
            )
            is False
        )

        assert (
            app.check_action(
                "next_turn",
                (),
            )
            is True
        )


@pytest.mark.asyncio
async def test_reviewer_tui_hides_next_on_last_turn(
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
    session.next()

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test():
        assert (
            app.check_action(
                "previous_turn",
                (),
            )
            is True
        )

        assert (
            app.check_action(
                "next_turn",
                (),
            )
            is False
        )


@pytest.mark.asyncio
async def test_reviewer_tui_hides_recovery_actions_without_suggestion(
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

    async with app.run_test():
        assert (
            app.check_action(
                "accept_alignment_recovery",
                (),
            )
            is False
        )

        assert (
            app.check_action(
                "accept_edge_recovery",
                (),
            )
            is False
        )


@pytest.mark.asyncio
async def test_reviewer_tui_hides_merge_on_last_turn(
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

    async with app.run_test():
        assert (
            app.check_action(
                "merge_with_next",
                (),
            )
            is False
        )


@pytest.mark.asyncio
async def test_reviewer_tui_shows_manual_split_without_split_point(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    storage.update_turn(
        "turn_000001",
        lambda turn: {
            **turn,
            "source_regions": [
                "region_000001",
            ],
        },
    )

    session = ReviewerSession(storage)

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test():
        assert (
            app.check_action(
                "split_turn",
                (),
            )
            is True
        )


@pytest.mark.asyncio
async def test_reviewer_tui_shows_merge_when_next_turn_exists(
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
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test():
        assert (
            app.check_action(
                "merge_with_next",
                (),
            )
            is True
        )


@pytest.mark.asyncio
async def test_reviewer_tui_shows_split_with_valid_split_point(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    storage.update_turn(
        "turn_000001",
        lambda turn: {
            **turn,
            "source_regions": [
                "region_000001",
                "region_000002",
            ],
        },
    )

    session = ReviewerSession(storage)

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test():
        assert (
            app.check_action(
                "split_turn",
                (),
            )
            is True
        )


@pytest.mark.asyncio
async def test_voice_picker_hides_assign_shortcut_without_voices(
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

        shortcuts = str(
            app.screen.query_one(
                "#voice-shortcuts",
                Static,
            ).render()
        )

        assert "Enter Assign" not in shortcuts
        assert "n New Voice" in shortcuts
        assert "Esc Cancel" in shortcuts


def test_voice_picker_shows_assign_with_voice():
    screen = VoicePickerScreen(
        voices=[
            {
                "id": "voice_001",
                "character": "Vander",
            },
        ],
        candidates=(),
    )

    assert (
        screen.check_action(
            "assign",
            (),
        )
        is True
    )


@pytest.mark.asyncio
async def test_voice_picker_has_contextual_footer(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    storage.add_voice(
        VoiceProfile(
            id="voice_001",
            character="Vander",
            language="en",
            aliases=[],
            ignored=False,
            notes=None,
        )
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

        assert isinstance(
            app.screen,
            VoicePickerScreen,
        )

        assert list(
            app.screen.query(Footer)
        ) == []

        shortcuts = str(
            app.screen.query_one(
                "#voice-shortcuts",
                Static,
            ).render()
        )

        assert "Enter Assign" in shortcuts
        assert "n New Voice" in shortcuts
        assert "Esc Cancel" in shortcuts


@pytest.mark.asyncio
async def test_new_voice_screen_has_contextual_footer(
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
        app.push_screen(
            NewVoiceScreen()
        )
        await pilot.pause()

        assert isinstance(
            app.screen,
            NewVoiceScreen,
        )

        assert list(
            app.screen.query(Footer)
        ) == []

        assert app.screen.query_one(
            "#new-voice-shortcuts",
            Static,
        ) is not None


@pytest.mark.asyncio
async def test_recovery_screen_has_contextual_footer(
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
        app.push_screen(
            ConfirmRecoveryScreen(
                content="Recovery",
            )
        )
        await pilot.pause()

        assert isinstance(
            app.screen,
            ConfirmRecoveryScreen,
        )

        assert app.screen.query_one(
            Footer
        ) is not None


@pytest.mark.asyncio
async def test_merge_screen_has_contextual_footer(
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
        first = storage.get_turn(
            "turn_000001"
        )

        assert first is not None

        second = {
            **first,
            "id": "turn_000002",
            "source_start": 2.0,
            "source_end": 3.0,
            "transcript": "Second",
        }

        app.push_screen(
            ConfirmMergeScreen(
                turns=(
                    first,
                    second,
                ),
            )
        )
        await pilot.pause()

        assert isinstance(
            app.screen,
            ConfirmMergeScreen,
        )

        assert list(
            app.screen.query(Footer)
        ) == []

        shortcuts = app.screen.query_one(
            "#merge-shortcuts",
            Static,
        )

        rendered = str(
            shortcuts.render()
        )

        assert "+ Add" in rendered
        assert "- Remove" in rendered
        assert "p Play Turn" in rendered
        assert "Space Play All" in rendered
        assert "Enter Merge" in rendered


@pytest.mark.asyncio
async def test_split_screen_has_contextual_shortcuts(
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
        app.push_screen(
            SplitTurnScreen(
                turn_id="turn_000001",
                candidates=(
                    SplitTurnCandidate(
                        region_id="region_000001",
                        left_region_id="region_000001",
                        left_start=1.0,
                        left_end=1.4,
                        left_speaker="SPEAKER_01",
                        left_text="left",
                        right_region_id="region_000002",
                        right_start=1.4,
                        right_end=2.0,
                        right_speaker="SPEAKER_02",
                        right_text="right",
                    ),
                ),
            )
        )
        await pilot.pause()

        assert isinstance(
            app.screen,
            SplitTurnScreen,
        )

        shortcuts = app.screen.query_one(
            "#split-shortcuts",
            Static,
        )

        rendered = str(shortcuts.render())

        assert "Space Around" in rendered
        assert ", Before" in rendered
        assert ". After" in rendered
        assert "Enter Split" in rendered
        assert "Esc Cancel" in rendered


@pytest.mark.asyncio
async def test_edit_value_screen_has_contextual_footer(
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
        app.push_screen(
            EditValueScreen(
                title="Edit Transcript",
                value="hello",
            )
        )
        await pilot.pause()

        assert isinstance(
            app.screen,
            EditValueScreen,
        )

        assert app.screen.query_one(
            Footer
        ) is not None


@pytest.mark.asyncio
async def test_reviewer_tui_hides_current_review_state_action(
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

    async with app.run_test():
        assert (
            app.check_action(
                "mark_pending",
                (),
            )
            is False
        )

        assert (
            app.check_action(
                "mark_reviewed",
                (),
            )
            is True
        )


@pytest.mark.asyncio
async def test_reviewer_tui_hides_reviewed_after_marking_reviewed(
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

    async with app.run_test():
        assert (
            app.check_action(
                "mark_reviewed",
                (),
            )
            is False
        )

        assert (
            app.check_action(
                "mark_pending",
                (),
            )
            is True
        )


@pytest.mark.asyncio
async def test_reviewer_tui_hides_unknown_when_already_unknown(
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

    async with app.run_test():
        assert (
            app.check_action(
                "mark_unknown",
                (),
            )
            is False
        )


@pytest.mark.asyncio
async def test_reviewer_tui_hides_boundary_complete_when_already_complete(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    storage.update_turn(
        "turn_000001",
        lambda turn: {
            **turn,
            "review": {
                **(turn.get("review") or {}),
                "boundary": {
                    "status": "complete",
                },
            },
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

    async with app.run_test():
        assert (
            app.check_action(
                "mark_boundary_complete",
                (),
            )
            is False
        )

        assert (
            app.check_action(
                "mark_boundary_clipped",
                (),
            )
            is True
        )


@pytest.mark.asyncio
async def test_reviewer_tui_hides_boundary_clipped_when_already_clipped(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    storage.update_turn(
        "turn_000001",
        lambda turn: {
            **turn,
            "review": {
                **(turn.get("review") or {}),
                "boundary": {
                    "status": "clipped",
                },
            },
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

    async with app.run_test():
        assert (
            app.check_action(
                "mark_boundary_clipped",
                (),
            )
            is False
        )

        assert (
            app.check_action(
                "mark_boundary_complete",
                (),
            )
            is True
        )


@pytest.mark.asyncio
async def test_reviewer_tui_hides_stop_when_not_playing(
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

    monkeypatch.setattr(
        "voice_dataset.reviewer_tui.is_playing",
        lambda: False,
    )

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test():
        assert (
            app.check_action(
                "stop_playback",
                (),
            )
            is False
        )


@pytest.mark.asyncio
async def test_reviewer_tui_shows_stop_when_playing(
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

    monkeypatch.setattr(
        "voice_dataset.reviewer_tui.is_playing",
        lambda: True,
    )

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test():
        assert (
            app.check_action(
                "stop_playback",
                (),
            )
            is True
        )


@pytest.mark.asyncio
async def test_reviewer_tui_refreshes_bindings_when_playback_state_changes(
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

    state = {
        "playing": False,
    }

    monkeypatch.setattr(
        "voice_dataset.reviewer_tui.is_playing",
        lambda: state["playing"],
    )

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    calls = []

    original_refresh_bindings = (
        app.refresh_bindings
    )

    def refresh_bindings():
        calls.append(
            state["playing"]
        )

        return original_refresh_bindings()

    monkeypatch.setattr(
        app,
        "refresh_bindings",
        refresh_bindings,
    )

    async with app.run_test():
        calls.clear()

        state["playing"] = True

        await asyncio.sleep(0.2)

        assert True in calls

        calls.clear()

        state["playing"] = False

        await asyncio.sleep(0.2)

        assert False in calls


@pytest.mark.asyncio
async def test_reviewer_tui_refreshes_bindings_immediately_on_stop(
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

    state = {
        "playing": True,
    }

    monkeypatch.setattr(
        "voice_dataset.reviewer_tui.is_playing",
        lambda: state["playing"],
    )

    def fake_stop():
        state["playing"] = False

    monkeypatch.setattr(
        "voice_dataset.reviewer_tui.stop",
        fake_stop,
    )

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test():
        app._last_playback_active = True

        calls = []

        original_refresh_bindings = (
            app.refresh_bindings
        )

        def refresh_bindings():
            calls.append(
                state["playing"]
            )
            return original_refresh_bindings()

        monkeypatch.setattr(
            app,
            "refresh_bindings",
            refresh_bindings,
        )

        app.action_stop_playback()

        assert (
            app._last_playback_active
            is False
        )
        assert False in calls


@pytest.mark.asyncio
async def test_reviewer_tui_refreshes_bindings_immediately_on_play(
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

    state = {
        "playing": False,
    }

    monkeypatch.setattr(
        "voice_dataset.reviewer_tui.is_playing",
        lambda: state["playing"],
    )

    def fake_play(
        dataset,
        turn,
        *,
        blocking,
    ):
        state["playing"] = True

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

    async with app.run_test():
        app._last_playback_active = False

        calls = []

        original_refresh_bindings = (
            app.refresh_bindings
        )

        def refresh_bindings():
            calls.append(
                state["playing"]
            )
            return original_refresh_bindings()

        monkeypatch.setattr(
            app,
            "refresh_bindings",
            refresh_bindings,
        )

        app.action_play_preferred()

        assert (
            app._last_playback_active
            is True
        )
        assert True in calls


@pytest.mark.asyncio
async def test_reviewer_tui_refreshes_bindings_immediately_on_raw_play(
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

    state = {"playing": False}

    monkeypatch.setattr(
        "voice_dataset.reviewer_tui.is_playing",
        lambda: state["playing"],
    )

    monkeypatch.setattr(
        "voice_dataset.reviewer_tui.raw_representation",
        lambda turn: {"path": "raw.wav"},
    )

    def fake_play(
        dataset,
        representation,
        *,
        blocking,
    ):
        state["playing"] = True
        return dataset / "raw.wav"

    monkeypatch.setattr(
        "voice_dataset.reviewer_tui.play_representation",
        fake_play,
    )

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test():
        app._last_playback_active = False

        app.action_play_raw()

        assert (
            app._last_playback_active
            is True
        )


@pytest.mark.asyncio
async def test_reviewer_tui_refreshes_bindings_immediately_on_context_play(
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

    state = {"playing": False}

    monkeypatch.setattr(
        "voice_dataset.reviewer_tui.is_playing",
        lambda: state["playing"],
    )

    def fake_context(
        source,
        turn,
        *,
        padding,
        blocking,
    ):
        state["playing"] = True

        return (
            "center",
            tmp_path / "center.wav",
            0.0,
            2.0,
        )

    monkeypatch.setattr(
        "voice_dataset.reviewer_tui.play_turn_context",
        fake_context,
    )

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test():
        app._last_playback_active = False

        app.action_play_context()

        assert (
            app._last_playback_active
            is True
        )


@pytest.mark.asyncio
async def test_reviewer_tui_hides_play_without_review_representation(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    storage.update_turn(
        "turn_000001",
        lambda turn: {
            **turn,
            "representations": {},
        },
    )

    session = ReviewerSession(storage)

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test():
        assert (
            app.check_action(
                "play_preferred",
                (),
            )
            is False
        )


@pytest.mark.asyncio
async def test_reviewer_tui_shows_play_with_review_representation(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    storage.update_turn(
        "turn_000001",
        lambda turn: {
            **turn,
            "representations": {
                "review": {
                    "path": (
                        "turns/turn_000001/"
                        "review.wav"
                    ),
                    "purposes": ["review"],
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

    async with app.run_test():
        assert (
            app.check_action(
                "play_preferred",
                (),
            )
            is True
        )


@pytest.mark.asyncio
async def test_reviewer_tui_hides_raw_without_raw_representation(
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

    monkeypatch.setattr(
        "voice_dataset.reviewer_tui.raw_representation",
        lambda turn: (_ for _ in ()).throw(
            ValueError("no raw representation")
        ),
    )

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test():
        assert (
            app.check_action(
                "play_raw",
                (),
            )
            is False
        )


@pytest.mark.asyncio
async def test_reviewer_tui_shows_raw_with_raw_representation(
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

    monkeypatch.setattr(
        "voice_dataset.reviewer_tui.raw_representation",
        lambda turn: {
            "path": "turns/turn_000001/raw.wav",
        },
    )

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test():
        assert (
            app.check_action(
                "play_raw",
                (),
            )
            is True
        )


@pytest.mark.asyncio
async def test_reviewer_tui_hides_context_without_source(
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

    async with app.run_test():
        assert (
            app.check_action(
                "play_context",
                (),
            )
            is False
        )


@pytest.mark.asyncio
async def test_reviewer_tui_shows_context_with_context_representation(
    tmp_path,
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
            "representations": {
                "center": {
                    "path": "audio/source/center.wav",
                    "purposes": ["context"],
                },
            },
            "metadata": {},
        }
    )

    session = ReviewerSession(storage)

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test():
        assert (
            app.check_action(
                "play_context",
                (),
            )
            is True
        )


@pytest.mark.asyncio
async def test_reviewer_tui_uses_two_shortcut_rows(
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

    async with app.run_test():
        assert list(
            app.screen.query(Footer)
        ) == []

        assert app.screen.query_one(
            "#shortcuts-primary",
            Static,
        ) is not None

        assert app.screen.query_one(
            "#shortcuts-secondary",
            Static,
        ) is not None


@pytest.mark.asyncio
async def test_reviewer_tui_renders_contextual_shortcut_rows(
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

    storage.update_turn(
        "turn_000001",
        lambda turn: {
            **turn,
            "representations": {
                "review": {
                    "path": (
                        "turns/turn_000001/"
                        "review.wav"
                    ),
                    "purposes": ["review"],
                },
            },
        },
    )

    storage.sources.append(
        {
            "schema_version": 1,
            "record_type": "source",
            "id": "source_001",
            "representations": {
                "center": {
                    "path": "audio/source_001/center.wav",
                    "purposes": ["context"],
                },
            },
            "metadata": {},
        }
    )

    session = ReviewerSession(storage)

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test():
        primary = str(
            app.screen.query_one(
                "#shortcuts-primary",
                Static,
            ).render()
        )

        secondary = str(
            app.screen.query_one(
                "#shortcuts-secondary",
                Static,
            ).render()
        )

        assert "Next" in primary
        assert "Play" in primary
        assert "Context" in primary
        assert "Help" in primary
        assert "Quit" in primary

        assert "Previous" not in primary
        assert "Stop" not in primary

        assert "Voice" in secondary
        assert "Reviewed" in secondary
        assert "Transcript" in secondary
        assert "Language" in secondary


@pytest.mark.asyncio
async def test_reviewer_tui_shows_assigned_voice_character(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    storage.add_voice(
        VoiceProfile(
            id="voice_001",
            character="Vander",
            language="en",
            aliases=[],
            ignored=False,
            notes=None,
        )
    )

    storage.update_turn(
        "turn_000001",
        lambda turn: {
            **turn,
            "assignment": {
                "status": "assigned",
                "voice_id": "voice_001",
                "method": "manual",
                "confidence": None,
            },
        },
    )

    session = ReviewerSession(storage)

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test():
        evidence = str(
            app.screen.query_one(
                "#evidence-content",
                Static,
            ).render()
        )

        assert "voice_001" in evidence
        assert "Vander" in evidence


@pytest.mark.asyncio
async def test_new_voice_screen_shows_enter_next_for_character(
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
        app.push_screen(
            NewVoiceScreen()
        )
        await pilot.pause()

        footer = str(
            app.screen.query_one(
                "#new-voice-shortcuts",
                Static,
            ).render()
        )

        assert "Enter Next" in footer
        assert "Esc Cancel" in footer


@pytest.mark.asyncio
async def test_new_voice_screen_shows_enter_create_for_language(
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
        app.push_screen(
            NewVoiceScreen()
        )
        await pilot.pause()

        await pilot.press("enter")
        await pilot.pause()

        footer = str(
            app.screen.query_one(
                "#new-voice-shortcuts",
                Static,
            ).render()
        )

        assert "Enter Create" in footer
        assert "Esc Cancel" in footer


@pytest.mark.asyncio
async def test_reviewer_tui_hides_boundary_actions_without_evidence(
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

    async with app.run_test():
        assert (
            app.check_action(
                "mark_boundary_complete",
                (),
            )
            is False
        )

        assert (
            app.check_action(
                "mark_boundary_clipped",
                (),
            )
            is False
        )


@pytest.mark.asyncio
async def test_reviewer_tui_shows_language_in_evidence(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=1.0,
    )

    storage.update_turn(
        "turn_000001",
        lambda turn: {
            **turn,
            "language": "en",
        },
    )

    session = ReviewerSession(storage)

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test():
        evidence = str(
            app.screen.query_one(
                "#evidence-content",
                Static,
            ).render()
        )

        assert "Language:" in evidence
        assert "en" in evidence


@pytest.mark.asyncio
async def test_reviewer_tui_shows_turn_timing_in_title(
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

    async with app.run_test():
        title = str(
            app.screen.query_one(
                "#turn-title",
                Static,
            ).render()
        )

        assert "Turn 1 / 1" in title
        assert "turn_000001" in title
        assert "00:01.000-00:02.000" in title
        assert "1.000s" in title


@pytest.mark.asyncio
async def test_reviewer_tui_status_is_transient(
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

    async with app.run_test():
        status = app.screen.query_one(
            "#status",
            Static,
        )

        assert str(status.render()) == ""
        assert status.display is False

        app._set_status(
            "Temporary status"
        )

        assert (
            "Temporary status"
            in str(status.render())
        )
        assert status.display is True

        app._refresh_view()

        assert str(status.render()) == ""
        assert status.display is False


@pytest.mark.asyncio
async def test_trim_turn_screen_returns_updated_range(
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

    result = []

    async with app.run_test() as pilot:
        app.push_screen(
            TrimTurnScreen(
                source_start=1.0,
                source_end=2.0,
            ),
            result.append,
        )
        await pilot.pause()

        assert isinstance(
            app.screen,
            TrimTurnScreen,
        )

        start_input = app.screen.query_one(
            "#trim-start",
            Input,
        )
        end_input = app.screen.query_one(
            "#trim-end",
            Input,
        )

        assert start_input.value == "1.000000"
        assert end_input.value == "2.000000"

        end_input.value = "1.750"
        end_input.focus()

        await pilot.press("enter")
        await pilot.pause()

    assert result == [
        TrimTurnRequest(
            source_start=1.0,
            source_end=1.75,
        )
    ]


@pytest.mark.asyncio
async def test_trim_turn_screen_rejects_invalid_number(
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

    result = []

    async with app.run_test() as pilot:
        app.push_screen(
            TrimTurnScreen(
                source_start=1.0,
                source_end=2.0,
            ),
            result.append,
        )
        await pilot.pause()

        end_input = app.screen.query_one(
            "#trim-end",
            Input,
        )

        end_input.value = "nope"
        end_input.focus()

        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(
            app.screen,
            TrimTurnScreen,
        )

        assert result == []


@pytest.mark.asyncio
async def test_reviewer_tui_trims_current_turn(
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

    def fake_trim(
        *,
        source_start=None,
        source_end=None,
    ):
        calls.append(
            (
                source_start,
                source_end,
            )
        )

        def update(record):
            record["source_start"] = source_start
            record["source_end"] = source_end
            return record

        return storage.update_turn(
            "turn_000001",
            update,
        )

    monkeypatch.setattr(
        session,
        "trim",
        fake_trim,
    )

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        await pilot.press("z")
        await pilot.pause()

        assert isinstance(
            app.screen,
            TrimTurnScreen,
        )

        end_input = app.screen.query_one(
            "#trim-end",
            Input,
        )

        end_input.value = "1.750"
        end_input.focus()

        await pilot.press("enter")
        await pilot.pause()

        assert calls == [
            (
                1.0,
                1.75,
            )
        ]

        assert (
            "Turn boundaries updated."
            in str(
                app.screen.query_one(
                    "#status",
                    Static,
                ).render()
            )
        )


@pytest.mark.asyncio
async def test_reviewer_tui_reports_rejected_trim(
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

    def reject_trim(
        *,
        source_start=None,
        source_end=None,
    ):
        raise ValueError(
            "Trim would change continuous ASR words"
        )

    monkeypatch.setattr(
        session,
        "trim",
        reject_trim,
    )

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        await pilot.press("z")
        await pilot.pause()

        end_input = app.screen.query_one(
            "#trim-end",
            Input,
        )

        end_input.value = "1.250"
        end_input.focus()

        await pilot.press("enter")
        await pilot.pause()

        status = str(
            app.screen.query_one(
                "#status",
                Static,
            ).render()
        )

        assert (
            "Boundary edit failed: "
            "Trim would change continuous ASR words"
            in status
        )

        turn = storage.get_turn(
            "turn_000001"
        )

        assert turn is not None
        assert turn["source_start"] == 1.0
        assert turn["source_end"] == 2.0


@pytest.mark.asyncio
async def test_reviewer_tui_boundary_edit_runs_in_background(
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

    started = threading.Event()
    release = threading.Event()

    def fake_trim(
        *,
        source_start=None,
        source_end=None,
    ):
        started.set()

        assert release.wait(
            timeout=5.0,
        )

        def update(record):
            record["source_start"] = source_start
            record["source_end"] = source_end
            return record

        return storage.update_turn(
            "turn_000001",
            update,
        )

    monkeypatch.setattr(
        session,
        "trim",
        fake_trim,
    )

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        await pilot.press("z")
        await pilot.pause()

        end_input = app.screen.query_one(
            "#trim-end",
            Input,
        )
        end_input.value = "1.750"
        end_input.focus()

        await pilot.press("enter")
        await pilot.pause()

        assert started.wait(
            timeout=1.0,
        )

        assert isinstance(
            app.screen,
            UpdatingTurnScreen,
        )

        content = str(
            app.screen.query_one(
                "#updating-audio-content",
                Static,
            ).render()
        )

        assert "review audio" in content
        assert "speaker evidence" in content

        release.set()

        for _ in range(20):
            await pilot.pause()

            status = str(
                app.screen.query_one(
                    "#status",
                    Static,
                ).render()
            )

            if "updated" in status.lower():
                break

        assert (
            "updated"
            in status.lower()
        )


@pytest.mark.asyncio
async def test_reviewer_tui_shows_blocking_audio_rebuild_modal(
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

    started = threading.Event()
    release = threading.Event()

    def fake_trim(
        *,
        source_start=None,
        source_end=None,
    ):
        started.set()

        assert release.wait(
            timeout=5.0,
        )

        return storage.get_turn(
            "turn_000001"
        )

    monkeypatch.setattr(
        session,
        "trim",
        fake_trim,
    )

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        await pilot.press("z")
        await pilot.pause()

        end_input = app.screen.query_one(
            "#trim-end",
            Input,
        )
        end_input.value = "1.750"
        end_input.focus()

        await pilot.press("enter")
        await pilot.pause()

        assert started.wait(
            timeout=1.0,
        )

        assert isinstance(
            app.screen,
            UpdatingTurnScreen,
        )

        title = str(
            app.screen.query_one(
                "#updating-audio-title",
                Static,
            ).render()
        )

        content = str(
            app.screen.query_one(
                "#updating-audio-content",
                Static,
            ).render()
        )

        assert "Updating turn" in title
        assert "review audio" in content
        assert "speaker evidence" in content
        assert "review audio" in content
        assert "speaker evidence" in content

        release.set()

        for _ in range(30):
            await pilot.pause()

            if not isinstance(
                app.screen,
                UpdatingTurnScreen,
            ):
                break

        assert not isinstance(
            app.screen,
            UpdatingTurnScreen,
        )

        status = str(
            app.screen.query_one(
                "#status",
                Static,
            ).render()
        )

        assert (
            "Turn boundaries updated."
            in status
        )


@pytest.mark.asyncio
async def test_boundary_editor_uses_boundary_terminology(
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
        await pilot.press("z")
        await pilot.pause()

        title = str(
            app.screen.query_one(
                "#trim-title",
                Static,
            ).render()
        )

        assert title == "Edit Boundaries"


def test_reviewer_shortcuts_style_keys_and_descriptions(
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

    text = app._format_shortcut_group({
        "show_help",
        "quit",
    })

    assert isinstance(text, Text)
    assert text.plain == "? Help   q Quit"

    styles = [
        str(span.style)
        for span in text.spans
    ]

    assert "bold" in styles
    assert "dim" in styles


@pytest.mark.asyncio
async def test_voice_picker_suggest_does_not_prefill_selection(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

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

    candidate = type(
        "Candidate",
        (),
        {
            "voice_id": "voice_001",
        },
    )()

    app = ReviewerTUI(
        ReviewerSession(storage),
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        app.push_screen(
            VoicePickerScreen(
                voices=storage.voices.load(),
                candidates=(candidate,),
                review_mode="suggest",
            )
        )

        await pilot.pause()

        options = app.screen.query_one(
            "#voice-options",
            OptionList,
        )

        assert options.highlighted is None


@pytest.mark.asyncio
async def test_voice_picker_prefill_selects_top_candidate(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    for voice_id, character in (
        ("voice_001", "Silco"),
        ("voice_002", "Vander"),
    ):
        storage.voices.append(
            {
                "schema_version": 1,
                "record_type": "voice_profile",
                "id": voice_id,
                "character": character,
                "language": "en",
                "aliases": [],
                "ignored": False,
                "notes": None,
                "metadata": {},
            }
        )

    candidate = type(
        "Candidate",
        (),
        {
            "voice_id": "voice_002",
        },
    )()

    app = ReviewerTUI(
        ReviewerSession(storage),
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        app.push_screen(
            VoicePickerScreen(
                voices=storage.voices.load(),
                candidates=(candidate,),
                review_mode="prefill",
            )
        )

        await pilot.pause()

        options = app.screen.query_one(
            "#voice-options",
            OptionList,
        )

        assert options.highlighted is not None

        selected = options.get_option_at_index(
            options.highlighted
        )

        assert selected.id == "voice_002"


def test_format_source_time_under_one_hour():
    from voice_dataset.reviewer_tui import (
        format_source_time,
    )

    assert (
        format_source_time(835.669)
        == "13:55.669"
    )


def test_format_source_time_over_one_hour():
    from voice_dataset.reviewer_tui import (
        format_source_time,
    )

    assert (
        format_source_time(4091.123)
        == "01:08:11.123"
    )


def test_build_split_turn_candidates_collapses_diarization_jitter(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    turn = {
        "id": "turn_000383",
        "source_regions": [
            "region_000483",
            "region_000484",
            "region_000485",
            "region_000486",
            "region_000487",
            "region_000488",
            "region_000489",
        ],
    }

    regions = [
        (
            "region_000483",
            1992.85034375,
            1995.02721875,
            "SPEAKER_09",
            "People need to feel safe. Yeah.",
        ),
        (
            "region_000484",
            1995.28034375,
            1995.29721875,
            "SPEAKER_09",
            None,
        ),
        (
            "region_000485",
            1995.29721875,
            1995.34784375,
            "SPEAKER_11",
            None,
        ),
        (
            "region_000486",
            1995.34784375,
            1995.38159375,
            "SPEAKER_09",
            None,
        ),
        (
            "region_000487",
            1995.38159375,
            1995.46596875,
            "SPEAKER_11",
            "Closed Captioning by Kris Brandhagen. brandhagen.com",
        ),
        (
            "region_000488",
            1995.46596875,
            1995.48284375,
            "SPEAKER_09",
            None,
        ),
        (
            "region_000489",
            1995.48284375,
            1996.44471875,
            "SPEAKER_11",
            "Popsite people.",
        ),
    ]

    for (
        region_id,
        start,
        end,
        speaker,
        transcript,
    ) in regions:
        storage.regions.append(
            {
                "schema_version": 2,
                "record_type": "candidate_region",
                "id": region_id,
                "source_id": "source_001",
                "source_start": start,
                "source_end": end,
                "detector": "pyannote-community-1",
                "detector_label": speaker,
                "representations": {},
                "transcripts": {
                    "whisper": {
                        "text": transcript,
                    }
                },
                "embeddings": {},
                "reconciliation": {
                    "status": "pending",
                    "reason": None,
                    "notes": None,
                },
                "metadata": {},
            }
        )

    candidates = build_split_turn_candidates(
        storage,
        turn,
    )

    assert len(candidates) == 1

    candidate = candidates[0]

    assert candidate.left_region_id == "region_000483"
    assert candidate.left_speaker == "SPEAKER_09"
    assert candidate.left_text == (
        "People need to feel safe. Yeah."
    )

    assert candidate.right_region_id == "region_000489"
    assert candidate.right_speaker == "SPEAKER_11"
    assert candidate.right_text == "Popsite people."

    # Tiny alternating diarization regions are kept as
    # provenance on the left. The stable SPEAKER_11
    # region begins the new right-hand turn.
    assert candidate.region_id == "region_000488"

    assert candidate.left_end == pytest.approx(
        1995.02721875
    )
    assert candidate.right_start == pytest.approx(
        1995.48284375
    )


@pytest.mark.asyncio
async def test_trim_screen_space_previews_while_input_is_focused(
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

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    played = []

    monkeypatch.setattr(
        "voice_dataset.reviewer_tui.play_file_region",
        lambda path, start, end:
            played.append(
                (
                    path,
                    start,
                    end,
                )
            ),
    )

    audio_path = tmp_path / "review.wav"

    async with app.run_test() as pilot:
        app.push_screen(
            TrimTurnScreen(
                source_start=1.0,
                source_end=2.0,
                audio_path=audio_path,
            )
        )
        await pilot.pause()

        assert isinstance(
            app.screen,
            TrimTurnScreen,
        )

        start_input = app.screen.query_one(
            "#trim-start",
            Input,
        )

        assert start_input.has_focus

        await pilot.press("space")
        await pilot.pause()

        assert played == [
            (
                audio_path,
                1.0,
                2.0,
            )
        ]

        # Space must not be inserted into the numeric field.
        assert start_input.value == "1.000000"


@pytest.mark.asyncio
async def test_reviewer_tui_merges_three_consecutive_turns(
    tmp_path,
    monkeypatch,
):
    storage = DatasetStorage(tmp_path)

    for index in range(1, 5):
        add_turn(
            storage,
            f"turn_{index:06d}",
            source_start=float(index),
        )

    session = ReviewerSession(storage)

    calls = []

    def merge(turn_ids):
        calls.append(
            list(turn_ids)
        )

        return storage.get_turn(
            "turn_000001"
        )

    monkeypatch.setattr(
        session,
        "merge",
        merge,
    )

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        await pilot.press("m")
        await pilot.pause()

        assert isinstance(
            app.screen,
            ConfirmMergeScreen,
        )

        content = app.screen.query_one(
            "#merge-content",
            Static,
        )

        rendered = str(
            content.render()
        )

        assert (
            "Selected 2 consecutive turns"
            in rendered
        )
        assert "turn_000001" in rendered
        assert "turn_000002" in rendered
        assert "turn_000003" not in rendered

        await pilot.press("plus")
        await pilot.pause()

        rendered = str(
            content.render()
        )

        assert (
            "Selected 3 consecutive turns"
            in rendered
        )
        assert "turn_000003" in rendered
        assert "turn_000004" not in rendered

        await pilot.press("enter")
        await pilot.pause()

        assert calls == [
            [
                "turn_000001",
                "turn_000002",
                "turn_000003",
            ],
        ]

        assert len(app.screen_stack) == 1


@pytest.mark.asyncio
async def test_trim_screen_preserves_unchanged_boundary_precision(
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

    original_start = 1156.9084687500003
    original_end = 1159.4684687500003

    result = []

    async with app.run_test() as pilot:
        app.push_screen(
            TrimTurnScreen(
                source_start=original_start,
                source_end=original_end,
            ),
            result.append,
        )
        await pilot.pause()

        start_input = app.screen.query_one(
            "#trim-start",
            Input,
        )

        end_input = app.screen.query_one(
            "#trim-end",
            Input,
        )

        # Change only the start.
        start_input.value = "1157.000000"

        # End remains merely its rounded display value.
        assert end_input.value == "1159.468469"

        end_input.focus()
        await pilot.press("enter")
        await pilot.pause()

    assert len(result) == 1
    request = result[0]

    assert request is not None
    assert request.source_start == pytest.approx(
        1157.0
    )

    # Must retain the exact stored float rather than
    # the six-decimal representation shown in the UI.
    assert request.source_end == original_end


@pytest.mark.asyncio
async def test_manual_split_without_diarization_uses_midpoint(
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

    def split(
        *,
        after_region_id=None,
        split_time=None,
    ):
        calls.append(
            (
                after_region_id,
                split_time,
            )
        )

        turn = storage.get_turn(
            "turn_000001"
        )

        assert turn is not None

        return turn, turn

    monkeypatch.setattr(
        session,
        "split",
        split,
    )

    monkeypatch.setattr(
        "voice_dataset.reviewer_tui."
        "build_split_turn_candidates",
        lambda *args, **kwargs: (),
    )

    monkeypatch.setattr(
        "voice_dataset.reviewer_tui."
        "build_split_word_hints",
        lambda *args, **kwargs: (),
    )

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    async with app.run_test() as pilot:
        await pilot.press("/")
        await pilot.pause()

        assert isinstance(
            app.screen,
            SplitTurnScreen,
        )

        assert app.screen.split_time == pytest.approx(
            1.5
        )

        await pilot.press("enter")
        await pilot.pause()

        assert calls == [
            (
                None,
                pytest.approx(1.5),
            ),
        ]

        assert len(app.screen_stack) == 1


@pytest.mark.asyncio
async def test_voice_picker_renames_selected_voice(
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
            "character": "Vi",
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

        options = app.screen.query_one(
            "#voice-options",
            OptionList,
        )
        options.highlighted = 0

        await pilot.press("e")
        await pilot.pause()

        field = app.screen.query_one(
            "#edit-value",
            Input,
        )

        assert field.value == "Vi"

        field.value = "Vi (young)"

        await pilot.press("enter")
        await pilot.pause()

        voice = storage.get_voice(
            "voice_001"
        )

        assert voice is not None
        assert (
            voice["character"]
            == "Vi (young)"
        )

        turn = storage.get_turn(
            "turn_000001"
        )

        assert turn is not None
        assert (
            turn["assignment"]["status"]
            == "unknown"
        )

        assert len(app.screen_stack) == 1


def test_reviewer_formats_speaker_change_evidence(
    tmp_path,
):
    storage = DatasetStorage(tmp_path)

    add_turn(
        storage,
        "turn_000001",
        source_start=10.0,
    )

    def update(record):
        metadata = dict(
            record.get("metadata") or {}
        )

        metadata["automatic_pipeline"] = {
            "status": "review",
            "review_reasons": [
                "suspected_speaker_change",
            ],
        }

        metadata["speaker_change_evidence"] = {
            "status": "suspected",
            "strongest": {
                "boundary": 10.5,
                "window_kind": "adaptive",
                "left_start": 10.0,
                "left_end": 10.5,
                "right_start": 10.5,
                "right_end": 11.0,
                "ecapa_similarity": 0.05,
                "wespeaker_similarity": 0.07,
                "mean_similarity": 0.06,
                "encoder_disagreement": 0.02,
                "divergence": 0.94,
                "suspected": True,
            },
            "changes": [],
        }

        record["metadata"] = metadata
        return record

    storage.update_turn(
        "turn_000001",
        update,
    )

    session = ReviewerSession(storage)

    app = ReviewerTUI(
        session,
        embedding_names=(),
        context_padding=2.0,
    )

    view = session.current_view(
        embedding_names=(),
        speaker_limit=3,
    )

    assert view is not None

    rendered = app._format_evidence(view)

    assert "suspected_speaker_change" in rendered
    assert "Spk change: suspected" in rendered
    assert "Change at: 00:10.500 (adaptive)" in rendered
    assert "Divergence: 0.9400" in rendered
    assert "ECAPA:     0.0500" in rendered
    assert "WeSpeaker: 0.0700" in rendered
    assert "Disagree:  0.0200" in rendered

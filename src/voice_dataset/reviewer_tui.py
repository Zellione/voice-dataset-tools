from __future__ import annotations
from dataclasses import dataclass

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Footer,
    Header,
    Input,
    OptionList,
    Static,
)
from textual.widgets.option_list import Option

from .playback import (
    is_playing,
    play_preferred_review_audio,
    play_representation,
    play_turn_context,
    stop,
)
from .reviewer import (
    raw_representation,
    sorted_turns,
)
from .reviewer_session import ReviewerSession
from .reviewer_view import ReviewerTurnView


class HelpScreen(ModalScreen[None]):
    CSS = """
    HelpScreen {
        align: center middle;
    }

    #help-dialog {
        width: 76;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        border: round $primary;
        background: $surface;
    }

    #help-title {
        text-style: bold;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding(
            "escape",
            "dismiss_help",
            "Close",
        ),
        Binding(
            "question_mark",
            "dismiss_help",
            "Close",
            key_display="?",
        ),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="help-dialog"):
            yield Static(
                "Reviewer Help",
                id="help-title",
            )
            yield Static(
                "\n".join([
                    "Navigation",
                    "  ← / b     Previous turn",
                    "  → / n     Next turn",
                    "",
                    "Playback",
                    "  Space / p Play review audio",
                    "  c         Play context",
                    "  r         Play raw audio",
                    "  s         Stop playback",
                    "",
                    "Turn",
                    "  t         Edit transcript",
                    "  l         Edit language",
                    "  a         Mark reviewed",
                    "  x         Mark pending",
                    "",
                    "Speaker",
                    "  v         Assign/create voice",
                    "  u         Mark voice unknown",
                    "  i         Ignore turn",
                    "",
                    "Boundary",
                    "  k         Mark complete",
                    "  d         Mark clipped",
                    "",
                    "General",
                    "  ?         Help",
                    "  q         Quit",
                    "",
                    "Coming next",
                    "  f         Accept alignment recovery",
                    "  e         Accept edge recovery",
                    "  m         Merge",
                    "  /         Split",
                ])
            )

    def action_dismiss_help(self) -> None:
        self.dismiss(None)


@dataclass(frozen=True)
class NewVoiceRequest:
    character: str | None
    language: str | None


class NewVoiceScreen(
    ModalScreen[NewVoiceRequest | None]
):
    CSS = """
    NewVoiceScreen {
        align: center middle;
    }

    #new-voice-dialog {
        width: 70;
        height: auto;
        padding: 1 2;
        border: round $primary;
        background: $surface;
    }

    #new-voice-title {
        text-style: bold;
        margin-bottom: 1;
    }

    Input {
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding(
            "escape",
            "cancel",
            "Cancel",
        ),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="new-voice-dialog"):
            yield Static(
                "Create Voice",
                id="new-voice-title",
            )

            yield Static("Character")
            yield Input(
                placeholder="Character name",
                id="new-voice-character",
            )

            yield Static("Language")
            yield Input(
                placeholder="Language, e.g. en",
                id="new-voice-language",
            )

        yield Footer()

    def on_mount(self) -> None:
        self.query_one(
            "#new-voice-character",
            Input,
        ).focus()

    def on_input_submitted(
        self,
        event: Input.Submitted,
    ) -> None:
        if event.input.id == "new-voice-character":
            self.query_one(
                "#new-voice-language",
                Input,
            ).focus()
            return

        self.action_create()

    def action_create(self) -> None:
        character_value = self.query_one(
            "#new-voice-character",
            Input,
        ).value.strip()

        language_value = self.query_one(
            "#new-voice-language",
            Input,
        ).value.strip()

        self.dismiss(
            NewVoiceRequest(
                character=character_value or None,
                language=language_value or None,
            )
        )

    def action_cancel(self) -> None:
        self.dismiss(None)


class VoicePickerScreen(
    ModalScreen[str | NewVoiceRequest | None]
):
    def __init__(
        self,
        *,
        voices: list[dict],
        candidates: tuple = (),
    ) -> None:
        super().__init__()
        self.voices = voices
        self.candidates = candidates

        self._voices_by_id = {
            str(voice["id"]): voice
            for voice in voices
        }

        ordered_ids: list[str] = []

        for candidate in candidates:
            if (
                candidate.voice_id
                in self._voices_by_id
                and candidate.voice_id
                not in ordered_ids
            ):
                ordered_ids.append(
                    candidate.voice_id
                )

        for voice in voices:
            voice_id = str(voice["id"])

            if voice_id not in ordered_ids:
                ordered_ids.append(voice_id)

        self.voice_ids = tuple(ordered_ids)

    CSS = """
    VoicePickerScreen {
        align: center middle;
    }

    #voice-dialog {
        width: 80;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        border: round $primary;
        background: $surface;
    }

    #voice-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #voice-options {
        height: auto;
        max-height: 24;
        margin-top: 1;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding(
            "enter",
            "assign",
            "Assign",
        ),
        Binding(
            "n",
            "new_voice",
            "New Voice",
        ),
        Binding(
            "escape",
            "cancel",
            "Cancel",
        ),
    ]

    def compose(self) -> ComposeResult:
        candidate_ids = {
            candidate.voice_id
            for candidate in self.candidates
        }

        summary_lines = [
            "Suggested voices are listed first.",
            "",
            "All voices",
        ]

        if self.voices:
            for voice in self.voices:
                summary_lines.append(
                    f"  {voice['id']}  "
                    f"{voice.get('character') or '-'}"
                )
        else:
            summary_lines.append(
                "  No voice profiles."
            )

        options: list[Option] = []

        for voice_id in self.voice_ids:
            voice = self._voices_by_id[
                voice_id
            ]

            character = (
                voice.get("character")
                or "-"
            )

            suggested = (
                "  [suggested]"
                if voice_id in candidate_ids
                else ""
            )

            options.append(
                Option(
                    f"{voice_id}  "
                    f"{character}"
                    f"{suggested}",
                    id=voice_id,
                )
            )

        with Vertical(id="voice-dialog"):
            yield Static(
                "Assign Voice",
                id="voice-title",
            )

            yield Static(
                "\n".join(summary_lines),
                id="voice-content",
            )

            if options:
                yield OptionList(
                    *options,
                    id="voice-options",
                )

        yield Footer()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_new_voice(self) -> None:
        self.app.push_screen(
            NewVoiceScreen(),
            self._new_voice_created,
        )

    def check_action(
        self,
        action: str,
        parameters: tuple,
    ) -> bool | None:
        if action == "assign":
            return bool(self.voice_ids)

        return super().check_action(
            action,
            parameters,
        )

    def action_assign(self) -> None:
        if not self.voice_ids:
            return

        options = self.query_one(
            "#voice-options",
            OptionList,
        )

        highlighted = options.highlighted

        if highlighted is None:
            return

        option = options.get_option_at_index(
            highlighted
        )

        if option.id is None:
            return

        self.dismiss(
            str(option.id)
        )

    def _new_voice_created(
        self,
        request: NewVoiceRequest | None,
    ) -> None:
        if request is None:
            return

        self.dismiss(request)

    def on_option_list_option_selected(
        self,
        event: OptionList.OptionSelected,
    ) -> None:
        option_id = event.option.id

        if option_id is None:
            return

        self.dismiss(
            str(option_id)
        )


class EditValueScreen(
    ModalScreen[str | None]
):
    CSS = """
    EditValueScreen {
        align: center middle;
    }

    #edit-dialog {
        width: 80;
        height: auto;
        padding: 1 2;
        border: round $primary;
        background: $surface;
    }

    #edit-title {
        text-style: bold;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding(
            "escape",
            "cancel",
            "Cancel",
        ),
    ]

    def __init__(
        self,
        *,
        title: str,
        value: str,
        placeholder: str = "",
    ) -> None:
        super().__init__()
        self.title_text = title
        self.initial_value = value
        self.placeholder = placeholder

    def compose(self) -> ComposeResult:
        with Vertical(id="edit-dialog"):
            yield Static(
                self.title_text,
                id="edit-title",
            )

            yield Input(
                value=self.initial_value,
                placeholder=self.placeholder,
                id="edit-value",
            )

        yield Footer()

    def on_mount(self) -> None:
        self.query_one(
            "#edit-value",
            Input,
        ).focus()

    def on_input_submitted(
        self,
        event: Input.Submitted,
    ) -> None:
        self.dismiss(
            event.value.strip()
        )

    def action_cancel(self) -> None:
        self.dismiss(None)


class ConfirmRecoveryScreen(
    ModalScreen[bool | None]
):
    CSS = """
    ConfirmRecoveryScreen {
        align: center middle;
    }

    #recovery-dialog {
        width: 80;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        border: round $primary;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding(
            "enter",
            "accept",
            "Accept",
        ),
        Binding(
            "escape",
            "cancel",
            "Cancel",
        ),
    ]

    def __init__(
        self,
        *,
        content: str,
    ) -> None:
        super().__init__()
        self.content = content

    def compose(self) -> ComposeResult:
        with Vertical(id="recovery-dialog"):
            yield Static(
                self.content,
                id="recovery-content",
            )

        yield Footer()

    def action_accept(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ConfirmMergeScreen(
    ModalScreen[bool | None]
):
    CSS = """
    ConfirmMergeScreen {
        align: center middle;
    }

    #merge-dialog {
        width: 90;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        border: round $primary;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding(
            "enter",
            "accept",
            "Merge",
        ),
        Binding(
            "escape",
            "cancel",
            "Cancel",
        ),
    ]

    def __init__(
        self,
        *,
        content: str,
    ) -> None:
        super().__init__()
        self.content = content

    def compose(self) -> ComposeResult:
        with Vertical(id="merge-dialog"):
            yield Static(
                self.content,
                id="merge-content",
            )

        yield Footer()

    def action_accept(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(None)


class SplitTurnScreen(
    ModalScreen[str | None]
):
    CSS = """
    SplitTurnScreen {
        align: center middle;
    }

    #split-dialog {
        width: 80;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        border: round $primary;
        background: $surface;
    }

    #split-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #split-options {
        height: auto;
        max-height: 20;
        margin-top: 1;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding(
            "escape",
            "cancel",
            "Cancel",
        ),
    ]

    def __init__(
        self,
        *,
        turn_id: str,
        region_ids: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.turn_id = turn_id
        self.region_ids = region_ids

    def compose(self) -> ComposeResult:
        options = [
            Option(
                f"after {region_id}",
                id=region_id,
            )
            for region_id in self.region_ids
        ]

        with Vertical(id="split-dialog"):
            yield Static(
                "Split Turn",
                id="split-title",
            )

            yield Static(
                "\n".join([
                    f"Turn: {self.turn_id}",
                    "",
                    "Choose the final region "
                    "for the left turn:",
                ]),
                id="split-content",
            )

            yield OptionList(
                *options,
                id="split-options",
            )

        yield Footer()

    def on_option_list_option_selected(
        self,
        event: OptionList.OptionSelected,
    ) -> None:
        option_id = event.option.id

        if option_id is None:
            return

        self.dismiss(
            str(option_id)
        )

    def action_cancel(self) -> None:
        self.dismiss(None)


class ReviewerTUI(App[None]):
    TITLE = "Voice Dataset Reviewer"

    CSS = """
    Screen {
        layout: vertical;
    }

    #main {
        height: 1fr;
        padding: 1 2;
    }

    #turn-title {
        height: auto;
        text-style: bold;
        margin-bottom: 1;
    }

    #transcript {
        height: auto;
        min-height: 3;
        padding: 1 2;
        border: round $primary;
        margin-bottom: 1;
    }

    #columns {
        height: 1fr;
    }

    #evidence-panel,
    #speaker-panel {
        width: 1fr;
        height: 100%;
        padding: 1 2;
        border: round $secondary;
    }

    #evidence-panel {
        margin-right: 1;
    }

    .panel-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #status {
        height: 1;
        padding: 0 2;
    }
    """

    BINDINGS = [
        Binding(
            "left,b",
            "previous_turn",
            "Previous",
        ),
        Binding(
            "right,n",
            "next_turn",
            "Next",
        ),
        Binding(
            "question_mark",
            "show_help",
            "Help",
            key_display="?",
        ),
        Binding(
            "q",
            "quit",
            "Quit",
        ),
        Binding(
            "space,p",
            "play_preferred",
            "Play",
            key_display="Space",
        ),
        Binding(
            "c",
            "play_context",
            "Context",
        ),
        Binding(
            "r",
            "play_raw",
            "Raw",
        ),
        Binding(
            "s",
            "stop_playback",
            "Stop",
        ),
        Binding(
            "v",
            "assign_voice",
            "Voice",
        ),
        Binding(
            "a",
            "mark_reviewed",
            "Reviewed",
        ),
        Binding(
            "x",
            "mark_pending",
            "Pending",
        ),
        Binding(
            "u",
            "mark_unknown",
            "Unknown",
        ),
        Binding(
            "i",
            "ignore_turn",
            "Ignore",
        ),
        Binding(
            "t",
            "edit_transcript",
            "Transcript",
        ),
        Binding(
            "l",
            "edit_language",
            "Language",
        ),
        Binding(
            "k",
            "mark_boundary_complete",
            "Boundary OK",
        ),
        Binding(
            "d",
            "mark_boundary_clipped",
            "Boundary clipped",
        ),
        Binding(
            "f",
            "accept_alignment_recovery",
            "Alignment recovery",
        ),
        Binding(
            "e",
            "accept_edge_recovery",
            "Edge recovery",
        ),
        Binding(
            "m",
            "merge_with_next",
            "Merge",
        ),
        Binding(
            "/",
            "split_turn",
            "Split",
        ),
    ]

    def __init__(
        self,
        session: ReviewerSession,
        *,
        embedding_names: tuple[str, ...],
        context_padding: float,
    ) -> None:
        super().__init__()
        self.session = session
        self.embedding_names = embedding_names
        self.context_padding = context_padding
        self._last_playback_active = False

    def _current_turn(self) -> dict:
        turn = self.session.current()

        if turn is None:
            raise ValueError(
                "No current turn"
            )

        return turn

    def _set_status(
        self,
        message: str,
    ) -> None:
        self.query_one(
            "#status",
            Static,
        ).update(message)

    def action_play_preferred(self) -> None:
        try:
            turn = self._current_turn()

            name, path = play_preferred_review_audio(
                self.session.storage.root,
                turn,
                blocking=False,
            )

            self._sync_playback_state()

            self._set_status(
                f"Playing {name}: {path}"
            )
        except (
            ValueError,
            KeyError,
            RuntimeError,
            OSError,
        ) as exc:
            self._set_status(
                f"Playback failed: {exc}"
            )

    def action_play_raw(self) -> None:
        try:
            turn = self._current_turn()

            representation = raw_representation(
                turn
            )

            path = play_representation(
                self.session.storage.root,
                representation,
                blocking=False,
            )

            self._sync_playback_state()

            self._set_status(
                f"Playing raw: {path}"
            )
        except (
            ValueError,
            KeyError,
            RuntimeError,
            OSError,
        ) as exc:
            self._set_status(
                f"Playback failed: {exc}"
            )

    def action_play_context(self) -> None:
        try:
            turn = self._current_turn()

            source = self.session.storage.get_source(
                turn["source_id"]
            )

            if source is None:
                raise KeyError(
                    "Source does not exist: "
                    f"{turn['source_id']}"
                )

            (
                name,
                path,
                start,
                end,
            ) = play_turn_context(
                source,
                turn,
                padding=self.context_padding,
                blocking=False,
            )

            self._sync_playback_state()

            self._set_status(
                f"Playing context {name}: "
                f"{start:.3f}-{end:.3f} "
                f"from {path}"
            )
        except (
            ValueError,
            KeyError,
            RuntimeError,
            OSError,
        ) as exc:
            self._set_status(
                f"Playback failed: {exc}"
            )

    def action_stop_playback(self) -> None:
        stop()
        self._sync_playback_state()
        self._set_status(
            "Playback stopped."
        )

    def action_previous_turn(self) -> None:
        stop()
        self.session.previous()
        self._refresh_view()

    def action_next_turn(self) -> None:
        stop()
        self.session.next()
        self._refresh_view()

    def action_quit(self) -> None:
        stop()
        self.exit()

    def _voice_selected(
        self,
        result: str | NewVoiceRequest | None,
    ) -> None:
        if result is None:
            return

        try:
            if isinstance(
                result,
                NewVoiceRequest,
            ):
                voice = (
                    self.session
                    .create_and_assign_voice(
                        character=result.character,
                        language=result.language,
                    )
                )

                message = (
                    "Created and assigned voice: "
                    f"{voice['id']}"
                )
            else:
                self.session.assign_voice(
                    result
                )

                message = (
                    f"Assigned voice: {result}"
                )

            self._refresh_view()
            self._set_status(message)

        except (
            ValueError,
            KeyError,
            RuntimeError,
            OSError,
        ) as exc:
            self._set_status(
                f"Voice assignment failed: {exc}"
            )

    def action_assign_voice(self) -> None:
        view = self._current_view()

        candidates = (
            view.speaker_candidates
            if view is not None
            else ()
        )

        self.push_screen(
            VoicePickerScreen(
                voices=(
                    self.session.storage
                    .voices.load()
                ),
                candidates=candidates,
            ),
            self._voice_selected,
        )

    def action_mark_reviewed(self) -> None:
        try:
            self.session.mark_reviewed()
            self._refresh_view()
            self._set_status(
                "Turn marked reviewed."
            )
        except (
            ValueError,
            KeyError,
            RuntimeError,
            OSError,
        ) as exc:
            self._set_status(
                f"Action failed: {exc}"
            )

    def action_mark_pending(self) -> None:
        try:
            self.session.mark_pending()
            self._refresh_view()
            self._set_status(
                "Turn marked pending."
            )
        except (
            ValueError,
            KeyError,
            RuntimeError,
            OSError,
        ) as exc:
            self._set_status(
                f"Action failed: {exc}"
            )

    def action_mark_unknown(self) -> None:
        try:
            self.session.mark_unknown()
            self._refresh_view()
            self._set_status(
                "Voice marked unknown."
            )
        except (
            ValueError,
            KeyError,
            RuntimeError,
            OSError,
        ) as exc:
            self._set_status(
                f"Action failed: {exc}"
            )

    def action_ignore_turn(self) -> None:
        try:
            self.session.ignore()
            self._refresh_view()
            self._set_status(
                "Turn ignored."
            )
        except (
            ValueError,
            KeyError,
            RuntimeError,
            OSError,
        ) as exc:
            self._set_status(
                f"Action failed: {exc}"
            )

    def _transcript_edited(
        self,
        value: str | None,
    ) -> None:
        if value is None:
            return

        try:
            self.session.edit_transcript(
                value
            )
            self._refresh_view()
            self._set_status(
                "Transcript updated."
            )
        except (
            ValueError,
            KeyError,
            RuntimeError,
            OSError,
        ) as exc:
            self._set_status(
                f"Action failed: {exc}"
            )

    def action_edit_transcript(self) -> None:
        turn = self.session.current()

        if turn is None:
            return

        self.push_screen(
            EditValueScreen(
                title="Edit Transcript",
                value=str(
                    turn.get("transcript")
                    or ""
                ),
                placeholder="Transcript",
            ),
            self._transcript_edited,
        )

    def _language_edited(
        self,
        value: str | None,
    ) -> None:
        if value is None:
            return

        if not value:
            self._set_status(
                "Language must not be empty."
            )
            return

        try:
            self.session.set_language(
                value
            )
            self._refresh_view()
            self._set_status(
                f"Language set to {value}."
            )
        except (
            ValueError,
            KeyError,
            RuntimeError,
            OSError,
        ) as exc:
            self._set_status(
                f"Action failed: {exc}"
            )

    def action_edit_language(self) -> None:
        turn = self.session.current()

        if turn is None:
            return

        self.push_screen(
            EditValueScreen(
                title="Edit Language",
                value=str(
                    turn.get("language")
                    or ""
                ),
                placeholder="Language, e.g. en",
            ),
            self._language_edited,
        )

    def action_mark_boundary_complete(self) -> None:
        try:
            self.session.mark_boundary_complete()
            self._refresh_view()
            self._set_status(
                "Boundary marked complete."
            )
        except (
            ValueError,
            KeyError,
            RuntimeError,
            OSError,
        ) as exc:
            self._set_status(
                f"Action failed: {exc}"
            )

    def action_mark_boundary_clipped(self) -> None:
        try:
            self.session.mark_boundary_clipped()
            self._refresh_view()
            self._set_status(
                "Boundary marked clipped."
            )
        except (
            ValueError,
            KeyError,
            RuntimeError,
            OSError,
        ) as exc:
            self._set_status(
                f"Action failed: {exc}"
            )

    def _alignment_recovery_confirmed(
        self,
        accepted: bool | None,
    ) -> None:
        if not accepted:
            return

        try:
            self.session.accept_alignment_recovery()
            self._refresh_view()
            self._set_status(
                "Alignment recovery accepted."
            )
        except (
            ValueError,
            KeyError,
            RuntimeError,
            OSError,
        ) as exc:
            self._set_status(
                f"Action failed: {exc}"
            )

    def action_accept_alignment_recovery(
        self,
    ) -> None:
        view = self._current_view()

        if view is None:
            return

        recovery = view.alignment.recovery

        if (
            recovery is None
            or recovery.status != "suggested"
        ):
            self._set_status(
                "No suggested alignment recovery."
            )
            return

        source_range = "-"

        if (
            recovery.source_start is not None
            and recovery.source_end is not None
        ):
            source_range = (
                f"{recovery.source_start:.3f}"
                "-"
                f"{recovery.source_end:.3f}"
            )

        regions = (
            ", ".join(recovery.region_ids)
            if recovery.region_ids
            else "-"
        )

        token_match = "-"

        if (
            recovery.matched_token_count is not None
            and recovery.candidate_token_count
            is not None
        ):
            token_match = (
                f"{recovery.matched_token_count}"
                " / "
                f"{recovery.candidate_token_count}"
            )

        issue_words = (
            ", ".join(
                str(index)
                for index
                in view.alignment.issue_word_indices
            )
            if view.alignment.issue_word_indices
            else "-"
        )

        content = "\n".join([
            "Alignment Recovery",
            "",
            (
                "Current range:   "
                f"{view.start:.3f}-{view.end:.3f}"
            ),
            (
                "Proposed range:  "
                f"{source_range}"
            ),
            (
                "Speaker:         "
                f"{recovery.speaker or '-'}"
            ),
            (
                "Regions:         "
                f"{regions}"
            ),
            (
                "Token match:     "
                f"{token_match}"
            ),
            (
                "Issue words:     "
                f"{issue_words}"
            ),
        ])

        self.push_screen(
            ConfirmRecoveryScreen(
                content=content,
            ),
            self._alignment_recovery_confirmed,
        )

    def _edge_recovery_confirmed(
        self,
        accepted: bool | None,
    ) -> None:
        if not accepted:
            return

        try:
            self.session.accept_edge_recovery()
            self._refresh_view()
            self._set_status(
                "Edge recovery accepted."
            )
        except (
            ValueError,
            KeyError,
            RuntimeError,
            OSError,
        ) as exc:
            self._set_status(
                f"Action failed: {exc}"
            )

    def action_accept_edge_recovery(
        self,
    ) -> None:
        view = self._current_view()

        if view is None:
            return

        edge = view.edge_recovery

        if (
            edge is None
            or edge.status != "suggested"
        ):
            self._set_status(
                "No suggested edge recovery."
            )
            return

        if edge.edge == "start":
            proposed = (
                f"{edge.source_start:.3f}"
                if edge.source_start is not None
                else "-"
            )
        elif edge.edge == "end":
            proposed = (
                f"{edge.source_end:.3f}"
                if edge.source_end is not None
                else "-"
            )
        else:
            proposed = "-"

        lines = [
            "Edge Recovery",
            "",
            f"Edge:            {edge.edge or '-'}",
            (
                "Current range:   "
                f"{view.start:.3f}-{view.end:.3f}"
            ),
            (
                "Proposed edge:   "
                f"{proposed}"
            ),
            (
                "Speaker:         "
                f"{edge.speaker or '-'}"
            ),
            (
                "Region:          "
                f"{edge.region_id or '-'}"
            ),
        ]

        if edge.conflicting_region_id:
            lines.append(
                "Conflict region: "
                f"{edge.conflicting_region_id}"
            )

        if edge.candidate_token:
            lines.append(
                "Candidate token: "
                f"{edge.candidate_token}"
            )

        if edge.whisper_token:
            lines.append(
                "Whisper token:   "
                f"{edge.whisper_token}"
            )

        if edge.candidate_text:
            lines.extend([
                "",
                "Candidate:",
                edge.candidate_text,
            ])

        if edge.whisper_text:
            lines.extend([
                "",
                "Whisper:",
                edge.whisper_text,
            ])

        if edge.conflicting_whisper_text:
            lines.extend([
                "",
                "Conflicting Whisper:",
                edge.conflicting_whisper_text,
            ])

        self.push_screen(
            ConfirmRecoveryScreen(
                content="\n".join(lines),
            ),
            self._edge_recovery_confirmed,
        )

    def _merge_with_next_confirmed(
        self,
        accepted: bool | None,
    ) -> None:
        if not accepted:
            return

        try:
            merged = self.session.merge_with_next()

            self._refresh_view()

            self._set_status(
                f"Merged turn: {merged['id']}"
            )
        except (
            ValueError,
            KeyError,
            RuntimeError,
            OSError,
        ) as exc:
            self._set_status(
                f"Merge failed: {exc}"
            )

    def action_merge_with_next(self) -> None:
        current = self.session.current()

        if current is None:
            return

        turn_id = str(current["id"])
        source_id = current.get("source_id")

        if not isinstance(source_id, str) or not source_id:
            self._set_status(
                "Current turn has invalid source_id"
            )
            return

        turns = sorted_turns(
            self.session.storage,
            source_id=source_id,
        )

        turn_ids = [
            str(turn["id"])
            for turn in turns
        ]

        try:
            index = turn_ids.index(turn_id)
        except ValueError:
            self._set_status(
                "Current turn is missing from "
                "canonical turn timeline"
            )
            return

        if index >= len(turns) - 1:
            self._set_status(
                "Current turn has no next turn"
            )
            return

        next_turn = turns[index + 1]

        current_start = float(
            current["source_start"]
        )
        current_end = float(
            current["source_end"]
        )
        next_start = float(
            next_turn["source_start"]
        )
        next_end = float(
            next_turn["source_end"]
        )

        content = "\n".join([
            "Merge Turns",
            "",
            (
                f"{current['id']}  "
                f"{current_start:.3f}-{current_end:.3f}"
            ),
            str(
                current.get("transcript")
                or "-"
            ),
            "",
            (
                f"{next_turn['id']}  "
                f"{next_start:.3f}-{next_end:.3f}"
            ),
            str(
                next_turn.get("transcript")
                or "-"
            ),
        ])

        self.push_screen(
            ConfirmMergeScreen(
                content=content,
            ),
            self._merge_with_next_confirmed,
        )

    def _split_region_selected(
        self,
        region_id: str | None,
    ) -> None:
        if region_id is None:
            return

        try:
            left, right = self.session.split(
                after_region_id=region_id,
            )

            self._refresh_view()

            self._set_status(
                "Split turn: "
                f"{left['id']} / {right['id']}"
            )
        except (
            ValueError,
            KeyError,
            RuntimeError,
            OSError,
        ) as exc:
            self._set_status(
                f"Split failed: {exc}"
            )

    def action_split_turn(self) -> None:
        turn = self.session.current()

        if turn is None:
            return

        source_regions = (
            turn.get("source_regions")
            or []
        )

        if not isinstance(
            source_regions,
            list,
        ):
            self._set_status(
                "Turn has invalid source_regions"
            )
            return

        valid_regions = tuple(
            str(region_id)
            for region_id
            in source_regions[:-1]
            if isinstance(region_id, str)
            and region_id
        )

        if not valid_regions:
            self._set_status(
                "Turn has no valid split point"
            )
            return

        self.push_screen(
            SplitTurnScreen(
                turn_id=str(turn["id"]),
                region_ids=valid_regions,
            ),
            self._split_region_selected,
        )

    def compose(self) -> ComposeResult:
        yield Header()

        with Vertical(id="main"):
            yield Static(id="turn-title")
            yield Static(id="transcript")

            with Horizontal(id="columns"):
                with Vertical(id="evidence-panel"):
                    yield Static(
                        "Evidence",
                        classes="panel-title",
                    )
                    yield Static(
                        id="evidence-content"
                    )

                with Vertical(id="speaker-panel"):
                    yield Static(
                        "Speaker Candidates",
                        classes="panel-title",
                    )
                    yield Static(
                        id="speaker-content"
                    )

        yield Static(id="status")
        yield Footer()

    def on_mount(self) -> None:
        self._last_playback_active = (
            is_playing()
        )

        self._refresh_view()

        self.set_interval(
            0.1,
            self._poll_playback_state,
        )

    def _poll_playback_state(
        self,
    ) -> None:
        self._sync_playback_state()

    def _sync_playback_state(
        self,
    ) -> None:
        playback_active = is_playing()

        if (
            playback_active
            == self._last_playback_active
        ):
            return

        self._last_playback_active = (
            playback_active
        )

        self.refresh_bindings()

    def _current_view(
        self,
    ) -> ReviewerTurnView | None:
        return self.session.current_view(
            embedding_names=self.embedding_names,
            speaker_limit=3,
        )

    def check_action(
        self,
        action: str,
        parameters: tuple,
    ) -> bool | None:
        if action == "previous_turn":
            return self.session.position > 1

        if action == "next_turn":
            return (
                self.session.position
                < self.session.total
            )

        if action == "accept_alignment_recovery":
            turn = self.session.current()

            if turn is None:
                return False

            metadata = turn.get("metadata")

            if not isinstance(metadata, dict):
                return False

            alignment = metadata.get(
                "alignment_evidence"
            )

            if not isinstance(alignment, dict):
                return False

            recovery = alignment.get("recovery")

            return (
                isinstance(recovery, dict)
                and recovery.get("status")
                == "suggested"
            )

        if action == "accept_edge_recovery":
            turn = self.session.current()

            if turn is None:
                return False

            metadata = turn.get("metadata")

            if not isinstance(metadata, dict):
                return False

            edge = metadata.get(
                "edge_evidence"
            )

            return (
                isinstance(edge, dict)
                and edge.get("status")
                == "suggested"
            )

        if action == "merge_with_next":
            current = self.session.current()

            if current is None:
                return False

            turn_id = str(current["id"])
            source_id = current.get("source_id")

            if (
                not isinstance(source_id, str)
                or not source_id
            ):
                return False

            turns = sorted_turns(
                self.session.storage,
                source_id=source_id,
            )

            turn_ids = [
                str(turn["id"])
                for turn in turns
            ]

            try:
                index = turn_ids.index(turn_id)
            except ValueError:
                return False

            return index < len(turn_ids) - 1

        if action == "split_turn":
            turn = self.session.current()

            if turn is None:
                return False

            source_regions = (
                turn.get("source_regions")
                or []
            )

            if not isinstance(
                source_regions,
                list,
            ):
                return False

            valid_regions = [
                region_id
                for region_id
                in source_regions[:-1]
                if isinstance(region_id, str)
                and region_id
            ]

            return bool(valid_regions)

        if action == "mark_pending":
            turn = self.session.current()

            if turn is None:
                return False

            review = turn.get("review") or {}

            return (
                review.get("status")
                != "pending"
            )

        if action == "mark_reviewed":
            turn = self.session.current()

            if turn is None:
                return False

            review = turn.get("review") or {}

            return (
                review.get("status")
                != "reviewed"
            )

        if action == "mark_unknown":
            turn = self.session.current()

            if turn is None:
                return False

            assignment = (
                turn.get("assignment")
                or {}
            )

            return (
                assignment.get("status")
                != "unknown"
            )

        if action in {
            "mark_boundary_complete",
            "mark_boundary_clipped",
        }:
            turn = self.session.current()

            if turn is None:
                return False

            review = turn.get("review") or {}

            if not isinstance(review, dict):
                return False

            boundary = review.get("boundary") or {}

            if not isinstance(boundary, dict):
                return False

            status = boundary.get("status")

            if action == "mark_boundary_complete":
                return status != "complete"

            return status != "clipped"

        if action == "stop_playback":
            return is_playing()

        return super().check_action(
            action,
            parameters,
        )

    def _refresh_view(self) -> None:
        view = self._current_view()

        if view is None:
            self.query_one(
                "#turn-title",
                Static,
            ).update("No turns")

            self.query_one(
                "#transcript",
                Static,
            ).update("")

            self.query_one(
                "#evidence-content",
                Static,
            ).update("")

            self.query_one(
                "#speaker-content",
                Static,
            ).update("")

            self.query_one(
                "#status",
                Static,
            ).update("No speech turns to review.")

            self.refresh_bindings()

            self.query_one(
                "#status",
                Static,
            ).update(
                "No speech turns to review."
            )

            self.refresh_bindings()
            return

        self.query_one(
            "#turn-title",
            Static,
        ).update(
            f"Turn {view.position} / {view.total}  "
            f"{view.turn_id}"
        )

        self.query_one(
            "#transcript",
            Static,
        ).update(
            view.transcript or "-"
        )

        self.query_one(
            "#evidence-content",
            Static,
        ).update(
            self._format_evidence(view)
        )

        self.query_one(
            "#speaker-content",
            Static,
        ).update(
            self._format_speakers(view)
        )

        self.query_one(
            "#status",
            Static,
        ).update(
            f"{view.start:.3f}-{view.end:.3f}  "
            f"{view.duration:.3f}s  "
            f"language={view.language or '-'}"
        )

        self.refresh_bindings()

    def _format_evidence(
        self,
        view: ReviewerTurnView,
    ) -> str:
        lines = [
            (
                "Review:    "
                f"{view.review_status}"
            ),
            (
                "Boundary:  "
                f"{view.boundary.status}"
            ),
            (
                "Assigned:  "
                f"{view.assignment.status}"
            ),
            (
                "Auto:      "
                f"{view.automatic_pipeline.status or '-'}"
            ),
        ]

        if view.automatic_pipeline.review_reasons:
            lines.append(
                "Reasons:   "
                + ", ".join(
                    view.automatic_pipeline.review_reasons
                )
            )

        lines.extend([
            "",
            (
                "Speaker:   "
                f"{view.speaker_evidence.speaker or '-'}"
            ),
            (
                "Known:     "
                + (
                    ", ".join(
                        view.speaker_evidence.known_speakers
                    )
                    or "-"
                )
            ),
            "",
            (
                "Alignment: "
                f"{view.alignment.status or '-'}"
            ),
        ])

        if view.alignment.issue_word_indices:
            lines.append(
                "Words:     "
                + ", ".join(
                    str(index)
                    for index
                    in view.alignment.issue_word_indices
                )
            )

        if view.alignment.recovery is not None:
            recovery = view.alignment.recovery

            lines.extend([
                "",
                (
                    "Recovery:  "
                    f"{recovery.status or '-'}"
                ),
                (
                    "Range:     "
                    f"{recovery.source_start or 0:.3f}-"
                    f"{recovery.source_end or 0:.3f}"
                ),
            ])

        if view.edge_recovery is not None:
            edge = view.edge_recovery

            lines.extend([
                "",
                (
                    "Edge:      "
                    f"{edge.edge or '-'}"
                ),
                (
                    "Edge state:"
                    f" {edge.status or '-'}"
                ),
            ])

        return "\n".join(lines)

    def _format_speakers(
        self,
        view: ReviewerTurnView,
    ) -> str:
        if not view.speaker_candidates:
            return "No speaker candidates."

        lines: list[str] = []

        for index, candidate in enumerate(
            view.speaker_candidates,
            start=1,
        ):
            name = (
                candidate.character
                or candidate.voice_id
            )

            lines.append(
                f"{index}. {name}"
            )
            lines.append(
                f"   {candidate.voice_id}"
            )
            lines.append(
                f"   Score: {candidate.score:.3f}"
            )

            for embedding in candidate.embedding_scores:
                lines.append(
                    "   "
                    f"{embedding.embedding_name}: "
                    f"{embedding.score:.3f} "
                    f"(n={embedding.support})"
                )

            lines.append("")

        return "\n".join(lines).rstrip()

    def action_show_help(self) -> None:
        self.push_screen(HelpScreen())


def run_reviewer_tui(
    session: ReviewerSession,
    *,
    embedding_names: tuple[str, ...],
    context_padding: float,
) -> None:
    ReviewerTUI(
        session,
        embedding_names=embedding_names,
        context_padding=context_padding,
    ).run()

from __future__ import annotations

from typing import Any

from .reconciliation import edit_turn
from .review import (
    mark_turn_boundary_clipped,
    mark_turn_boundary_complete,
    mark_turn_pending,
    mark_turn_reviewed,
)
from .reviewer import sorted_turns
from .storage import DatasetStorage
from .turn_curation import (
    accept_alignment_recovery_and_prepare,
    accept_edge_recovery_and_prepare,
    merge_and_prepare_turns,
    split_and_prepare_turn,
)
from .voices import (
    assign_turn,
    ignore_turn,
    mark_turn_unknown,
)


class ReviewerSession:
    def __init__(
        self,
        storage: DatasetStorage,
        *,
        source_id: str | None = None,
        auto_review_only: bool = False,
    ) -> None:
        self.storage = storage
        self.source_id = source_id
        self.auto_review_only = auto_review_only
        self._turn_ids: list[str] = []
        self._current_turn_id: str | None = None

        self.refresh()

    def _matches_filter(
        self,
        turn: dict[str, Any],
    ) -> bool:
        if not self.auto_review_only:
            return True

        return (
            turn.get("metadata", {})
            .get("automatic_pipeline", {})
            .get("status")
            == "review"
        )

    def refresh(
        self,
        *,
        anchor_turn_id: str | None = None,
    ) -> None:
        previous_id = (
            anchor_turn_id
            if anchor_turn_id is not None
            else self._current_turn_id
        )

        previous_index: int | None = None

        if previous_id in self._turn_ids:
            previous_index = self._turn_ids.index(
                previous_id
            )

        turns = [
            turn
            for turn in sorted_turns(
                self.storage,
                source_id=self.source_id,
            )
            if self._matches_filter(turn)
        ]

        self._turn_ids = [
            str(turn["id"])
            for turn in turns
        ]

        if not self._turn_ids:
            self._current_turn_id = None
            return

        if previous_id in self._turn_ids:
            self._current_turn_id = previous_id
            return

        if previous_index is not None:
            index = min(
                previous_index,
                len(self._turn_ids) - 1,
            )
            self._current_turn_id = self._turn_ids[
                index
            ]
            return

        self._current_turn_id = self._turn_ids[0]

    @property
    def turn_ids(self) -> tuple[str, ...]:
        return tuple(self._turn_ids)

    @property
    def total(self) -> int:
        return len(self._turn_ids)

    @property
    def current_turn_id(self) -> str | None:
        return self._current_turn_id

    @property
    def position(self) -> int | None:
        if self._current_turn_id is None:
            return None

        return (
            self._turn_ids.index(
                self._current_turn_id
            )
            + 1
        )

    def current(self) -> dict[str, Any] | None:
        if self._current_turn_id is None:
            return None

        turn = self.storage.get_turn(
            self._current_turn_id
        )

        if turn is None:
            self.refresh()
            if self._current_turn_id is None:
                return None

            turn = self.storage.get_turn(
                self._current_turn_id
            )

        return turn

    def next(self) -> dict[str, Any] | None:
        if self._current_turn_id is None:
            return None

        index = self._turn_ids.index(
            self._current_turn_id
        )

        if index < len(self._turn_ids) - 1:
            self._current_turn_id = (
                self._turn_ids[index + 1]
            )

        return self.current()

    def previous(self) -> dict[str, Any] | None:
        if self._current_turn_id is None:
            return None

        index = self._turn_ids.index(
            self._current_turn_id
        )

        if index > 0:
            self._current_turn_id = (
                self._turn_ids[index - 1]
            )

        return self.current()

    def _require_current_id(self) -> str:
        if self._current_turn_id is None:
            raise ValueError(
                "Reviewer session has no current turn"
            )

        return self._current_turn_id

    def edit_transcript(
        self,
        transcript: str,
    ) -> dict[str, Any]:
        return edit_turn(
            self.storage,
            self._require_current_id(),
            transcript=transcript,
        )

    def set_language(
        self,
        language: str,
    ) -> dict[str, Any]:
        return edit_turn(
            self.storage,
            self._require_current_id(),
            language=language,
        )

    def assign_voice(
        self,
        voice_id: str,
    ) -> dict[str, Any]:
        return assign_turn(
            self.storage,
            self._require_current_id(),
            voice_id,
        )

    def mark_unknown(self) -> dict[str, Any]:
        return mark_turn_unknown(
            self.storage,
            self._require_current_id(),
        )

    def ignore(self) -> dict[str, Any]:
        return ignore_turn(
            self.storage,
            self._require_current_id(),
        )

    def mark_reviewed(self) -> dict[str, Any]:
        return mark_turn_reviewed(
            self.storage,
            self._require_current_id(),
        )

    def mark_pending(self) -> dict[str, Any]:
        return mark_turn_pending(
            self.storage,
            self._require_current_id(),
        )

    def mark_boundary_complete(
        self,
    ) -> dict[str, Any]:
        return mark_turn_boundary_complete(
            self.storage,
            self._require_current_id(),
        )

    def mark_boundary_clipped(
        self,
    ) -> dict[str, Any]:
        return mark_turn_boundary_clipped(
            self.storage,
            self._require_current_id(),
        )

    def accept_alignment_recovery(
        self,
    ) -> dict[str, Any]:
        turn_id = self._require_current_id()

        updated = (
            accept_alignment_recovery_and_prepare(
                self.storage,
                turn_id,
            )
        )

        self.refresh(
            anchor_turn_id=updated["id"],
        )

        return updated

    def accept_edge_recovery(
        self,
    ) -> dict[str, Any]:
        turn_id = self._require_current_id()

        updated = accept_edge_recovery_and_prepare(
            self.storage,
            turn_id,
        )

        self.refresh(
            anchor_turn_id=updated["id"],
        )

        return updated


    def merge(
        self,
        turn_ids: list[str],
    ) -> dict[str, Any]:
        merged = merge_and_prepare_turns(
            self.storage,
            turn_ids,
        )

        self.refresh(
            anchor_turn_id=merged["id"],
        )

        return merged

    def merge_with_next(
        self,
    ) -> dict[str, Any]:
        turn_id = self._require_current_id()

        turns = sorted_turns(
            self.storage,
            source_id=self.source_id,
        )
        turn_ids = [
            str(turn["id"])
            for turn in turns
        ]

        try:
            index = turn_ids.index(turn_id)
        except ValueError as exc:
            raise RuntimeError(
                "Current turn is missing from "
                "canonical turn timeline"
            ) from exc

        if index >= len(turn_ids) - 1:
            raise ValueError(
                "Current turn has no next turn"
            )

        next_turn_id = turn_ids[index + 1]

        return self.merge([
            turn_id,
            next_turn_id,
        ])

    def split(
        self,
        *,
        after_region_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        turn_id = self._require_current_id()

        left, right = split_and_prepare_turn(
            self.storage,
            turn_id,
            after_region_id=after_region_id,
        )

        self.refresh(
            anchor_turn_id=left["id"],
        )

        return left, right

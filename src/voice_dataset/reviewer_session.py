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
from .reviewer_view import (
    ReviewerTurnView,
    build_reviewer_turn_view,
)
from .storage import DatasetStorage
from .turn_curation import (
    accept_alignment_recovery_and_prepare,
    accept_edge_recovery_and_prepare,
    merge_and_prepare_turns,
    split_and_prepare_turn,
    trim_and_prepare_turn,
    mark_turn_accepted,
    mark_turn_curation_pending,
    mark_turn_rejected,
)
from .voices import (
    assign_turn,
    create_voice,
    mark_turn_unknown,
    set_voice_ignored,
)
from .speaker_candidates import (
    SpeakerCandidate,
    aggregate_voice_matches,
    combine_embedding_candidates,
)
from .speaker_calibration import (
    SpeakerCalibrationRule,
    build_calibration_observation,
    classify_speaker_review_mode,
    load_calibration_observations,
    speaker_calibration_observation_to_dict,
)
from .speaker_similarity import rank_voice_matches


class ReviewerSession:
    def __init__(
        self,
        storage: DatasetStorage,
        *,
        source_id: str | None = None,
        auto_review_only: bool = False,
        embedding_names: tuple[str, ...] = (),
        suggest_rule: SpeakerCalibrationRule | None = None,
        prefill_rule: SpeakerCalibrationRule | None = None,
    ) -> None:
        self.storage = storage
        self.source_id = source_id
        self.auto_review_only = auto_review_only
        self.embedding_names = embedding_names
        self.suggest_rule = suggest_rule
        self.prefill_rule = prefill_rule

        if (
            (suggest_rule is None)
            != (prefill_rule is None)
        ):
            raise ValueError(
                "suggest_rule and prefill_rule "
                "must be configured together"
            )
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

    def select_turn(
        self,
        turn_id: str,
    ) -> dict[str, Any]:
        if turn_id not in self._turn_ids:
            raise ValueError(
                f"Turn is not in the current reviewer queue: "
                f"{turn_id}"
            )

        self._current_turn_id = turn_id

        turn = self.current()

        if turn is None:
            raise RuntimeError(
                f"Selected turn disappeared: {turn_id}"
            )

        return turn

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

    def _turn_review_is_pending(
        self,
        turn_id: str,
    ) -> bool:
        turn = self.storage.get_turn(
            turn_id
        )

        if turn is None:
            return False

        review = turn.get("review")

        if not isinstance(review, dict):
            return True

        return (
            review.get("status", "pending")
            == "pending"
        )

    def next_pending(
        self,
    ) -> dict[str, Any] | None:
        if self._current_turn_id is None:
            return None

        index = self._turn_ids.index(
            self._current_turn_id
        )

        for turn_id in self._turn_ids[
            index + 1:
        ]:
            if self._turn_review_is_pending(
                turn_id
            ):
                self._current_turn_id = turn_id
                break

        return self.current()

    def previous_pending(
        self,
    ) -> dict[str, Any] | None:
        if self._current_turn_id is None:
            return None

        index = self._turn_ids.index(
            self._current_turn_id
        )

        for turn_id in reversed(
            self._turn_ids[:index]
        ):
            if self._turn_review_is_pending(
                turn_id
            ):
                self._current_turn_id = turn_id
                break

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

    def create_and_assign_voice(
        self,
        *,
        character: str | None = None,
        language: str | None = None,
        ignored: bool = False,
    ) -> dict[str, Any]:
        turn_id = self._require_current_id()

        voice = create_voice(
            self.storage,
            character=character,
            language=language,
            ignored=ignored,
        )

        assign_turn(
            self.storage,
            turn_id,
            voice["id"],
        )

        return voice

    def ignore_assigned_voice(
        self,
    ) -> dict[str, Any]:
        turn = self.current()

        if turn is None:
            raise ValueError(
                "Reviewer session has no current turn"
            )

        assignment = turn.get("assignment") or {}

        if assignment.get("status") != "assigned":
            raise ValueError(
                "Current turn has no assigned voice"
            )

        voice_id = assignment.get("voice_id")

        if not isinstance(voice_id, str) or not voice_id:
            raise ValueError(
                "Current turn has invalid assigned voice"
            )

        return set_voice_ignored(
            self.storage,
            voice_id,
            True,
        )

    def mark_unknown(self) -> dict[str, Any]:
        return mark_turn_unknown(
            self.storage,
            self._require_current_id(),
        )

    def reject(self) -> dict[str, Any]:
        return mark_turn_rejected(
            self.storage,
            self._require_current_id(),
        )

    def mark_reviewed(self) -> dict[str, Any]:
        turn_id = self._require_current_id()

        turn = self.current()

        if turn is None:
            raise RuntimeError(
                "Current turn is missing from storage"
            )

        assignment = turn.get("assignment") or {}
        calibration = None

        if assignment.get("status") == "assigned":
            voice_id = assignment.get("voice_id")
            source_id = turn.get("source_id")

            if not isinstance(
                voice_id,
                str,
            ) or not voice_id:
                raise RuntimeError(
                    "Assigned turn has invalid voice_id"
                )

            if not isinstance(
                source_id,
                str,
            ) or not source_id:
                raise RuntimeError(
                    "Current turn has invalid source_id"
                )

            candidates = self.speaker_candidates(
                embedding_names=self.embedding_names,
            )

            observation = (
                build_calibration_observation(
                    turn_id=turn_id,
                    source_id=source_id,
                    confirmed_voice_id=voice_id,
                    candidates=candidates,
                )
            )

            calibration = (
                speaker_calibration_observation_to_dict(
                    observation
                )
            )

        curation = turn.get("curation") or {}

        if curation.get("status") != "rejected":
            mark_turn_accepted(
                self.storage,
                turn_id,
            )

        mark_turn_reviewed(
            self.storage,
            turn_id,
        )

        def update(
            record: dict[str, Any],
        ) -> dict[str, Any]:
            review = dict(
                record.get("review") or {}
            )

            if calibration is None:
                review.pop(
                    "speaker_calibration",
                    None,
                )
            else:
                review["speaker_calibration"] = (
                    calibration
                )

            record["review"] = review
            return record

        return self.storage.update_turn(
            turn_id,
            update,
        )

    def mark_pending(self) -> dict[str, Any]:
        turn_id = self._require_current_id()

        mark_turn_pending(
            self.storage,
            turn_id,
        )

        mark_turn_curation_pending(
            self.storage,
            turn_id,
        )

        def update(
            record: dict[str, Any],
        ) -> dict[str, Any]:
            review = dict(
                record.get("review") or {}
            )

            review.pop(
                "speaker_calibration",
                None,
            )

            record["review"] = review
            return record

        return self.storage.update_turn(
            turn_id,
            update,
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
            capture_output=True,
        )

        self.refresh(
            anchor_turn_id=merged["id"],
        )

        return merged

    def merge_with_next(
        self,
    ) -> dict[str, Any]:
        turn_id = self._require_current_id()

        turn = self.storage.get_turn(
            turn_id
        )

        if turn is None:
            raise RuntimeError(
                "Current turn is missing from storage"
            )

        source_id = turn.get("source_id")

        if not isinstance(source_id, str) or not source_id:
            raise RuntimeError(
                "Current turn has invalid source_id"
            )

        turns = sorted_turns(
            self.storage,
            source_id=source_id,
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
            capture_output=True,
        )

        self.refresh(
            anchor_turn_id=left["id"],
        )

        return left, right

    def trim(
        self,
        *,
        source_start: float | None = None,
        source_end: float | None = None,
    ) -> dict[str, Any]:
        turn_id = self._require_current_id()

        updated = trim_and_prepare_turn(
            self.storage,
            turn_id,
            source_start=source_start,
            source_end=source_end,
        )

        self.refresh(
            anchor_turn_id=updated["id"],
        )

        return updated

    def speaker_candidates(
        self,
        *,
        embedding_names: tuple[str, ...],
        limit: int | None = None,
    ) -> list[SpeakerCandidate]:
        if limit is not None and limit <= 0:
            raise ValueError(
                "limit must be positive"
            )

        turn = self.current()

        if turn is None:
            return []

        embeddings = turn.get("embeddings") or {}

        groups = []

        for embedding_name in embedding_names:
            if embedding_name not in embeddings:
                continue

            matches = rank_voice_matches(
                self.storage,
                turn["id"],
                embedding_name,
            )

            groups.append(
                aggregate_voice_matches(
                    matches,
                    embedding_name=embedding_name,
                )
            )

        candidates = combine_embedding_candidates(
            *groups,
        )

        if limit is not None:
            return candidates[:limit]

        return candidates

    def current_view(
        self,
        *,
        embedding_names: tuple[str, ...],
        speaker_limit: int | None = None,
    ) -> ReviewerTurnView | None:
        turn = self.current()

        if turn is None:
            return None

        candidates = self.speaker_candidates(
            embedding_names=embedding_names,
        )

        speaker_review_mode = "none"

        if (
            self.suggest_rule is not None
            and self.prefill_rule is not None
        ):
            observations = (
                load_calibration_observations(
                    self.storage
                )
            )

            speaker_review_mode = (
                classify_speaker_review_mode(
                    candidates=candidates,
                    observations=observations,
                    suggest_rule=self.suggest_rule,
                    prefill_rule=self.prefill_rule,
                )
            )

        display_candidates = candidates

        if speaker_limit is not None:
            display_candidates = candidates[
                :speaker_limit
            ]

        voices = {
            str(voice["id"]): voice
            for voice in self.storage.voices.load()
        }

        return build_reviewer_turn_view(
            turn,
            position=self.position,
            total=self.total,
            voices=voices,
            speaker_candidates=display_candidates,
            speaker_review_mode=speaker_review_mode,
        )

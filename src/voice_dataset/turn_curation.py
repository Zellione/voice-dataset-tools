from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .storage import DatasetStorage
from .turn_transcripts import (
    TurnTranscriptProjection,
    project_continuous_asr_to_range,
)
from .utterance_pipeline import (
    prepare_source_review_audio,
    prepare_source_speaker_evidence,
    mark_source_utterance_reconciliation_curated,
    require_source_utterance_reconciliation,
)
from .reconciliation import (
    merge_turns,
    split_turn,
    split_turn_ranges,
)
from .review import (
    accept_alignment_recovery,
    accept_edge_recovery,
)

@dataclass(frozen=True)
class CuratedTurnProjection:
    evidence_name: str
    projection: TurnTranscriptProjection
    transcript: str | None
    language: str | None
    start_word_index: int | None
    end_word_index: int | None


def _continuous_asr_evidence(
    storage: DatasetStorage,
    source_id: str,
    *,
    evidence_name: str,
) -> dict[str, Any]:
    source = storage.get_source(source_id)

    if source is None:
        raise KeyError(
            f"Source does not exist: {source_id}"
        )

    metadata = source.get("metadata")

    if not isinstance(metadata, dict):
        raise ValueError(
            f"Source has invalid metadata: {source_id}"
        )

    continuous_asr = metadata.get(
        "continuous_asr"
    )

    if not isinstance(continuous_asr, dict):
        raise ValueError(
            "Source has no continuous ASR evidence: "
            f"{source_id}"
        )

    evidence = continuous_asr.get(evidence_name)

    if not isinstance(evidence, dict):
        raise ValueError(
            "Source has no continuous ASR evidence "
            f"{evidence_name}: {source_id}"
        )

    return evidence


def project_curated_turn(
    storage: DatasetStorage,
    *,
    source_id: str,
    source_start: float,
    source_end: float,
    evidence_name: str = "qwen3",
    language: str | None = None,
) -> CuratedTurnProjection:
    evidence = _continuous_asr_evidence(
        storage,
        source_id,
        evidence_name=evidence_name,
    )

    projection = project_continuous_asr_to_range(
        evidence,
        source_start,
        source_end,
    )

    word_indices = projection.word_indices

    if word_indices:
        expected_indices = tuple(
            range(
                word_indices[0],
                word_indices[-1] + 1,
            )
        )

        if word_indices != expected_indices:
            raise ValueError(
                "Curated turn projection contains "
                "non-contiguous continuous ASR words: "
                f"{source_id}/{evidence_name}: "
                f"{word_indices}"
            )

        start_word_index = word_indices[0]
        end_word_index = word_indices[-1]
    else:
        start_word_index = None
        end_word_index = None

    words = evidence.get("words")

    if not isinstance(words, list):
        raise ValueError(
            "Continuous ASR evidence has invalid words: "
            f"{source_id}/{evidence_name}"
        )

    selected_words: list[str] = []

    for index in projection.word_indices:
        word = words[index]

        if not isinstance(word, dict):
            raise ValueError(
                "Continuous ASR evidence has invalid word: "
                f"{source_id}/{evidence_name}/{index}"
            )

        text = word.get("text")

        if not isinstance(text, str):
            raise ValueError(
                "Continuous ASR word has invalid text: "
                f"{source_id}/{evidence_name}/{index}"
            )

        text = text.strip()

        if text:
            selected_words.append(text)

    transcript = " ".join(selected_words) or None

    return CuratedTurnProjection(
        evidence_name=evidence_name,
        projection=projection,
        transcript=transcript,
        language=language,
        start_word_index=start_word_index,
        end_word_index=end_word_index,
    )


def prepare_curated_source_turns(
    storage: DatasetStorage,
    source_id: str,
) -> None:
    prepare_source_review_audio(
        storage,
        source_id,
    )
    prepare_source_speaker_evidence(
        storage,
        source_id,
    )


def accept_alignment_recovery_and_prepare(
    storage: DatasetStorage,
    turn_id: str,
) -> dict[str, Any]:
    updated = accept_alignment_recovery(
        storage,
        turn_id,
    )

    source_id = updated.get("source_id")

    if not isinstance(source_id, str) or not source_id:
        raise ValueError(
            f"Turn has invalid source_id: {turn_id}"
        )

    prepare_curated_source_turns(
        storage,
        source_id,
    )

    stored = storage.get_turn(turn_id)

    if stored is None:
        raise RuntimeError(
            "Failed to read back recovered turn"
        )

    return stored


def accept_edge_recovery_and_prepare(
    storage: DatasetStorage,
    turn_id: str,
) -> dict[str, Any]:
    updated = accept_edge_recovery(
        storage,
        turn_id,
    )

    source_id = updated.get("source_id")

    if not isinstance(source_id, str) or not source_id:
        raise ValueError(
            f"Turn has invalid source_id: {turn_id}"
        )

    prepare_curated_source_turns(
        storage,
        source_id,
    )

    stored = storage.get_turn(turn_id)

    if stored is None:
        raise RuntimeError(
            "Failed to read back recovered turn"
        )

    return stored


def _project_merge(
    storage: DatasetStorage,
    turn_ids: list[str],
    *,
    evidence_name: str,
) -> CuratedTurnProjection:
    if len(turn_ids) < 2:
        raise ValueError(
            "At least two turns are required to merge"
        )

    turns: list[dict[str, Any]] = []

    for turn_id in turn_ids:
        turn = storage.get_turn(turn_id)

        if turn is None:
            raise KeyError(
                f"Turn does not exist: {turn_id}"
            )

        turns.append(turn)

    source_ids = {
        turn.get("source_id")
        for turn in turns
    }

    if len(source_ids) != 1:
        raise ValueError(
            "All turns must belong to the same source"
        )

    source_id = next(iter(source_ids))

    if not isinstance(source_id, str):
        raise ValueError(
            "Turns contain an invalid source_id"
        )

    source_start = float(
        turns[0]["source_start"]
    )
    source_end = float(
        turns[-1]["source_end"]
    )

    languages = {
        turn.get("language")
        for turn in turns
        if turn.get("language")
    }

    language = (
        next(iter(languages))
        if len(languages) == 1
        else None
    )

    projection = project_curated_turn(
        storage,
        source_id=source_id,
        source_start=source_start,
        source_end=source_end,
        evidence_name=evidence_name,
        language=language,
    )

    if (
        projection.start_word_index is None
        or projection.end_word_index is None
    ):
        raise ValueError(
            "Merged turn does not project to "
            "continuous ASR words"
        )

    return projection


def _project_split(
    storage: DatasetStorage,
    turn_id: str,
    *,
    after_region_id: str,
    evidence_name: str,
) -> tuple[
    str,
    CuratedTurnProjection,
    CuratedTurnProjection,
]:
    (
        source_id,
        _left_ids,
        (left_start, left_end),
        _right_ids,
        (right_start, right_end),
    ) = split_turn_ranges(
        storage,
        turn_id,
        after_region_id=after_region_id,
    )

    original = storage.get_turn(turn_id)

    if original is None:
        raise KeyError(
            f"Turn does not exist: {turn_id}"
        )

    language = original.get("language")

    if language is not None and not isinstance(
        language,
        str,
    ):
        raise ValueError(
            f"{turn_id}: invalid language"
        )

    left = project_curated_turn(
        storage,
        source_id=source_id,
        source_start=left_start,
        source_end=left_end,
        evidence_name=evidence_name,
        language=language,
    )

    right = project_curated_turn(
        storage,
        source_id=source_id,
        source_start=right_start,
        source_end=right_end,
        evidence_name=evidence_name,
        language=language,
    )

    for side, projection in (
        ("left", left),
        ("right", right),
    ):
        if (
            projection.start_word_index is None
            or projection.end_word_index is None
        ):
            raise ValueError(
                f"{side.capitalize()} split turn "
                "does not project to continuous "
                "ASR words"
            )

    return source_id, left, right


def _apply_projection(
    storage: DatasetStorage,
    turn_id: str,
    projection: CuratedTurnProjection,
    *,
    set_transcript: bool = False,
    set_language: bool = False,
) -> dict[str, Any]:
    if (
        projection.start_word_index is None
        or projection.end_word_index is None
    ):
        raise ValueError(
            f"{turn_id}: projection has no words"
        )

    def update(
        record: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = record.setdefault(
            "metadata",
            {},
        )

        if not isinstance(metadata, dict):
            raise ValueError(
                f"{turn_id}: invalid metadata"
            )

        metadata["word_range"] = {
            "start": projection.start_word_index,
            "end": projection.end_word_index,
        }

        metadata["continuous_asr"] = {
            "evidence": projection.evidence_name,
            "word_indices": list(
                projection.projection.word_indices
            ),
        }

        if set_transcript:
            record["transcript"] = (
                projection.transcript
            )

        if set_language:
            record["language"] = (
                projection.language
            )

        return record

    return storage.update_turn(
        turn_id,
        update,
    )


def merge_and_prepare_turns(
    storage: DatasetStorage,
    turn_ids: list[str],
    *,
    evidence_name: str = "qwen3",
) -> dict[str, Any]:
    projection = _project_merge(
        storage,
        turn_ids,
        evidence_name=evidence_name,
    )

    source_id = storage.get_turn(
        turn_ids[0]
    )["source_id"]

    require_source_utterance_reconciliation(
        storage,
        source_id,
        asr_evidence_name=evidence_name,
    )

    merged = merge_turns(
        storage,
        turn_ids,
    )

    source_id = merged.get("source_id")

    if not isinstance(source_id, str):
        raise ValueError(
            "Merged turn has invalid source_id"
        )

    merged = _apply_projection(
        storage,
        merged["id"],
        projection,
    )

    mark_source_utterance_reconciliation_curated(
        storage,
        source_id,
        asr_evidence_name=evidence_name,
    )

    prepare_curated_source_turns(
        storage,
        source_id,
    )

    stored = storage.get_turn(
        merged["id"]
    )

    if stored is None:
        raise RuntimeError(
            "Failed to read back prepared merged turn"
        )

    return stored


def split_and_prepare_turn(
    storage: DatasetStorage,
    turn_id: str,
    *,
    after_region_id: str,
    evidence_name: str = "qwen3",
) -> tuple[dict[str, Any], dict[str, Any]]:
    (
        source_id,
        left_projection,
        right_projection,
    ) = _project_split(
        storage,
        turn_id,
        after_region_id=after_region_id,
        evidence_name=evidence_name,
    )

    require_source_utterance_reconciliation(
        storage,
        source_id,
        asr_evidence_name=evidence_name,
    )

    left, right = split_turn(
        storage,
        turn_id,
        after_region_id=after_region_id,
    )

    left = _apply_projection(
        storage,
        left["id"],
        left_projection,
        set_transcript=True,
        set_language=True,
    )

    right = _apply_projection(
        storage,
        right["id"],
        right_projection,
        set_transcript=True,
        set_language=True,
    )

    mark_source_utterance_reconciliation_curated(
        storage,
        source_id,
        asr_evidence_name=evidence_name,
    )

    prepare_curated_source_turns(
        storage,
        source_id,
    )

    stored_left = storage.get_turn(
        left["id"]
    )
    stored_right = storage.get_turn(
        right["id"]
    )

    if (
        stored_left is None
        or stored_right is None
    ):
        raise RuntimeError(
            "Failed to read back prepared split turns"
        )

    return stored_left, stored_right


def trim_and_prepare_turn(
    storage: DatasetStorage,
    turn_id: str,
    *,
    source_start: float | None = None,
    source_end: float | None = None,
    evidence_name: str = "qwen3",
) -> dict[str, Any]:
    turn = storage.get_turn(turn_id)

    if turn is None:
        raise KeyError(
            f"Turn does not exist: {turn_id}"
        )

    current_start = float(
        turn["source_start"]
    )
    current_end = float(
        turn["source_end"]
    )

    new_start = (
        current_start
        if source_start is None
        else float(source_start)
    )
    new_end = (
        current_end
        if source_end is None
        else float(source_end)
    )

    if new_start >= new_end:
        raise ValueError(
            "source_start must be before source_end"
        )

    if (
        new_start < current_start
        or new_end > current_end
    ):
        raise ValueError(
            "Trim must stay within current turn range"
        )

    source_id = turn.get("source_id")

    if not isinstance(source_id, str) or not source_id:
        raise ValueError(
            f"Turn has invalid source_id: {turn_id}"
        )

    require_source_utterance_reconciliation(
        storage,
        source_id,
        asr_evidence_name=evidence_name,
    )

    language = turn.get("language")

    if language is not None and not isinstance(
        language,
        str,
    ):
        raise ValueError(
            f"{turn_id}: invalid language"
        )

    current_projection = project_curated_turn(
        storage,
        source_id=source_id,
        source_start=current_start,
        source_end=current_end,
        evidence_name=evidence_name,
        language=language,
    )

    trimmed_projection = project_curated_turn(
        storage,
        source_id=source_id,
        source_start=new_start,
        source_end=new_end,
        evidence_name=evidence_name,
        language=language,
    )

    if (
        current_projection.projection.word_indices
        != trimmed_projection.projection.word_indices
    ):
        raise ValueError(
            "Trim would change continuous ASR words"
        )

    def update(
        record: dict[str, Any],
    ) -> dict[str, Any]:
        record["source_start"] = new_start
        record["source_end"] = new_end
        return record

    storage.update_turn(
        turn_id,
        update,
    )

    _apply_projection(
        storage,
        turn_id,
        trimmed_projection,
    )

    prepare_curated_source_turns(
        storage,
        source_id,
    )

    stored = storage.get_turn(turn_id)

    if stored is None:
        raise RuntimeError(
            "Failed to read back trimmed turn"
        )

    return stored

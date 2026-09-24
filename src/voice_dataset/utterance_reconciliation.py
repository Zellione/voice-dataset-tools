from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import TurnRecord
from .speaker_words import SpeakerAttributedWord
from .storage import DatasetStorage
from .utterance_candidates import UtteranceCandidate


@dataclass(frozen=True)
class UtteranceTurnResult:
    created: int
    skipped: int
    unresolved: int


def _source_regions(
    storage: DatasetStorage,
    source_id: str,
) -> list[dict[str, Any]]:
    regions = [
        region
        for region in storage.regions.load()
        if region.get("source_id") == source_id
    ]

    regions.sort(
        key=lambda region: (
            region["source_start"],
            region["source_end"],
            region["id"],
        )
    )

    return regions


def _overlapping_region_ids(
    regions: list[dict[str, Any]],
    *,
    start: float,
    end: float,
) -> list[str]:
    result: list[str] = []

    for region in regions:
        region_start = float(
            region["source_start"]
        )
        region_end = float(
            region["source_end"]
        )

        if min(end, region_end) <= max(
            start,
            region_start,
        ):
            continue

        result.append(region["id"])

    return result


def _lexical_tokens(
    text: str | None,
) -> set[str]:
    if not text:
        return set()

    return {
        "".join(
            character
            for character in token.casefold()
            if character.isalnum()
            or character == "'"
        )
        for token in text.split()
    } - {""}


def _whisper_text(
    region: dict[str, Any],
) -> str | None:
    transcripts = region.get("transcripts")

    if not isinstance(transcripts, dict):
        return None

    whisper = transcripts.get("whisper")

    if not isinstance(whisper, dict):
        return None

    text = whisper.get("text")

    if not isinstance(text, str):
        return None

    text = text.strip()

    return text or None


def _turn_edge_evidence(
    regions: list[dict[str, Any]],
    *,
    candidate: UtteranceCandidate,
) -> dict[str, Any] | None:
    if candidate.alignment_issue_word_indices:
        return None

    def lexical_tokens(
        text: str,
    ) -> list[str]:
        return [
            normalized
            for token in text.split()
            if (
                normalized := "".join(
                    character
                    for character
                    in token.casefold()
                    if (
                        character.isalnum()
                        or character == "'"
                    )
                )
            )
        ]

    candidate_tokens = lexical_tokens(
        candidate.text
    )

    if not candidate_tokens:
        return None

    end_suggestions: list[
        dict[str, Any]
    ] = []

    for region in regions:
        region_start = float(
            region["source_start"]
        )
        region_end = float(
            region["source_end"]
        )

        if not (
            region_start < candidate.end
            < region_end
        ):
            continue

        whisper_text = _whisper_text(region)

        if whisper_text is None:
            continue

        whisper_tokens = lexical_tokens(
            whisper_text
        )

        if not whisper_tokens:
            continue

        limit = min(
            len(candidate_tokens),
            len(whisper_tokens),
        )

        mismatch = next(
            (
                index
                for index in range(limit)
                if (
                    candidate_tokens[index]
                    != whisper_tokens[index]
                )
            ),
            None,
        )

        if (
            mismatch is None
            or mismatch
            != len(candidate_tokens) - 1
        ):
            continue

        candidate_last = (
            candidate_tokens[-1]
        )
        whisper_last = (
            whisper_tokens[mismatch]
        )

        if not (
            len(whisper_last)
            > len(candidate_last)
            and whisper_last.startswith(
                candidate_last
            )
        ):
            continue

        end_suggestions.append(
            {
                "status": "suggested",
                "method": (
                    "community_whisper_"
                    "edge_extension"
                ),
                "edge": "end",
                "source_end": region_end,
                "region_id": region["id"],
                "speaker": region.get(
                    "detector_label"
                ),
                "candidate_token": (
                    candidate_last
                ),
                "whisper_token": (
                    whisper_last
                ),
                "candidate_text": (
                    candidate.text
                ),
                "whisper_text": (
                    whisper_text
                ),
            }
        )

    if len(end_suggestions) == 1:
        return end_suggestions[0]

    containing_start_regions = [
        region
        for region in regions
        if (
            float(region["source_start"])
            < candidate.start
            < float(region["source_end"])
        )
    ]

    if len(containing_start_regions) != 1:
        return None

    conflicting_region = (
        containing_start_regions[0]
    )
    conflicting_text = _whisper_text(
        conflicting_region
    )

    if conflicting_text is None:
        return None

    conflicting_tokens = lexical_tokens(
        conflicting_text
    )

    if (
        conflicting_tokens
        and conflicting_tokens[0]
        == candidate_tokens[0]
    ):
        return None

    start_suggestions: list[
        dict[str, Any]
    ] = []

    for region in regions:
        region_start = float(
            region["source_start"]
        )
        region_end = float(
            region["source_end"]
        )

        if not (
            candidate.start + 0.001
            < region_start
            < candidate.end
        ):
            continue

        whisper_text = _whisper_text(region)

        if whisper_text is None:
            continue

        whisper_tokens = lexical_tokens(
            whisper_text
        )

        if not whisper_tokens:
            continue

        if (
            whisper_tokens[0]
            != candidate_tokens[0]
        ):
            continue

        if (
            len(candidate_tokens) > 1
            and len(whisper_tokens) > 1
            and (
                whisper_tokens[1]
                != candidate_tokens[1]
            )
        ):
            continue

        start_suggestions.append(
            {
                "status": "suggested",
                "method": (
                    "community_whisper_"
                    "start_conflict"
                ),
                "edge": "start",
                "source_start": region_start,
                "region_id": region["id"],
                "conflicting_region_id": (
                    conflicting_region["id"]
                ),
                "speaker": region.get(
                    "detector_label"
                ),
                "candidate_text": (
                    candidate.text
                ),
                "whisper_text": (
                    whisper_text
                ),
                "conflicting_whisper_text": (
                    conflicting_text
                ),
            }
        )

    if len(start_suggestions) != 1:
        return None

    return start_suggestions[0]


def _alignment_recovery_evidence(
    regions: list[dict[str, Any]],
    *,
    candidate: UtteranceCandidate,
) -> dict[str, Any] | None:
    if not candidate.alignment_issue_word_indices:
        return None

    candidate_tokens = _lexical_tokens(
        candidate.text
    )

    if not candidate_tokens:
        return None

    overlapping = [
        region
        for region in regions
        if min(
            candidate.end,
            float(region["source_end"]),
        )
        > max(
            candidate.start,
            float(region["source_start"]),
        )
    ]

    groups: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for region in overlapping:
        speaker = region.get(
            "detector_label"
        )
        text = _whisper_text(region)

        if (
            not isinstance(speaker, str)
            or not speaker
            or text is None
        ):
            continue

        shared_tokens = (
            candidate_tokens
            & _lexical_tokens(text)
        )

        if not shared_tokens:
            continue

        groups.setdefault(
            speaker,
            [],
        ).append(region)

    if not groups:
        return None

    ranked: list[
        tuple[
            int,
            float,
            str,
            list[dict[str, Any]],
            set[str],
        ]
    ] = []

    for speaker, speaker_regions in (
        groups.items()
    ):
        matched_tokens: set[str] = set()

        for region in speaker_regions:
            matched_tokens.update(
                candidate_tokens
                & _lexical_tokens(
                    _whisper_text(region)
                )
            )

        start = min(
            float(region["source_start"])
            for region in speaker_regions
        )
        end = max(
            float(region["source_end"])
            for region in speaker_regions
        )

        ranked.append(
            (
                len(matched_tokens),
                -(end - start),
                speaker,
                speaker_regions,
                matched_tokens,
            )
        )

    ranked.sort(reverse=True)

    (
        matched_count,
        _,
        speaker,
        selected_regions,
        matched_tokens,
    ) = ranked[0]

    start = min(
        float(region["source_start"])
        for region in selected_regions
    )
    end = max(
        float(region["source_end"])
        for region in selected_regions
    )

    return {
        "status": "suggested",
        "method": (
            "community_whisper_lexical_overlap"
        ),
        "speaker": speaker,
        "source_start": start,
        "source_end": end,
        "region_ids": [
            region["id"]
            for region in selected_regions
        ],
        "matched_tokens": sorted(
            matched_tokens
        ),
        "matched_token_count": matched_count,
        "candidate_token_count": len(
            candidate_tokens
        ),
        "whisper_texts": [
            {
                "region_id": region["id"],
                "text": _whisper_text(region),
            }
            for region in selected_regions
        ],
    }


def _existing_utterance_turn(
    storage: DatasetStorage,
    *,
    source_id: str,
    start_word_index: int,
    end_word_index: int,
) -> dict[str, Any] | None:
    matches: list[dict[str, Any]] = []

    for turn in storage.turns.load():
        if turn.get("source_id") != source_id:
            continue

        metadata = turn.get("metadata")

        if not isinstance(metadata, dict):
            continue

        creation = metadata.get("creation")

        if not isinstance(creation, dict):
            continue

        if (
            creation.get("method")
            != "continuous_asr_utterance"
        ):
            continue

        word_range = metadata.get(
            "word_range"
        )

        if not isinstance(word_range, dict):
            continue

        if (
            word_range.get("start")
            == start_word_index
            and word_range.get("end")
            == end_word_index
        ):
            matches.append(turn)

    if len(matches) > 1:
        raise RuntimeError(
            "Multiple utterance turns exist for "
            f"{source_id} words "
            f"{start_word_index}-{end_word_index}"
        )

    if matches:
        return matches[0]

    return None


def create_turn_from_utterance(
    storage: DatasetStorage,
    *,
    source_id: str,
    candidate: UtteranceCandidate,
    words: list[SpeakerAttributedWord],
    language: str | None,
    pipeline_status: str,
    review_reasons: tuple[str, ...] = (),
    asr_evidence_name: str = "qwen3",
    segmentation_evidence_name: str = "sat",
) -> dict[str, Any]:
    if candidate.end <= candidate.start:
        raise ValueError(
            "Utterance candidate has invalid geometry: "
            f"{candidate.start}-{candidate.end}"
        )

    indexed_words = {
        word.index: word
        for word in words
    }

    candidate_words = [
        indexed_words[index]
        for index in candidate.word_indices
    ]

    regions = _source_regions(
        storage,
        source_id,
    )

    alignment_recovery = (
        _alignment_recovery_evidence(
            regions,
            candidate=candidate,
        )
    )

    edge_evidence = _turn_edge_evidence(
        regions,
        candidate=candidate,
    )

    derived_metadata: dict[str, Any] = {
        "automatic_pipeline": {
            "status": pipeline_status,
            "review_reasons": list(
                review_reasons
            ),
        },
        "word_range": {
            "start": (
                candidate.start_word_index
            ),
            "end": (
                candidate.end_word_index
            ),
        },
        "utterance_segmentation": {
            "evidence": (
                segmentation_evidence_name
            ),
            "end_boundary": (
                candidate.end_boundary
            ),
        },
        "speaker_evidence": {
            "speaker": candidate.speaker,
            "known_speakers": list(
                candidate.known_speakers
            ),
            "unresolved_word_indices": list(
                candidate.unresolved_word_indices
            ),
        },
        "alignment_evidence": {
            "status": (
                "invalid"
                if candidate.alignment_issue_word_indices
                else "valid"
            ),
            "issue_word_indices": list(
                candidate.alignment_issue_word_indices
            ),
            "recovery": alignment_recovery,
        },
        "edge_evidence": edge_evidence,
        "continuous_asr": {
            "evidence": asr_evidence_name,
            "word_indices": list(
                candidate.word_indices
            ),
            "word_assignment_methods": {
                str(word.index): (
                    word.assignment_method
                )
                for word in candidate_words
            },
        },
    }

    existing = _existing_utterance_turn(
        storage,
        source_id=source_id,
        start_word_index=(
            candidate.start_word_index
        ),
        end_word_index=(
            candidate.end_word_index
        ),
    )

    if existing is not None:
        metadata = existing.get("metadata")

        if not isinstance(metadata, dict):
            metadata = {}

        updated_metadata = dict(metadata)
        updated_metadata.update(
            derived_metadata
        )

        storage.update_turn(
            existing["id"],
            lambda turn: {
                **turn,
                "metadata": updated_metadata,
            },
        )

        stored = storage.get_turn(
            existing["id"]
        )

        if stored is None:
            raise RuntimeError(
                "Failed to read back updated turn: "
                f"{existing['id']}"
            )

        return stored

    region_ids = _overlapping_region_ids(
        regions,
        start=candidate.start,
        end=candidate.end,
    )

    metadata: dict[str, Any] = {
        "creation": {
            "method": (
                "continuous_asr_utterance"
            ),
        },
        **derived_metadata,
    }

    turn = TurnRecord(
        id=storage.next_turn_id(),
        source_id=source_id,
        source_start=candidate.start,
        source_end=candidate.end,
        source_regions=region_ids,
        language=language,
        transcript=candidate.text,
        metadata=metadata,
    )

    storage.add_turn(turn)

    stored = storage.get_turn(turn.id)

    if stored is None:
        raise RuntimeError(
            "Failed to read back created turn: "
            f"{turn.id}"
        )

    return stored

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .reconciliation import (
    create_turn_from_regions,
    edit_turn,
    effective_region_reconciliation,
    merge_turns,
)
from .storage import DatasetStorage


MERGE_CANDIDATE_MAX_WORD_GAP = 0.75


@dataclass(frozen=True)
class AutomaticReconciliationResult:
    created: int
    skipped: int
    unresolved: int


@dataclass(frozen=True)
class ContinuousReconciliationBoundary:
    left_region_id: str
    right_region_id: str
    left_speaker: str | None
    right_speaker: str | None
    region_gap: float
    word_gap: float
    left_word: str
    right_word: str


@dataclass(frozen=True)
class ContinuousReconciliationCandidate:
    utterance_index: int
    text: str
    word_start: int
    word_end: int
    start: float
    end: float
    region_ids: tuple[str, ...]
    speaker_labels: tuple[str, ...]
    status: str
    reasons: tuple[str, ...]
    boundaries: tuple[
        ContinuousReconciliationBoundary,
        ...,
    ]


def analyze_continuous_reconciliation(
    storage: DatasetStorage,
    source_id: str,
    *,
    evidence_name: str = "qwen3",
) -> list[ContinuousReconciliationCandidate]:
    source = storage.get_source(source_id)

    if source is None:
        raise KeyError(
            f"Source does not exist: {source_id}"
        )

    metadata = source.get("metadata", {})

    if not isinstance(metadata, dict):
        raise ValueError(
            f"Source has invalid metadata: {source_id}"
        )

    continuous_asr = metadata.get(
        "continuous_asr",
        {},
    )

    if not isinstance(continuous_asr, dict):
        raise ValueError(
            f"Source has invalid continuous_asr metadata: "
            f"{source_id}"
        )

    evidence = continuous_asr.get(evidence_name)

    if not isinstance(evidence, dict):
        raise ValueError(
            "Continuous ASR evidence does not exist: "
            f"{source_id}/{evidence_name}"
        )

    words = evidence.get("words")
    utterances = evidence.get("utterances")

    if not isinstance(words, list):
        raise ValueError(
            "Continuous ASR words must be a list"
        )

    if not isinstance(utterances, list):
        raise ValueError(
            "Continuous ASR utterances must be a list"
        )

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

    results: list[
        ContinuousReconciliationCandidate
    ] = []

    for utterance_index, utterance in enumerate(
        utterances
    ):
        text = utterance["text"]
        word_start = int(utterance["word_start"])
        word_end = int(utterance["word_end"])

        selected_words = words[
            word_start:word_end
        ]

        if not selected_words:
            results.append(
                ContinuousReconciliationCandidate(
                    utterance_index=utterance_index,
                    text=text,
                    word_start=word_start,
                    word_end=word_end,
                    start=0.0,
                    end=0.0,
                    region_ids=(),
                    speaker_labels=(),
                    status="conflict",
                    reasons=("no_aligned_words",),
                    boundaries=(),
                )
            )
            continue

        start = float(selected_words[0]["start"])
        end = float(selected_words[-1]["end"])

        matched_regions: list[
            dict[str, Any]
        ] = []

        for region in regions:
            region_start = float(
                region["source_start"]
            )
            region_end = float(
                region["source_end"]
            )

            contains_word = False

            for word in selected_words:
                word_start_time = float(
                    word["start"]
                )
                word_end_time = float(
                    word["end"]
                )

                midpoint = (
                    word_start_time
                    + word_end_time
                ) / 2.0

                if (
                    region_start
                    <= midpoint
                    <= region_end
                ):
                    contains_word = True
                    break

            if contains_word:
                matched_regions.append(region)

        region_ids = tuple(
            region["id"]
            for region in matched_regions
        )

        speaker_labels = tuple(
            dict.fromkeys(
                region.get("detector_label")
                or "<unknown>"
                for region in matched_regions
            )
        )

        boundaries: list[
            ContinuousReconciliationBoundary
        ] = []

        for left, right in zip(
            matched_regions,
            matched_regions[1:],
        ):
            left_start = float(
                left["source_start"]
            )
            left_end = float(
                left["source_end"]
            )
            right_start = float(
                right["source_start"]
            )
            right_end = float(
                right["source_end"]
            )

            left_words = []
            right_words = []

            for word in selected_words:
                word_start_time = float(
                    word["start"]
                )
                word_end_time = float(
                    word["end"]
                )
                midpoint = (
                    word_start_time
                    + word_end_time
                ) / 2.0

                if (
                    left_start
                    <= midpoint
                    <= left_end
                ):
                    left_words.append(word)

                if (
                    right_start
                    <= midpoint
                    <= right_end
                ):
                    right_words.append(word)

            if not left_words or not right_words:
                continue

            left_word = left_words[-1]
            right_word = right_words[0]

            boundaries.append(
                ContinuousReconciliationBoundary(
                    left_region_id=left["id"],
                    right_region_id=right["id"],
                    left_speaker=left.get(
                        "detector_label"
                    ),
                    right_speaker=right.get(
                        "detector_label"
                    ),
                    region_gap=(
                        right_start - left_end
                    ),
                    word_gap=(
                        float(right_word["start"])
                        - float(left_word["end"])
                    ),
                    left_word=left_word["text"],
                    right_word=right_word["text"],
                )
            )

        reasons: list[str] = []

        if not matched_regions:
            reasons.append("no_matching_region")

        if len(speaker_labels) > 1:
            reasons.append(
                "multiple_speakers"
            )

        if end <= start:
            reasons.append(
                "invalid_alignment_range"
            )

        reconciliation_states = [
            effective_region_reconciliation(
                storage,
                region,
            )
            for region in matched_regions
        ]

        reconciled_turn_ids = [
            reconciliation.get("turn_id")
            for reconciliation
            in reconciliation_states
            if (
                reconciliation.get("status")
                == "reconciled"
                and isinstance(
                    reconciliation.get("turn_id"),
                    str,
                )
            )
        ]

        all_reconciled = (
            bool(matched_regions)
            and len(reconciled_turn_ids)
            == len(matched_regions)
        )

        same_reconciled_turn = (
            all_reconciled
            and len(set(reconciled_turn_ids)) == 1
        )

        merge_geometry_candidate = (
            len(matched_regions) > 1
            and len(boundaries)
            == len(matched_regions) - 1
            and all(
                boundary.left_speaker
                == boundary.right_speaker
                and boundary.word_gap
                <= MERGE_CANDIDATE_MAX_WORD_GAP
                for boundary in boundaries
            )
        )

        if reasons:
            status = "conflict"
        elif (
            merge_geometry_candidate
            and same_reconciled_turn
        ):
            status = "reconciled"
        elif merge_geometry_candidate:
            status = "merge_candidate"
        else:
            status = "candidate"

        results.append(
            ContinuousReconciliationCandidate(
                utterance_index=utterance_index,
                text=text,
                word_start=word_start,
                word_end=word_end,
                start=start,
                end=end,
                region_ids=region_ids,
                speaker_labels=speaker_labels,
                status=status,
                reasons=tuple(reasons),
                boundaries=tuple(boundaries),
            )
        )

    return results


def apply_continuous_merge_candidate(
    storage: DatasetStorage,
    source_id: str,
    utterance_index: int,
    *,
    evidence_name: str = "qwen3",
) -> dict[str, Any]:
    results = analyze_continuous_reconciliation(
        storage,
        source_id,
        evidence_name=evidence_name,
    )

    if (
        utterance_index < 0
        or utterance_index >= len(results)
    ):
        raise ValueError(
            "Continuous ASR utterance index out of range: "
            f"{utterance_index + 1}"
        )

    candidate = results[utterance_index]

    if candidate.status != "merge_candidate":
        raise ValueError(
            "Continuous ASR utterance is not a "
            "merge candidate: "
            f"{utterance_index + 1} "
            f"({candidate.status})"
        )

    if len(candidate.region_ids) < 2:
        raise ValueError(
            "Merge candidate must contain at least "
            "two regions"
        )

    turn_ids: list[str] = []

    for region_id in candidate.region_ids:
        region = storage.get_region(region_id)

        if region is None:
            raise KeyError(
                f"Candidate region does not exist: "
                f"{region_id}"
            )

        reconciliation = (
            effective_region_reconciliation(
                storage,
                region,
            )
        )

        if reconciliation["status"] != "reconciled":
            raise ValueError(
                f"{region_id} is not currently "
                "reconciled"
            )

        turn_id = reconciliation.get("turn_id")

        if not isinstance(turn_id, str):
            raise ValueError(
                f"{region_id} has no valid "
                "reconciled turn"
            )

        if (
            not turn_ids
            or turn_ids[-1] != turn_id
        ):
            turn_ids.append(turn_id)

    if len(turn_ids) < 2:
        raise ValueError(
            "Merge candidate does not resolve to "
            "multiple turns"
        )

    merged = merge_turns(
        storage,
        turn_ids,
    )

    source = storage.get_source(source_id)

    if source is None:
        raise KeyError(
            f"Source does not exist: {source_id}"
        )

    metadata = source.get("metadata", {})
    continuous_asr = metadata.get(
        "continuous_asr",
        {},
    )
    evidence = continuous_asr.get(
        evidence_name,
        {},
    )

    language = evidence.get("language")

    if not isinstance(language, str):
        language = None

    return edit_turn(
        storage,
        merged["id"],
        transcript=candidate.text,
        language=language,
    )


def reconcile_source_regions(
    storage: DatasetStorage,
    source_id: str,
    *,
    transcript_name: str = "whisper",
) -> AutomaticReconciliationResult:
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

    created = 0
    skipped = 0
    unresolved = 0

    for region in regions:
        reconciliation = (
            effective_region_reconciliation(
                storage,
                region,
            )
        )

        if reconciliation["state"] in {
            "reconciled",
            "rejected",
        }:
            skipped += 1
            continue

        transcripts = region.get(
            "transcripts",
            {},
        )
        transcript = transcripts.get(
            transcript_name
        )

        if not isinstance(transcript, dict):
            unresolved += 1
            continue

        text = str(
            transcript.get("text", "")
        ).strip()

        if not text:
            unresolved += 1
            continue

        language = transcript.get("language")

        create_turn_from_regions(
            storage,
            [region["id"]],
            language=(
                str(language)
                if language
                else None
            ),
            transcript=text,
            metadata={
                "creation": {
                    "method": (
                        "automatic_transcript_"
                        "reconciliation"
                    ),
                    "transcript_name": (
                        transcript_name
                    ),
                },
            },
        )

        created += 1

    return AutomaticReconciliationResult(
        created=created,
        skipped=skipped,
        unresolved=unresolved,
    )

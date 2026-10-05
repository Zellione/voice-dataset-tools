from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import tempfile

from .speaker_words import (
    SpeakerAttributedWord,
    attribute_speakers_to_words,
    resolve_consistent_speaker_context,
    resolve_fragmented_speaker_words,
)
from .storage import DatasetStorage
from .utterance_candidates import (
    UtteranceCandidate,
    build_utterance_candidates,
)
from .workers import run_worker, worker
from .embeddings import (
    EmbeddingRunResult,
    embed_turns,
)
from .representations import (
    MaterializeTurnsResult,
    materialize_turns,
)
from .sources import (
    resolve_source_representation_for_purpose,
)
from .word_alignment import (
    EffectiveWordAlignment,
    build_effective_word_alignment,
    collect_region_evidence,
    recover_sat_boundary_alignments,
    validate_sat_boundaries,
)


@dataclass(frozen=True)
class UtterancePipelineResult:
    words: tuple[SpeakerAttributedWord, ...]
    candidates: tuple[UtteranceCandidate, ...]
    sat_boundary_after_word_indices: tuple[int, ...]
    alignment: EffectiveWordAlignment


def _continuous_asr_evidence(
    storage: DatasetStorage,
    source_id: str,
    *,
    evidence_name: str,
) -> dict:
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

    if not isinstance(
        continuous_asr,
        dict,
    ):
        raise ValueError(
            "Source has no continuous ASR metadata: "
            f"{source_id}"
        )

    evidence = continuous_asr.get(
        evidence_name
    )

    if not isinstance(evidence, dict):
        raise ValueError(
            "Continuous ASR evidence does not exist: "
            f"{source_id}/{evidence_name}"
        )

    words = evidence.get("words")

    if not isinstance(words, list):
        raise ValueError(
            "Continuous ASR evidence has invalid words"
        )

    return evidence


def _run_sat(
    words: list[dict],
) -> set[int]:
    sat_worker = worker("sat")

    with tempfile.TemporaryDirectory(
        prefix="voice-dataset-sat-"
    ) as temp_dir:
        root = Path(temp_dir)

        input_path = root / "input.json"
        output_path = root / "output.json"

        input_path.write_text(
            json.dumps(
                {
                    "words": words,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        run_worker(
            sat_worker,
            [
                "--input",
                input_path,
                "--output",
                output_path,
            ],
        )

        result = json.loads(
            output_path.read_text(
                encoding="utf-8"
            )
        )

    boundaries = result.get(
        "boundary_after_word_indices"
    )

    if not isinstance(boundaries, list):
        raise ValueError(
            "SaT worker returned invalid boundaries"
        )

    return {
        int(index)
        for index in boundaries
    }


def build_source_utterances(
    storage: DatasetStorage,
    source_id: str,
    *,
    asr_evidence_name: str = "qwen3",
) -> UtterancePipelineResult:
    evidence = _continuous_asr_evidence(
        storage,
        source_id,
        evidence_name=asr_evidence_name,
    )

    raw_words = evidence["words"]

    language = evidence.get("language")

    if not isinstance(language, str) or not language:
        raise ValueError(
            "Continuous ASR evidence has invalid language"
        )

    boundary_representation_name, _, _ = (
        resolve_source_representation_for_purpose(
            storage,
            source_id,
            "boundary_analysis",
        )
    )

    alignment = build_effective_word_alignment(
        storage,
        source_id,
        raw_words,
        representation_name=boundary_representation_name,
        language=language,
    )

    sat_boundaries = _run_sat(
        raw_words
    )

    alignment = recover_sat_boundary_alignments(
        storage,
        source_id,
        alignment,
        sat_boundaries,
        representation_name=boundary_representation_name,
        language=language,
    )

    region_evidence = collect_region_evidence(
        storage,
        source_id,
        list(alignment.words),
    )

    sat_boundaries = validate_sat_boundaries(
        list(alignment.words),
        region_evidence,
        sat_boundaries,
    )

    suppressed_word_indices = set(
        alignment.suppressed_word_indices
    )

    effective_word_indices = [
        index
        for index in range(
            len(alignment.words)
        )
        if index not in suppressed_word_indices
    ]

    effective_words = [
        dict(alignment.words[index])
        for index in effective_word_indices
    ]

    attributed = (
        attribute_speakers_to_words(
            storage,
            source_id,
            evidence_name=(
                asr_evidence_name
            ),
            words=effective_words,
            word_indices=effective_word_indices,
        )
    )

    contextual = (
        resolve_consistent_speaker_context(
            attributed
        )
    )

    resolved = (
        resolve_fragmented_speaker_words(
            storage,
            source_id,
            contextual,
            evidence_name=(
                asr_evidence_name
            ),
            geometry_words=effective_words,
            boundary_after_word_indices=(
                sat_boundaries
            ),
        )
    )


    candidates = build_utterance_candidates(
        resolved,
        boundary_after_word_indices=(
            sat_boundaries
        ),
    )

    return UtterancePipelineResult(
        words=tuple(resolved),
        candidates=tuple(candidates),
        sat_boundary_after_word_indices=tuple(
            sorted(sat_boundaries)
        ),
        alignment=alignment,
    )


@dataclass(frozen=True)
class PlannedUtteranceTurn:
    candidate: UtteranceCandidate
    status: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ApplyUtteranceTurnsResult:
    created: int
    skipped: int
    review: int


@dataclass(frozen=True)
class PrepareSpeakerEvidenceResult:
    representation: MaterializeTurnsResult
    ecapa: EmbeddingRunResult
    wespeaker: EmbeddingRunResult


def prepare_source_review_audio(
    storage: DatasetStorage,
    source_id: str,
) -> MaterializeTurnsResult:
    _, source_representation, source_path = (
        resolve_source_representation_for_purpose(
            storage,
            source_id,
            "review",
        )
    )

    source_kind = source_representation.get("kind")

    if not isinstance(source_kind, str) or not source_kind:
        raise ValueError(
            "Review source representation has "
            f"invalid kind: {source_id}"
        )

    return materialize_turns(
        storage=storage,
        source_id=source_id,
        source=source_path,
        representation_name="review",
        kind=source_kind,
        purposes=["review"],
    )


def prepare_source_speaker_evidence(
    storage: DatasetStorage,
    source_id: str,
) -> PrepareSpeakerEvidenceResult:
    _, source_representation, source_path = (
        resolve_source_representation_for_purpose(
            storage,
            source_id,
            "speaker_embedding",
        )
    )

    source_kind = source_representation.get("kind")

    if not isinstance(source_kind, str) or not source_kind:
        raise ValueError(
            "Source representation for "
            "'speaker_embedding' has invalid kind"
        )

    representation = materialize_turns(
        storage=storage,
        source_id=source_id,
        source=source_path,
        representation_name="speaker",
        kind=source_kind,
        purposes=["speaker_embedding"],
    )

    ecapa = embed_turns(
        storage,
        encoder="ecapa",
        representation="speaker",
        name="ecapa_speaker",
    )

    wespeaker = embed_turns(
        storage,
        encoder="wespeaker",
        representation="speaker",
        name="wespeaker_speaker",
    )

    return PrepareSpeakerEvidenceResult(
        representation=representation,
        ecapa=ecapa,
        wespeaker=wespeaker,
    )


def plan_source_utterance_turns(
    result: UtterancePipelineResult,
) -> tuple[PlannedUtteranceTurn, ...]:
    planned: list[PlannedUtteranceTurn] = []

    for candidate in result.candidates:
        reasons: list[str] = []

        if candidate.speaker is None:
            reasons.append(
                "unresolved_speaker"
            )

        if candidate.unresolved_word_indices:
            reasons.append(
                "unresolved_words"
            )

        if candidate.alignment_issue_word_indices:
            reasons.append(
                "invalid_alignment"
            )

        status = (
            "ready"
            if not reasons
            else "review"
        )

        planned.append(
            PlannedUtteranceTurn(
                candidate=candidate,
                status=status,
                reasons=tuple(reasons),
            )
        )

    return tuple(planned)


def apply_source_utterance_turns(
    storage: DatasetStorage,
    source_id: str,
    result: UtterancePipelineResult,
    *,
    language: str | None,
    asr_evidence_name: str = "qwen3",
    segmentation_evidence_name: str = "sat",
) -> ApplyUtteranceTurnsResult:
    from .utterance_reconciliation import (
        create_turn_from_utterance,
    )

    plan = plan_source_utterance_turns(
        result
    )

    planned_ranges = {
        (
            item.candidate.start_word_index,
            item.candidate.end_word_index,
        )
        for item in plan
    }

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

        word_range = metadata.get("word_range")

        if not isinstance(word_range, dict):
            continue

        start = word_range.get("start")
        end = word_range.get("end")

        if not isinstance(start, int):
            continue

        if not isinstance(end, int):
            continue

        existing_range = (start, end)

        if existing_range in planned_ranges:
            continue

        overlaps_plan = any(
            start <= planned_end
            and planned_start <= end
            for planned_start, planned_end
            in planned_ranges
        )

        if not overlaps_plan:
            continue

        raise ValueError(
            "Source has stale automatic turn "
            f"{turn['id']} with word range "
            f"{start}-{end}; current reconciliation "
            "produces a different overlapping turn "
            "layout. Refusing to mutate turns."
        )

    created = 0
    skipped = 0
    review = 0

    existing_ids = {
        turn["id"]
        for turn in storage.turns.load()
    }

    for item in plan:
        if item.status == "review":
            review += 1

        turn = create_turn_from_utterance(
            storage,
            source_id=source_id,
            candidate=item.candidate,
            words=list(result.words),
            language=language,
            pipeline_status=item.status,
            review_reasons=item.reasons,
            asr_evidence_name=(
                asr_evidence_name
            ),
            segmentation_evidence_name=(
                segmentation_evidence_name
            ),
        )

        if turn["id"] in existing_ids:
            skipped += 1
        else:
            created += 1
            existing_ids.add(turn["id"])

    mark_source_utterance_reconciliation_complete(
        storage,
        source_id,
        result,
        asr_evidence_name=asr_evidence_name,
    )

    return ApplyUtteranceTurnsResult(
        created=created,
        skipped=skipped,
        review=review,
    )


def source_utterance_reconciliation_is_curated(
    storage: DatasetStorage,
    source_id: str,
    *,
    asr_evidence_name: str = "qwen3",
) -> bool:
    source = storage.get_source(source_id)

    if source is None:
        return False

    metadata = source.get("metadata")

    if not isinstance(metadata, dict):
        return False

    reconciliation = metadata.get(
        "utterance_reconciliation"
    )

    if not isinstance(reconciliation, dict):
        return False

    if (
        reconciliation.get("asr_evidence")
        != asr_evidence_name
    ):
        return False

    return reconciliation.get("mode") == "curated"


def source_utterance_reconciliation_is_complete(
    storage: DatasetStorage,
    source_id: str,
    *,
    asr_evidence_name: str = "qwen3",
) -> bool:
    source = storage.get_source(source_id)

    if source is None:
        return False

    metadata = source.get("metadata")

    if not isinstance(metadata, dict):
        return False

    reconciliation = metadata.get(
        "utterance_reconciliation"
    )

    if not isinstance(reconciliation, dict):
        return False

    if (
        reconciliation.get("asr_evidence")
        != asr_evidence_name
    ):
        return False

    word_ranges = reconciliation.get("word_ranges")

    if not isinstance(word_ranges, list):
        return False

    expected_ranges: list[tuple[int, int]] = []

    for item in word_ranges:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], int)
            or not isinstance(item[1], int)
        ):
            return False

        expected_ranges.append((item[0], item[1]))

    actual_ranges: list[tuple[int, int]] = []

    for turn in storage.turns.load():
        if turn.get("source_id") != source_id:
            continue

        turn_metadata = turn.get("metadata")

        if not isinstance(turn_metadata, dict):
            continue

        creation = turn_metadata.get("creation")

        if not isinstance(creation, dict):
            continue

        if (
            creation.get("method")
            != "continuous_asr_utterance"
        ):
            continue

        word_range = turn_metadata.get("word_range")

        if not isinstance(word_range, dict):
            return False

        start = word_range.get("start")
        end = word_range.get("end")

        if not isinstance(start, int):
            return False

        if not isinstance(end, int):
            return False

        actual_ranges.append((start, end))

    return sorted(actual_ranges) == sorted(expected_ranges)


def require_source_utterance_reconciliation(
    storage: DatasetStorage,
    source_id: str,
    *,
    asr_evidence_name: str = "qwen3",
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

    reconciliation = metadata.get(
        "utterance_reconciliation"
    )

    if not isinstance(reconciliation, dict):
        raise ValueError(
            "Source has no completed utterance "
            f"reconciliation: {source_id}"
        )

    if (
        reconciliation.get("asr_evidence")
        != asr_evidence_name
    ):
        raise ValueError(
            "Source utterance reconciliation uses "
            "different ASR evidence: "
            f"{source_id}"
        )

    return reconciliation


def mark_source_utterance_reconciliation_complete(
    storage: DatasetStorage,
    source_id: str,
    result: UtterancePipelineResult,
    *,
    asr_evidence_name: str = "qwen3",
) -> None:
    word_ranges = [
        [
            candidate.start_word_index,
            candidate.end_word_index,
        ]
        for candidate in result.candidates
    ]

    def update(
        record: dict,
    ) -> dict:
        metadata = record.get("metadata", {})

        if not isinstance(metadata, dict):
            raise ValueError(
                f"Source has invalid metadata: "
                f"{source_id}"
            )

        metadata["utterance_reconciliation"] = {
            "mode": "automatic",
            "asr_evidence": asr_evidence_name,
            "word_ranges": word_ranges,
        }

        record["metadata"] = metadata

        return record

    storage.update_source(
        source_id,
        update,
    )


def mark_source_utterance_reconciliation_curated(
    storage: DatasetStorage,
    source_id: str,
    *,
    asr_evidence_name: str = "qwen3",
) -> None:
    require_source_utterance_reconciliation(
        storage,
        source_id,
        asr_evidence_name=asr_evidence_name,
    )

    def update(record: dict) -> dict:
        metadata = record["metadata"]
        reconciliation = metadata[
            "utterance_reconciliation"
        ]

        reconciliation["mode"] = "curated"

        return record

    storage.update_source(
        source_id,
        update,
    )

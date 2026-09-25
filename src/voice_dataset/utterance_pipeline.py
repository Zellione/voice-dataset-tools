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

    alignment = build_effective_word_alignment(
        storage,
        source_id,
        raw_words,
        representation_name="center",
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
        representation_name="center",
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


def _source_representation_for_purpose(
    storage: DatasetStorage,
    source_id: str,
    *,
    purpose: str,
) -> tuple[Path, str]:
    source = storage.get_source(source_id)

    if source is None:
        raise KeyError(
            f"Source does not exist: {source_id}"
        )

    representations = source.get(
        "representations"
    )

    if not isinstance(representations, dict):
        raise ValueError(
            f"Source has invalid representations: "
            f"{source_id}"
        )

    matches: list[tuple[Path, str]] = []

    for representation in representations.values():
        if not isinstance(representation, dict):
            continue

        purposes = representation.get("purposes")

        if (
            not isinstance(purposes, list)
            or purpose not in purposes
        ):
            continue

        path = representation.get("path")
        kind = representation.get("kind")

        if (
            not isinstance(path, str)
            or not path
            or not isinstance(kind, str)
            or not kind
        ):
            raise ValueError(
                "Source representation for "
                f"{purpose!r} is invalid"
            )

        matches.append(
            (
                storage.root / path,
                kind,
            )
        )

    if not matches:
        raise ValueError(
            "No source representation supports "
            f"purpose {purpose!r}: {source_id}"
        )

    if len(matches) != 1:
        raise ValueError(
            "Multiple source representations support "
            f"purpose {purpose!r}: {source_id}"
        )

    return matches[0]


def prepare_source_speaker_evidence(
    storage: DatasetStorage,
    source_id: str,
) -> PrepareSpeakerEvidenceResult:
    source_path, source_kind = (
        _source_representation_for_purpose(
            storage,
            source_id,
            purpose="speaker_embedding",
        )
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

    return ApplyUtteranceTurnsResult(
        created=created,
        skipped=skipped,
        review=review,
    )

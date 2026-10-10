from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .continuous_asr import (
    ForcedAlignmentRequest,
    align_text_qwen3,
    align_texts_qwen3,
)
from .media import extract_audio_region
from .sources import resolve_source_representation
from .storage import DatasetStorage

_TOKEN_RE = re.compile(
    r"\w+(?:['’]\w+)*",
    re.UNICODE,
)

MAX_WORD_DURATION_SECONDS = 5.0
MAX_INTER_WORD_GAP_SECONDS = 5.0


@dataclass(frozen=True)
class AlignmentIssue:
    word_indices: tuple[int, ...]
    reasons: tuple[str, ...]
    boundary_after_word_index: int | None = None


@dataclass(frozen=True)
class AlignmentRegionEvidence:
    region_id: str
    start: float
    end: float
    speaker: str | None
    whisper_text: str
    whisper_tokens: tuple[str, ...]
    text_matches: tuple[AlignmentTextMatch, ...]

    @property
    def unique_text_match(
        self,
    ) -> AlignmentTextMatch | None:
        if len(self.text_matches) != 1:
            return None

        return self.text_matches[0]


@dataclass(frozen=True)
class AlignmentComparison:
    word_indices: tuple[int, ...]
    original_start: float
    original_end: float
    recovered_start: float
    recovered_end: float
    start_delta: float
    end_delta: float
    original_duration: float
    recovered_duration: float


@dataclass(frozen=True)
class EffectiveWordAlignment:
    words: tuple[dict[str, Any], ...]
    recoveries: tuple[LocalAlignmentRecovery, ...]
    suppressed_word_indices: tuple[int, ...] = ()
    unresolved_alignment_word_indices: tuple[int, ...] = ()


def _plan_boundary_recovery_anchors(
    issues: list[AlignmentIssue],
    region_evidence: list[AlignmentRegionEvidence],
    *,
    claimed_word_indices: set[int],
) -> list[AlignmentRegionEvidence]:
    candidates: dict[
        tuple[int, ...],
        dict[str, AlignmentRegionEvidence],
    ] = {}

    for issue in issues:
        boundary = issue.boundary_after_word_index

        if boundary is None:
            continue

        boundary_evidence = (
            collect_boundary_text_evidence(
                boundary,
                region_evidence,
            )
        )

        for anchor in (
            *boundary_evidence.left,
            *boundary_evidence.right,
        ):
            match = anchor.unique_text_match

            if match is None:
                continue

            word_indices = match.word_indices

            if any(
                index in claimed_word_indices
                for index in word_indices
            ):
                continue

            by_region = candidates.setdefault(
                word_indices,
                {},
            )

            # The same region can appear from both sides
            # of adjacent gap issues. Keep it only once.
            by_region[anchor.region_id] = anchor

    planned: list[
        AlignmentRegionEvidence
    ] = []

    for word_indices, by_region in candidates.items():
        # Multiple distinct regions claiming exactly the
        # same word range are ambiguous evidence. Do not
        # guess which occurrence is correct.
        if len(by_region) != 1:
            continue

        planned.append(
            next(iter(by_region.values()))
        )

    planned.sort(
        key=lambda evidence: (
            evidence.unique_text_match.start_word_index,
            evidence.unique_text_match.end_word_index,
            evidence.region_id,
        )
    )

    return planned


def _candidate_from_region_evidence(
    evidence: AlignmentRegionEvidence,
) -> AlignmentRecoveryCandidate:
    match = evidence.unique_text_match

    if match is None:
        raise ValueError(
            "Region alignment recovery requires "
            "a unique text match"
        )

    return AlignmentRecoveryCandidate(
        word_indices=match.word_indices,
        start_word_index=match.start_word_index,
        end_word_index=match.end_word_index,
        region_id=evidence.region_id,
        region_start=evidence.start,
        region_end=evidence.end,
        whisper_text=evidence.whisper_text,
        speaker=evidence.speaker,
        issue_word_indices=(),
        issue_reasons=(
            "boundary_alignment_probe",
        ),
    )


def _stranded_to_alignment_candidate(
    candidate: StrandedAlignmentCandidate,
) -> AlignmentRecoveryCandidate:
    return AlignmentRecoveryCandidate(
        word_indices=candidate.word_indices,
        start_word_index=(
            candidate.word_indices[0]
        ),
        end_word_index=(
            candidate.word_indices[-1]
        ),
        region_id=candidate.region_id,
        region_start=candidate.region_start,
        region_end=candidate.region_end,
        whisper_text="",
        speaker=candidate.speaker,
        issue_word_indices=(
            candidate.word_indices
        ),
        issue_reasons=(
            "stranded_alignment",
        ),
    )


def recover_alignment_candidates_batch(
    storage: DatasetStorage,
    source_id: str,
    candidates: list[AlignmentRecoveryCandidate],
    words: list[dict[str, Any]],
    *,
    representation_name: str,
    language: str,
    max_inference_batch_size: int = 8,
) -> list[LocalAlignmentRecovery]:
    if not candidates:
        return []

    # Preserve the existing single-item path. Besides being
    # simple, this keeps individual recovery behavior and
    # tests unchanged.
    if len(candidates) == 1:
        return [
            recover_candidate_alignment(
                storage,
                source_id,
                candidates[0],
                words,
                representation_name=(
                    representation_name
                ),
                language=language,
            )
        ]

    _, source_path = resolve_source_representation(
        storage,
        source_id,
        representation_name,
    )

    with TemporaryDirectory(
        prefix="voice-dataset-alignment-batch-"
    ) as temporary_dir:
        temporary_root = Path(
            temporary_dir
        )

        requests: list[
            ForcedAlignmentRequest
        ] = []

        texts: list[str] = []

        for index, candidate in enumerate(
            candidates
        ):
            text = " ".join(
                str(words[word_index]["text"])
                for word_index
                in candidate.word_indices
            )

            if not text.strip():
                raise ValueError(
                    "Alignment recovery candidate "
                    f"{index} has no text"
                )

            clip_path = (
                temporary_root
                / f"candidate_{index:04d}.wav"
            )

            extract_audio_region(
                source=source_path,
                destination=clip_path,
                start=candidate.region_start,
                end=candidate.region_end,
            )

            texts.append(text)

            requests.append(
                ForcedAlignmentRequest(
                    audio_path=clip_path,
                    text=text,
                    language=language,
                )
            )

        alignments = align_texts_qwen3(
            requests,
            max_inference_batch_size=(
                max_inference_batch_size
            ),
        )

        if len(alignments) != len(
            candidates
        ):
            raise RuntimeError(
                "Forced alignment batch result "
                "count mismatch"
            )

        recoveries: list[
            LocalAlignmentRecovery
        ] = []

        for (
            candidate,
            text,
            alignment,
        ) in zip(
            candidates,
            texts,
            alignments,
            strict=True,
        ):
            recovered_words = tuple(
                {
                    "text": word["text"],
                    "start": max(
                        candidate.region_start,
                        min(
                            float(word["start"])
                            + candidate.region_start,
                            candidate.region_end,
                        ),
                    ),
                    "end": max(
                        candidate.region_start,
                        min(
                            float(word["end"])
                            + candidate.region_start,
                            candidate.region_end,
                        ),
                    ),
                }
                for word in alignment.words
            )

            recoveries.append(
                LocalAlignmentRecovery(
                    word_indices=(
                        candidate.word_indices
                    ),
                    region_id=(
                        candidate.region_id
                    ),
                    region_start=(
                        candidate.region_start
                    ),
                    region_end=(
                        candidate.region_end
                    ),
                    text=text,
                    words=recovered_words,
                )
            )

    return recoveries


ZERO_DURATION_REGION_MIN_TOKEN_SIMILARITY = 0.65
ZERO_DURATION_REGION_MIN_CHARACTER_SIMILARITY = 0.82


def _alignment_text_similarity(
    left: str,
    right: str,
) -> tuple[float, float]:
    left_tokens = _lexical_tokens(
        left
    )
    right_tokens = _lexical_tokens(
        right
    )

    if not left_tokens or not right_tokens:
        return 0.0, 0.0

    token_similarity = SequenceMatcher(
        None,
        left_tokens,
        right_tokens,
        autojunk=False,
    ).ratio()

    left_characters = "".join(
        left_tokens
    )
    right_characters = "".join(
        right_tokens
    )

    character_similarity = SequenceMatcher(
        None,
        left_characters,
        right_characters,
        autojunk=False,
    ).ratio()

    return (
        token_similarity,
        character_similarity,
    )


def _alignment_texts_are_similar(
    left: str,
    right: str,
) -> bool:
    (
        token_similarity,
        character_similarity,
    ) = _alignment_text_similarity(
        left,
        right,
    )

    return (
        character_similarity >= 0.95
        or (
            token_similarity
            >= ZERO_DURATION_REGION_MIN_TOKEN_SIMILARITY
            and character_similarity
            >= ZERO_DURATION_REGION_MIN_CHARACTER_SIMILARITY
        )
    )


def build_zero_duration_region_recovery_candidates(
    words: list[dict[str, Any]],
    issues: list[AlignmentIssue],
    region_evidence: list[AlignmentRegionEvidence],
    *,
    claimed_word_indices: set[int],
) -> list[AlignmentRecoveryCandidate]:
    candidates: list[
        AlignmentRecoveryCandidate
    ] = []

    seen: set[
        tuple[int, int, str]
    ] = set()

    for issue in issues:
        if (
            "zero_word_duration"
            not in issue.reasons
        ):
            continue

        if not issue.word_indices:
            continue

        run_indices = set(
            issue.word_indices
        )

        # Exact text-based recovery always wins. For a
        # partially claimed run, do not create an
        # overlapping geometry-based recovery.
        if (
            run_indices
            & claimed_word_indices
        ):
            continue

        timestamps = [
            float(words[index]["start"])
            for index in issue.word_indices
        ]

        containing_regions = [
            evidence
            for evidence in region_evidence
            if all(
                evidence.start - 1e-6
                <= timestamp
                <= evidence.end + 1e-6
                for timestamp in timestamps
            )
        ]

        if len(containing_regions) != 1:
            continue

        evidence = containing_regions[0]

        start_index = (
            issue.word_indices[0]
        )
        end_index = (
            issue.word_indices[-1]
        )

        run_text = " ".join(
            str(words[index]["text"])
            for index in issue.word_indices
        )

        run_matches_region = (
            _alignment_texts_are_similar(
                run_text,
                evidence.whisper_text,
            )
        )

        def inside_region(
            index: int,
        ) -> bool:
            word = words[index]

            word_start = float(
                word["start"]
            )
            word_end = float(
                word["end"]
            )

            return (
                word_start
                >= evidence.start - 1e-6
                and word_end
                <= evidence.end + 1e-6
            )

        if not run_matches_region:
            # The zero-duration fragment may be one
            # broken word inside an otherwise correctly
            # recognized local phrase. Expand only in
            # that case and validate the complete phrase.
            while start_index > 0:
                previous = start_index - 1

                if (
                    previous
                    in claimed_word_indices
                ):
                    break

                if not inside_region(
                    previous
                ):
                    break

                start_index = previous

            while (
                end_index + 1
                < len(words)
            ):
                following = end_index + 1

                if (
                    following
                    in claimed_word_indices
                ):
                    break

                if not inside_region(
                    following
                ):
                    break

                end_index = following

        word_indices = tuple(
            range(
                start_index,
                end_index + 1,
            )
        )

        if any(
            index in claimed_word_indices
            for index in word_indices
        ):
            continue

        candidate_text = " ".join(
            str(words[index]["text"])
            for index in word_indices
        )

        if not _alignment_texts_are_similar(
            candidate_text,
            evidence.whisper_text,
        ):
            continue

        key = (
            start_index,
            end_index,
            evidence.region_id,
        )

        if key in seen:
            continue

        seen.add(key)

        candidates.append(
            AlignmentRecoveryCandidate(
                word_indices=word_indices,
                start_word_index=(
                    start_index
                ),
                end_word_index=(
                    end_index
                ),
                region_id=(
                    evidence.region_id
                ),
                region_start=(
                    evidence.start
                ),
                region_end=(
                    evidence.end
                ),
                whisper_text=(
                    evidence.whisper_text
                ),
                speaker=evidence.speaker,
                issue_word_indices=(
                    issue.word_indices
                ),
                issue_reasons=(
                    "zero_word_duration_region_fallback",
                ),
            )
        )

        claimed_word_indices.update(
            word_indices
        )

    return candidates


LEXICAL_GEOMETRY_MAX_EDGE_DELTA_SECONDS = 1.0
LEXICAL_GEOMETRY_MIN_TOKENS = 2


def _alignment_region_geometry_is_good(
    words: list[dict[str, Any]],
    evidence: AlignmentRegionEvidence,
) -> bool:
    match = evidence.unique_text_match

    if match is None:
        return False

    qwen_start = float(
        words[match.start_word_index]["start"]
    )
    qwen_end = float(
        words[match.end_word_index]["end"]
    )

    overlap = max(
        0.0,
        min(qwen_end, evidence.end)
        - max(qwen_start, evidence.start),
    )

    return (
        overlap > 0.0
        and abs(qwen_start - evidence.start)
        <= LEXICAL_GEOMETRY_MAX_EDGE_DELTA_SECONDS
        and abs(qwen_end - evidence.end)
        <= LEXICAL_GEOMETRY_MAX_EDGE_DELTA_SECONDS
    )


def build_lexical_geometry_recovery_candidates(
    words: list[dict[str, Any]],
    region_evidence: list[AlignmentRegionEvidence],
    *,
    claimed_word_indices: set[int],
) -> list[AlignmentRecoveryCandidate]:
    ordered = [
        evidence
        for evidence in sorted(
            region_evidence,
            key=lambda item: (
                item.start,
                item.end,
                item.region_id,
            ),
        )
        if evidence.unique_text_match is not None
    ]

    candidates: list[
        AlignmentRecoveryCandidate
    ] = []

    for position, evidence in enumerate(
        ordered
    ):
        match = evidence.unique_text_match

        if match is None:
            continue

        if _alignment_region_geometry_is_good(
            words,
            evidence,
        ):
            continue

        if (
            len(match.tokens)
            < LEXICAL_GEOMETRY_MIN_TOKENS
        ):
            continue

        if any(
            index in claimed_word_indices
            for index in match.word_indices
        ):
            continue

        previous_good = next(
            (
                previous
                for previous in reversed(
                    ordered[:position]
                )
                if _alignment_region_geometry_is_good(
                    words,
                    previous,
                )
            ),
            None,
        )

        next_good = next(
            (
                following
                for following in ordered[
                    position + 1:
                ]
                if _alignment_region_geometry_is_good(
                    words,
                    following,
                )
            ),
            None,
        )

        # Require anchors on both sides. This is what
        # distinguishes a locally displaced phrase from
        # an unrelated repeated phrase elsewhere in the
        # source.
        if (
            previous_good is None
            or next_good is None
        ):
            continue

        previous_match = (
            previous_good.unique_text_match
        )
        next_match = (
            next_good.unique_text_match
        )

        if (
            previous_match is None
            or next_match is None
        ):
            continue

        if not (
            previous_match.end_word_index
            < match.start_word_index
            and match.end_word_index
            < next_match.start_word_index
        ):
            continue

        candidates.append(
            AlignmentRecoveryCandidate(
                word_indices=(
                    match.word_indices
                ),
                start_word_index=(
                    match.start_word_index
                ),
                end_word_index=(
                    match.end_word_index
                ),
                region_id=evidence.region_id,
                region_start=evidence.start,
                region_end=evidence.end,
                whisper_text=(
                    evidence.whisper_text
                ),
                speaker=evidence.speaker,
                issue_word_indices=(
                    match.word_indices
                ),
                issue_reasons=(
                    "lexical_geometry_mismatch",
                ),
            )
        )

        claimed_word_indices.update(
            match.word_indices
        )

    return candidates


def trim_alignment_recovery_to_stable_boundaries(
    words: list[dict[str, Any]],
    recovery: LocalAlignmentRecovery,
    *,
    recovered_word_indices: set[int],
    unstable_word_indices: set[int],
) -> LocalAlignmentRecovery | None:
    if not recovery.word_indices or not recovery.words:
        return None

    trimmed_words = [
        dict(word)
        for word in recovery.words
    ]

    first_index = recovery.word_indices[0]
    previous_index = first_index - 1

    while previous_index >= 0:
        if (
            previous_index in recovered_word_indices
            or previous_index in unstable_word_indices
        ):
            previous_index -= 1
            continue

        break

    if previous_index >= 0:
        previous_end = float(
            words[previous_index]["end"]
        )
        first_start = float(
            trimmed_words[0]["start"]
        )
        first_end = float(
            trimmed_words[0]["end"]
        )

        if first_start < previous_end - 1e-6:
            if previous_end >= first_end - 1e-6:
                return None

            trimmed_words[0]["start"] = previous_end

    last_index = recovery.word_indices[-1]
    next_index = last_index + 1

    while next_index < len(words):
        if (
            next_index in recovered_word_indices
            or next_index in unstable_word_indices
        ):
            next_index += 1
            continue

        break

    if next_index < len(words):
        next_start = float(
            words[next_index]["start"]
        )
        last_start = float(
            trimmed_words[-1]["start"]
        )
        last_end = float(
            trimmed_words[-1]["end"]
        )

        if last_end > next_start + 1e-6:
            if next_start <= last_start + 1e-6:
                return None

            trimmed_words[-1]["end"] = next_start

    return LocalAlignmentRecovery(
        word_indices=recovery.word_indices,
        region_id=recovery.region_id,
        region_start=recovery.region_start,
        region_end=recovery.region_end,
        text=recovery.text,
        words=tuple(trimmed_words),
    )



def lexical_geometry_recovery_is_usable(
    recovery: LocalAlignmentRecovery,
) -> bool:
    if not recovery.words:
        return False

    if (
        len(recovery.word_indices)
        != len(recovery.words)
    ):
        return False

    return all(
        float(word["end"])
        - float(word["start"])
        > 1e-6
        for word in recovery.words
    )


def build_effective_word_alignment(
    storage: DatasetStorage,
    source_id: str,
    words: list[dict[str, Any]],
    *,
    representation_name: str,
    language: str,
) -> EffectiveWordAlignment:
    issues = detect_alignment_issues(words)

    region_evidence = collect_region_evidence(
        storage,
        source_id,
        words,
    )

    recoveries: list[
        LocalAlignmentRecovery
    ] = []

    boundary_recoveries: list[
        LocalAlignmentRecovery
    ] = []

    word_candidates = build_recovery_candidates(
        issues,
        region_evidence,
    )

    claimed_word_indices: set[int] = set()

    planned_word_candidates: list[
        AlignmentRecoveryCandidate
    ] = []

    for candidate in word_candidates:
        candidate_indices = set(
            candidate.word_indices
        )

        if (
            candidate_indices
            & claimed_word_indices
        ):
            continue

        planned_word_candidates.append(
            candidate
        )

        claimed_word_indices.update(
            candidate.word_indices
        )

    zero_region_candidates = (
        build_zero_duration_region_recovery_candidates(
            words,
            issues,
            region_evidence,
            claimed_word_indices=(
                claimed_word_indices
            ),
        )
    )

    lexical_geometry_candidates = (
        build_lexical_geometry_recovery_candidates(
            words,
            region_evidence,
            claimed_word_indices=(
                claimed_word_indices
            ),
        )
    )

    non_boundary_candidates = [
        *planned_word_candidates,
        *zero_region_candidates,
        *lexical_geometry_candidates,
    ]

    boundary_anchors = (
        _plan_boundary_recovery_anchors(
            issues,
            region_evidence,
            claimed_word_indices=(
                claimed_word_indices
            ),
        )
    )

    boundary_candidates = [
        _candidate_from_region_evidence(
            anchor
        )
        for anchor in boundary_anchors
    ]

    initial_candidates = [
        *non_boundary_candidates,
        *boundary_candidates,
    ]

    initial_recoveries = (
        recover_alignment_candidates_batch(
            storage,
            source_id,
            initial_candidates,
            words,
            representation_name=(
                representation_name
            ),
            language=language,
        )
    )

    word_recovery_count = len(
        non_boundary_candidates
    )

    raw_word_recoveries = initial_recoveries[
        :word_recovery_count
    ]

    lexical_recovery_start = (
        len(planned_word_candidates)
        + len(zero_region_candidates)
    )

    word_recoveries = []
    unresolved_alignment_word_indices: set[int] = set()

    unstable_word_indices = {
        index
        for issue in issues
        if (
            "zero_word_duration"
            in issue.reasons
            or "excessive_word_duration"
            in issue.reasons
        )
        for index in issue.word_indices
    }

    recovered_word_indices = {
        index
        for recovery in raw_word_recoveries
        for index in recovery.word_indices
    }

    for position, recovery in enumerate(
        raw_word_recoveries
    ):
        candidate = non_boundary_candidates[
            position
        ]

        is_lexical_geometry_recovery = (
            position >= lexical_recovery_start
        )

        if (
            is_lexical_geometry_recovery
            and not lexical_geometry_recovery_is_usable(
                recovery
            )
        ):
            unresolved_alignment_word_indices.update(
                candidate.word_indices
            )

            continue

        if not alignment_recovery_is_valid(
            words,
            recovery,
        ):
            unresolved_alignment_word_indices.update(
                candidate.word_indices
            )

            continue

        trimmed_recovery = (
            trim_alignment_recovery_to_stable_boundaries(
                words,
                recovery,
                recovered_word_indices=recovered_word_indices,
                unstable_word_indices=unstable_word_indices,
            )
        )

        if trimmed_recovery is None:
            unresolved_alignment_word_indices.update(
                candidate.word_indices
            )
            continue

        word_recoveries.append(
            trimmed_recovery
        )

    candidate_boundary_recoveries = (
        initial_recoveries[
            word_recovery_count:
        ]
    )

    recoveries.extend(
        word_recoveries
    )

    for recovery in candidate_boundary_recoveries:
        comparison = (
            compare_alignment_recovery(
                words,
                recovery,
            )
        )

        if alignment_spans_overlap(
            comparison
        ):
            continue

        recoveries.append(recovery)
        boundary_recoveries.append(
            recovery
        )

    stranded_anchor_recoveries = [
        *word_recoveries,
        *boundary_recoveries,
    ]

    stranded_candidates = (
        find_stranded_alignment_candidates(
            words,
            region_evidence,
            stranded_anchor_recoveries,
        )
    )

    stranded_alignment_candidates = [
        _stranded_to_alignment_candidate(
            candidate
        )
        for candidate in stranded_candidates
    ]

    stranded_recoveries = (
        recover_alignment_candidates_batch(
            storage,
            source_id,
            stranded_alignment_candidates,
            words,
            representation_name=(
                representation_name
            ),
            language=language,
        )
    )

    for recovery in stranded_recoveries:
        if alignment_recovery_is_valid(
            words,
            recovery,
        ):
            recoveries.append(recovery)

    effective_words = apply_alignment_recoveries(
        words,
        recoveries,
    )

    effective_region_evidence = collect_region_evidence(
        storage,
        source_id,
        effective_words,
    )

    post_recovery_conflicts = (
        find_post_recovery_word_conflicts(
            effective_words,
            recoveries,
        )
    )

    conflict_rescue_candidates = (
        build_post_recovery_conflict_rescue_candidates(
            effective_words,
            post_recovery_conflicts,
            effective_region_evidence,
            recoveries,
        )
    )

    conflict_rescue_recoveries = (
        recover_alignment_candidates_batch(
            storage,
            source_id,
            conflict_rescue_candidates,
            effective_words,
            representation_name=(
                representation_name
            ),
            language=language,
        )
    )

    usable_conflict_rescues = [
        recovery
        for recovery in conflict_rescue_recoveries
        if (
            lexical_geometry_recovery_is_usable(
                recovery
            )
            and alignment_recovery_is_valid(
                effective_words,
                recovery,
            )
        )
    ]

    if usable_conflict_rescues:
        recoveries.extend(
            usable_conflict_rescues
        )

        effective_words = apply_alignment_recoveries(
            words,
            recoveries,
        )

        effective_region_evidence = (
            collect_region_evidence(
                storage,
                source_id,
                effective_words,
            )
        )

        post_recovery_conflicts = (
            find_post_recovery_word_conflicts(
                effective_words,
                recoveries,
            )
        )

    post_recovery_suppressed = set(
        find_unclaimed_post_recovery_words(
            effective_words,
            post_recovery_conflicts,
            effective_region_evidence,
        )
    )

    invalid_duration_indices = {
        index
        for issue in issues
        if (
            "zero_word_duration"
            in issue.reasons
            or "excessive_word_duration"
            in issue.reasons
        )
        for index in issue.word_indices
    }

    recovered_valid_duration_indices = {
        index
        for recovery in recoveries
        for index, recovered_word in zip(
            recovery.word_indices,
            recovery.words,
            strict=True,
        )
        if (
            1e-6
            < (
                float(recovered_word["end"])
                - float(recovered_word["start"])
            )
            <= MAX_WORD_DURATION_SECONDS
        )
    }

    suppressed_word_indices = tuple(
        sorted(
            (
                invalid_duration_indices
                - recovered_valid_duration_indices
            )
            | post_recovery_suppressed
        )
    )

    return EffectiveWordAlignment(
        words=tuple(effective_words),
        recoveries=tuple(recoveries),
        suppressed_word_indices=(
            suppressed_word_indices
        ),
        unresolved_alignment_word_indices=tuple(
            sorted(
                unresolved_alignment_word_indices
            )
        ),
    )


def recover_sat_boundary_alignments(
    storage: DatasetStorage,
    source_id: str,
    alignment: EffectiveWordAlignment,
    boundary_after_word_indices: set[int],
    *,
    representation_name: str,
    language: str,
) -> EffectiveWordAlignment:
    words = [
        dict(word)
        for word in alignment.words
    ]

    region_evidence = collect_region_evidence(
        storage,
        source_id,
        words,
    )

    candidates = (
        build_sat_boundary_recovery_candidates(
            words,
            region_evidence,
            boundary_after_word_indices,
        )
    )

    recoveries: list[
        LocalAlignmentRecovery
    ] = []

    planned_candidates: list[
        AlignmentRecoveryCandidate
    ] = []

    claimed_word_indices: set[int] = set()

    for candidate in candidates:
        candidate_indices = set(
            candidate.word_indices
        )

        if (
            candidate_indices
            & claimed_word_indices
        ):
            continue

        planned_candidates.append(
            candidate
        )

        claimed_word_indices.update(
            candidate.word_indices
        )

    candidate_recoveries = (
        recover_alignment_candidates_batch(
            storage,
            source_id,
            planned_candidates,
            words,
            representation_name=(
                representation_name
            ),
            language=language,
        )
    )

    for recovery in candidate_recoveries:
        if (
            alignment_recovery_is_valid(words, recovery)
            and recovery.words
            and float(recovery.words[-1]["end"])
            == recovery.region_end
        ):
            recoveries.append(recovery)

    if not recoveries:
        return alignment

    effective_words = apply_alignment_recoveries(
        words,
        recoveries,
    )

    conflicts = find_post_recovery_word_conflicts(
        effective_words,
        recoveries,
    )

    newly_suppressed = (
        find_unclaimed_post_recovery_words(
            effective_words,
            conflicts,
            region_evidence,
        )
    )

    suppressed_word_indices = tuple(
        dict.fromkeys(
            (
                *alignment.suppressed_word_indices,
                *newly_suppressed,
            )
        )
    )

    return EffectiveWordAlignment(
        words=tuple(effective_words),
        recoveries=(
            *alignment.recoveries,
            *recoveries,
        ),
        suppressed_word_indices=(
            suppressed_word_indices
        ),
        unresolved_alignment_word_indices=(
            alignment.unresolved_alignment_word_indices
        ),
    )


def compare_alignment_recovery(
    words: list[dict[str, Any]],
    recovery: LocalAlignmentRecovery,
) -> AlignmentComparison:
    if not recovery.word_indices:
        raise ValueError(
            "Alignment recovery has no words"
        )

    if len(recovery.word_indices) != len(
        recovery.words
    ):
        raise ValueError(
            "Alignment recovery word count mismatch"
        )

    first_index = recovery.word_indices[0]
    last_index = recovery.word_indices[-1]

    original_start = float(
        words[first_index]["start"]
    )
    original_end = float(
        words[last_index]["end"]
    )

    recovered_start = float(
        recovery.words[0]["start"]
    )
    recovered_end = float(
        recovery.words[-1]["end"]
    )

    return AlignmentComparison(
        word_indices=recovery.word_indices,
        original_start=original_start,
        original_end=original_end,
        recovered_start=recovered_start,
        recovered_end=recovered_end,
        start_delta=(
            recovered_start - original_start
        ),
        end_delta=(
            recovered_end - original_end
        ),
        original_duration=(
            original_end - original_start
        ),
        recovered_duration=(
            recovered_end - recovered_start
        ),
    )


@dataclass(frozen=True)
class BoundaryTextEvidence:
    boundary_after_word_index: int
    left: tuple[AlignmentRegionEvidence, ...]
    right: tuple[AlignmentRegionEvidence, ...]


def collect_boundary_text_evidence(
    boundary_after_word_index: int,
    region_evidence: list[AlignmentRegionEvidence],
) -> BoundaryTextEvidence:
    left: list[AlignmentRegionEvidence] = []
    right: list[AlignmentRegionEvidence] = []

    right_word_index = (
        boundary_after_word_index + 1
    )

    for evidence in region_evidence:
        match = evidence.unique_text_match

        if match is None:
            continue

        if (
            match.end_word_index
            == boundary_after_word_index
        ):
            left.append(evidence)

        if (
            match.start_word_index
            == right_word_index
        ):
            right.append(evidence)

    return BoundaryTextEvidence(
        boundary_after_word_index=(
            boundary_after_word_index
        ),
        left=tuple(left),
        right=tuple(right),
    )


@dataclass(frozen=True)
class AlignmentTextMatch:
    start_word_index: int
    end_word_index: int
    word_indices: tuple[int, ...]
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class AlignmentRecoveryCandidate:
    word_indices: tuple[int, ...]
    start_word_index: int
    end_word_index: int

    region_id: str
    region_start: float
    region_end: float

    whisper_text: str
    speaker: str | None

    issue_word_indices: tuple[int, ...]
    issue_reasons: tuple[str, ...]


@dataclass(frozen=True)
class StrandedAlignmentCandidate:
    word_indices: tuple[int, ...]
    region_id: str
    region_start: float
    region_end: float
    speaker: str | None


def find_stranded_alignment_candidates(
    words: list[dict[str, Any]],
    region_evidence: list[AlignmentRegionEvidence],
    recoveries: list[LocalAlignmentRecovery],
) -> list[StrandedAlignmentCandidate]:
    candidates: list[
        StrandedAlignmentCandidate
    ] = []

    recovered_by_region = {
        recovery.region_id: recovery
        for recovery in recoveries
    }

    matched_word_indices = {
        word_index
        for evidence in region_evidence
        if evidence.unique_text_match is not None
        for word_index in (
            evidence.unique_text_match.word_indices
        )
    }

    for recovery in recoveries:
        recovered_start_index = min(
            recovery.word_indices
        )

        earlier_matches = [
            (
                evidence,
                evidence.unique_text_match,
            )
            for evidence in region_evidence
            if evidence.unique_text_match is not None
            and (
                evidence.unique_text_match.end_word_index
                < recovered_start_index
            )
            and (
                float(
                    words[
                        evidence.unique_text_match.start_word_index
                    ]["start"]
                )
                < evidence.end
            )
            and (
                float(
                    words[
                        evidence.unique_text_match.end_word_index
                    ]["end"]
                )
                > evidence.start
            )
        ]

        if not earlier_matches:
            continue

        _, previous_match = max(
            earlier_matches,
            key=lambda item: (
                item[1].end_word_index
            ),
        )

        between_indices = tuple(
            range(
                previous_match.end_word_index + 1,
                recovered_start_index,
            )
        )
        if not between_indices:
            continue

        if any(
            word_index in matched_word_indices
            for word_index in between_indices
        ):
            continue

        stranded_indices = between_indices

        previous_word_end = float(
            words[
                previous_match.end_word_index
            ]["end"]
        )

        unclaimed_regions = [
            evidence
            for evidence in region_evidence
            if (
                evidence.region_id
                not in recovered_by_region
            )
            and evidence.unique_text_match is None
            and evidence.start > previous_word_end
            and evidence.end < recovery.region_start
        ]

        if len(unclaimed_regions) != 1:
            continue

        evidence = unclaimed_regions[0]

        candidates.append(
            StrandedAlignmentCandidate(
                word_indices=stranded_indices,
                region_id=evidence.region_id,
                region_start=evidence.start,
                region_end=evidence.end,
                speaker=evidence.speaker,
            )
        )

    return candidates


@dataclass(frozen=True)
class LocalAlignmentRecovery:
    word_indices: tuple[int, ...]
    region_id: str
    region_start: float
    region_end: float
    text: str
    words: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PostRecoveryWordConflict:
    recovered_word_index: int
    conflicting_word_index: int
    recovery_region_id: str
    reason: str


def find_post_recovery_word_conflicts(
    words: list[dict[str, Any]],
    recoveries: list[LocalAlignmentRecovery],
) -> list[PostRecoveryWordConflict]:
    conflicts: list[PostRecoveryWordConflict] = []

    recovered_indices = {
        word_index
        for recovery in recoveries
        for word_index in recovery.word_indices
    }

    for recovery in recoveries:
        if not recovery.word_indices:
            continue

        recovered_word_index = max(
            recovery.word_indices
        )
        conflicting_word_index = (
            recovered_word_index + 1
        )

        if conflicting_word_index >= len(words):
            continue

        if conflicting_word_index in recovered_indices:
            continue

        recovered_word = words[
            recovered_word_index
        ]
        conflicting_word = words[
            conflicting_word_index
        ]

        if (
            float(conflicting_word["start"])
            >= float(recovered_word["end"])
        ):
            continue

        conflicts.append(
            PostRecoveryWordConflict(
                recovered_word_index=(
                    recovered_word_index
                ),
                conflicting_word_index=(
                    conflicting_word_index
                ),
                recovery_region_id=(
                    recovery.region_id
                ),
                reason=(
                    "overlaps_recovered_word"
                ),
            )
        )

    return conflicts


def build_post_recovery_conflict_rescue_candidates(
    words: list[dict[str, Any]],
    conflicts: list[PostRecoveryWordConflict],
    region_evidence: list[AlignmentRegionEvidence],
    recoveries: list[LocalAlignmentRecovery],
) -> list[AlignmentRecoveryCandidate]:
    recoveries_by_region = {
        recovery.region_id: recovery
        for recovery in recoveries
    }

    evidence_by_region = {
        evidence.region_id: evidence
        for evidence in region_evidence
    }

    ordered_evidence = sorted(
        region_evidence,
        key=lambda evidence: (
            evidence.start,
            evidence.end,
            evidence.region_id,
        ),
    )

    positions = {
        evidence.region_id: position
        for position, evidence in enumerate(
            ordered_evidence
        )
    }

    recovered_word_indices = {
        index
        for recovery in recoveries
        for index in recovery.word_indices
    }

    candidates: list[
        AlignmentRecoveryCandidate
    ] = []

    claimed_word_indices: set[int] = set()

    for conflict in conflicts:
        source_recovery = recoveries_by_region.get(
            conflict.recovery_region_id
        )
        source_evidence = evidence_by_region.get(
            conflict.recovery_region_id
        )

        if (
            source_recovery is None
            or source_evidence is None
        ):
            continue

        position = positions.get(
            conflict.recovery_region_id
        )

        if position is None:
            continue

        following = ordered_evidence[
            position + 1:
        ]

        if not following:
            continue

        # Only the immediately following lexical region
        # may rescue a word displaced by the recovery.
        target_evidence = following[0]

        if (
            source_evidence.speaker is None
            or target_evidence.speaker
            != source_evidence.speaker
        ):
            continue

        gap = (
            target_evidence.start
            - source_recovery.region_end
        )

        if (
            gap < -1e-6
            or gap > MAX_INTER_WORD_GAP_SECONDS
        ):
            continue

        whisper_tokens = (
            target_evidence.whisper_tokens
        )

        if len(whisper_tokens) < 3:
            continue

        start_index = (
            conflict.conflicting_word_index
        )
        end_index = (
            start_index
            + len(whisper_tokens)
            - 1
        )

        if end_index >= len(words):
            continue

        word_indices = tuple(
            range(
                start_index,
                end_index + 1,
            )
        )

        if (
            set(word_indices)
            & recovered_word_indices
        ):
            continue

        if (
            set(word_indices)
            & claimed_word_indices
        ):
            continue

        qwen_token_groups = [
            _lexical_tokens(
                str(words[index]["text"])
            )
            for index in word_indices
        ]

        # Keep the mapping deliberately simple: exactly
        # one lexical token per Qwen word.
        if any(
            len(tokens) != 1
            for tokens in qwen_token_groups
        ):
            continue

        qwen_tokens = tuple(
            tokens[0]
            for tokens in qwen_token_groups
        )

        # The conflict word itself may disagree, but the
        # complete following suffix must agree exactly.
        if (
            qwen_tokens[0]
            == whisper_tokens[0]
            or qwen_tokens[1:]
            != whisper_tokens[1:]
        ):
            continue

        # At least the shared suffix must already have
        # local temporal support from the target region.
        suffix_has_overlap = any(
            float(words[index]["start"])
            < target_evidence.end
            and float(words[index]["end"])
            > target_evidence.start
            for index in word_indices[1:]
        )

        if not suffix_has_overlap:
            continue

        candidates.append(
            AlignmentRecoveryCandidate(
                word_indices=word_indices,
                start_word_index=start_index,
                end_word_index=end_index,
                region_id=(
                    target_evidence.region_id
                ),
                region_start=(
                    target_evidence.start
                ),
                region_end=(
                    target_evidence.end
                ),
                whisper_text=(
                    target_evidence.whisper_text
                ),
                speaker=target_evidence.speaker,
                issue_word_indices=(
                    word_indices
                ),
                issue_reasons=(
                    "post_recovery_conflict_rescue",
                ),
            )
        )

        claimed_word_indices.update(
            word_indices
        )

    return candidates


def find_unclaimed_post_recovery_words(
    words: list[dict[str, Any]],
    conflicts: list[PostRecoveryWordConflict],
    region_evidence: list[AlignmentRegionEvidence],
) -> tuple[int, ...]:
    unclaimed_indices: list[int] = []

    for conflict in conflicts:
        word_index = conflict.conflicting_word_index
        word = words[word_index]

        word_start = float(word["start"])
        word_end = float(word["end"])

        claimed_by_other_region = any(
            evidence.region_id
            != conflict.recovery_region_id
            and word_start < evidence.end
            and word_end > evidence.start
            for evidence in region_evidence
        )

        if claimed_by_other_region:
            continue

        if word_index not in unclaimed_indices:
            unclaimed_indices.append(word_index)

    return tuple(unclaimed_indices)


def recover_stranded_alignment(
    storage: DatasetStorage,
    source_id: str,
    candidate: StrandedAlignmentCandidate,
    words: list[dict[str, Any]],
    *,
    representation_name: str,
    language: str,
) -> LocalAlignmentRecovery:
    _, source_path = resolve_source_representation(
        storage,
        source_id,
        representation_name,
    )

    text = " ".join(
        str(words[word_index]["text"])
        for word_index in candidate.word_indices
    )

    with TemporaryDirectory() as temp_dir:
        audio_path = Path(temp_dir) / "region.wav"

        extract_audio_region(
            source=source_path,
            destination=audio_path,
            start=candidate.region_start,
            end=candidate.region_end,
        )

        alignment = align_text_qwen3(
            audio_path,
            text=text,
            language=language,
        )

    recovered_words = tuple(
        {
            **aligned_word,
            "start": (
                float(aligned_word["start"])
                + candidate.region_start
            ),
            "end": (
                float(aligned_word["end"])
                + candidate.region_start
            ),
        }
        for aligned_word in alignment.words
    )

    return LocalAlignmentRecovery(
        word_indices=candidate.word_indices,
        region_id=candidate.region_id,
        region_start=candidate.region_start,
        region_end=candidate.region_end,
        text=text,
        words=recovered_words,
    )


def infer_region_boundaries(
    region_evidence: list[AlignmentRegionEvidence],
) -> set[int]:
    boundaries: set[int] = set()

    ordered_evidence = sorted(
        region_evidence,
        key=lambda evidence: (
            evidence.start,
            evidence.end,
            evidence.region_id,
        ),
    )

    for left, right in zip(
        ordered_evidence,
        ordered_evidence[1:],
    ):
        if len(left.text_matches) != 1:
            continue

        if len(right.text_matches) != 1:
            continue

        left_match = left.text_matches[0]
        right_match = right.text_matches[0]

        if (
            left_match.end_word_index + 1
            != right_match.start_word_index
        ):
            continue

        if left.end > right.start:
            continue

        boundaries.add(
            left_match.end_word_index
        )

    return boundaries


def validate_sat_boundaries(
    words: list[dict[str, Any]],
    region_evidence: list[AlignmentRegionEvidence],
    boundary_after_word_indices: set[int],
) -> set[int]:
    validated = set(boundary_after_word_indices)

    for boundary_index in boundary_after_word_indices:
        if (
            boundary_index < 0
            or boundary_index + 1 >= len(words)
        ):
            continue

        boundary_end = float(
            words[boundary_index]["end"]
        )

        containing_regions = [
            evidence
            for evidence in region_evidence
            if (
                evidence.start
                < boundary_end
                < evidence.end
            )
        ]

        if len(containing_regions) != 1:
            continue

        evidence = containing_regions[0]

        next_word = words[boundary_index + 1]
        next_start = float(next_word["start"])
        next_end = float(next_word["end"])

        if not (
            next_start < evidence.end
            and next_end > evidence.start
        ):
            continue

        boundary_is_spanned = any(
            match.start_word_index <= boundary_index
            and match.end_word_index > boundary_index
            for match in evidence.text_matches
        )

        if boundary_is_spanned:
            validated.discard(boundary_index)

    return validated


def build_sat_boundary_recovery_candidates(
    words: list[dict[str, Any]],
    region_evidence: list[AlignmentRegionEvidence],
    boundary_after_word_indices: set[int],
) -> list[AlignmentRecoveryCandidate]:
    candidates: list[AlignmentRecoveryCandidate] = []

    for boundary_index in sorted(
        boundary_after_word_indices
    ):
        if (
            boundary_index < 0
            or boundary_index >= len(words)
        ):
            continue

        boundary_end = float(
            words[boundary_index]["end"]
        )

        containing_regions = [
            evidence
            for evidence in region_evidence
            if (
                evidence.start
                < boundary_end
                < evidence.end
            )
        ]

        if len(containing_regions) != 1:
            continue

        evidence = containing_regions[0]

        next_word_index = boundary_index + 1

        if next_word_index >= len(words):
            continue

        next_word = words[next_word_index]
        next_start = float(next_word["start"])
        next_end = float(next_word["end"])

        if not (
            next_start < evidence.end
            and next_end > evidence.start
        ):
            continue

        boundary_is_explained = any(
            match.start_word_index <= boundary_index
            and match.end_word_index > boundary_index
            for match in evidence.text_matches
        )

        if boundary_is_explained:
            continue

        previous_boundaries = [
            index
            for index in boundary_after_word_indices
            if index < boundary_index
        ]
        
        minimum_start_index = (
            max(previous_boundaries) + 1
            if previous_boundaries
            else 0
        )

        start_index = boundary_index
        
        while start_index > minimum_start_index:
            previous_index = start_index - 1
            previous = words[previous_index]
            previous_start = float(previous["start"])
            previous_end = float(previous["end"])
        
            if (
                previous_start < evidence.start
                or previous_end > evidence.end
            ):
                break
        
            start_index = previous_index

        word_indices = tuple(
            range(
                start_index,
                boundary_index + 1,
            )
        )

        candidates.append(
            AlignmentRecoveryCandidate(
                word_indices=word_indices,
                start_word_index=start_index,
                end_word_index=boundary_index,
                region_id=evidence.region_id,
                region_start=evidence.start,
                region_end=evidence.end,
                whisper_text=evidence.whisper_text,
                speaker=evidence.speaker,
                issue_word_indices=(
                    boundary_index,
                ),
                issue_reasons=(
                    "sat_boundary_inside_region",
                ),
            )
        )

    return candidates


def build_recovery_candidates(
    issues: list[AlignmentIssue],
    region_evidence: list[AlignmentRegionEvidence],
) -> list[AlignmentRecoveryCandidate]:
    candidates: list[
        AlignmentRecoveryCandidate
    ] = []

    seen: set[
        tuple[int, int, str]
    ] = set()

    for evidence in region_evidence:
        match = evidence.unique_text_match

        if match is None:
            continue

        match_indices = set(
            match.word_indices
        )

        related_issues = [
            issue
            for issue in issues
            if (
                issue.word_indices
                and any(
                    index in match_indices
                    for index in issue.word_indices
                )
            )
        ]

        if not related_issues:
            continue

        issue_indices = tuple(
            dict.fromkeys(
                index
                for issue in related_issues
                for index in issue.word_indices
            )
        )

        reasons = tuple(
            dict.fromkeys(
                reason
                for issue in related_issues
                for reason in issue.reasons
            )
        )

        key = (
            match.start_word_index,
            match.end_word_index,
            evidence.region_id,
        )

        if key in seen:
            continue

        seen.add(key)

        candidates.append(
            AlignmentRecoveryCandidate(
                word_indices=(
                    match.word_indices
                ),
                start_word_index=(
                    match.start_word_index
                ),
                end_word_index=(
                    match.end_word_index
                ),
                region_id=evidence.region_id,
                region_start=evidence.start,
                region_end=evidence.end,
                whisper_text=(
                    evidence.whisper_text
                ),
                speaker=evidence.speaker,
                issue_word_indices=issue_indices,
                issue_reasons=reasons,
            )
        )

    return candidates


def _lexical_tokens(
    text: str,
) -> tuple[str, ...]:
    return tuple(
        match.group(0)
        .casefold()
        .replace("’", "'")
        for match in _TOKEN_RE.finditer(text)
    )


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

    if not text:
        return None

    return text


def detect_alignment_issues(
    words: list[dict[str, Any]],
) -> list[AlignmentIssue]:
    issues: list[AlignmentIssue] = []

    zero_duration_run: list[int] = []

    def flush_zero_duration_run() -> None:
        if not zero_duration_run:
            return

        issues.append(
            AlignmentIssue(
                word_indices=tuple(
                    zero_duration_run
                ),
                reasons=(
                    "zero_word_duration",
                ),
            )
        )

        zero_duration_run.clear()

    for position, word in enumerate(words):
        start = float(word["start"])
        end = float(word["end"])
        duration = end - start

        if duration <= 1e-6:
            zero_duration_run.append(
                position
            )
        else:
            flush_zero_duration_run()

        if duration > MAX_WORD_DURATION_SECONDS:
            issues.append(
                AlignmentIssue(
                    word_indices=(position,),
                    reasons=(
                        "excessive_word_duration",
                    ),
                )
            )

        if position == 0:
            continue

        previous = words[position - 1]
        previous_end = float(
            previous["end"]
        )

        if (
            start - previous_end
            > MAX_INTER_WORD_GAP_SECONDS
        ):
            issues.append(
                AlignmentIssue(
                    word_indices=(),
                    reasons=(
                        "excessive_inter_word_gap",
                    ),
                    boundary_after_word_index=(
                        position - 1
                    ),
                )
            )

    flush_zero_duration_run()

    return issues


def collect_region_evidence(
    storage: DatasetStorage,
    source_id: str,
    words: list[dict[str, Any]],
) -> list[AlignmentRegionEvidence]:
    regions = [
        region
        for region in storage.regions.load()
        if region.get("source_id") == source_id
    ]

    regions.sort(
        key=lambda region: (
            float(region["source_start"]),
            float(region["source_end"]),
            region["id"],
        )
    )

    evidence: list[
        AlignmentRegionEvidence
    ] = []

    for region in regions:
        whisper_text = _whisper_text(region)

        if whisper_text is None:
            continue

        whisper_tokens = _lexical_tokens(
            whisper_text
        )

        if not whisper_tokens:
            continue

        text_matches = find_text_matches(
            words,
            whisper_text,
        )

        evidence.append(
            AlignmentRegionEvidence(
                region_id=region["id"],
                start=float(
                    region["source_start"]
                ),
                end=float(
                    region["source_end"]
                ),
                speaker=region.get(
                    "detector_label"
                ),
                whisper_text=whisper_text,
                whisper_tokens=whisper_tokens,
                text_matches=tuple(
                    text_matches
                ),
            )
        )

    return evidence


def find_text_matches(
    words: list[dict[str, Any]],
    text: str,
) -> list[AlignmentTextMatch]:
    target_tokens = _lexical_tokens(text)

    if not target_tokens:
        return []

    word_tokens = [
        _lexical_tokens(str(word["text"]))
        for word in words
    ]

    matches: list[AlignmentTextMatch] = []

    for start_index in range(len(words)):
        collected: list[str] = []

        for end_index in range(
            start_index,
            len(words),
        ):
            collected.extend(
                word_tokens[end_index]
            )

            collected_tuple = tuple(collected)

            if collected_tuple == target_tokens:
                matches.append(
                    AlignmentTextMatch(
                        start_word_index=start_index,
                        end_word_index=end_index,
                        word_indices=tuple(
                            range(
                                start_index,
                                end_index + 1,
                            )
                        ),
                        tokens=target_tokens,
                    )
                )
                break

            if len(collected) >= len(
                target_tokens
            ):
                break

            if (
                collected_tuple
                != target_tokens[
                    :len(collected_tuple)
                ]
            ):
                break

    return matches


def recover_candidate_alignment(
    storage: DatasetStorage,
    source_id: str,
    candidate: AlignmentRecoveryCandidate,
    words: list[dict[str, Any]],
    *,
    representation_name: str,
    language: str,
) -> LocalAlignmentRecovery:
    _, source_path = resolve_source_representation(
        storage,
        source_id,
        representation_name,
    )

    text = " ".join(
        str(words[index]["text"])
        for index in candidate.word_indices
    )

    if not text.strip():
        raise ValueError(
            "Alignment recovery candidate has no text"
        )

    with TemporaryDirectory(
        prefix="voice-dataset-alignment-recovery-"
    ) as temporary_dir:
        clip_path = (
            Path(temporary_dir)
            / "region.wav"
        )

        extract_audio_region(
            source=source_path,
            destination=clip_path,
            start=candidate.region_start,
            end=candidate.region_end,
        )

        alignment = align_text_qwen3(
            clip_path,
            text=text,
            language=language,
        )

        recovered_words = tuple(
            {
                "text": word["text"],
                "start": max(
                    candidate.region_start,
                    min(
                        float(word["start"])
                        + candidate.region_start,
                        candidate.region_end,
                    ),
                ),
                "end": max(
                    candidate.region_start,
                    min(
                        float(word["end"])
                        + candidate.region_start,
                        candidate.region_end,
                    ),
                ),
            }
            for word in alignment.words
        )

    return LocalAlignmentRecovery(
        word_indices=candidate.word_indices,
        region_id=candidate.region_id,
        region_start=candidate.region_start,
        region_end=candidate.region_end,
        text=text,
        words=recovered_words,
    )


def recover_region_alignment(
    storage: DatasetStorage,
    source_id: str,
    evidence: AlignmentRegionEvidence,
    words: list[dict[str, Any]],
    *,
    representation_name: str,
    language: str,
) -> LocalAlignmentRecovery:
    match = evidence.unique_text_match

    if match is None:
        raise ValueError(
            "Region alignment recovery requires "
            "a unique text match"
        )

    candidate = AlignmentRecoveryCandidate(
        word_indices=match.word_indices,
        start_word_index=match.start_word_index,
        end_word_index=match.end_word_index,
        region_id=evidence.region_id,
        region_start=evidence.start,
        region_end=evidence.end,
        whisper_text=evidence.whisper_text,
        speaker=evidence.speaker,
        issue_word_indices=(),
        issue_reasons=(
            "boundary_alignment_probe",
        ),
    )

    return recover_candidate_alignment(
        storage,
        source_id,
        candidate,
        words,
        representation_name=representation_name,
        language=language,
    )


def alignment_recovery_is_valid(
    words: list[dict[str, Any]],
    recovery: LocalAlignmentRecovery,
) -> bool:
    if len(recovery.word_indices) != len(
        recovery.words
    ):
        return False

    if not recovery.word_indices:
        return False

    previous_recovered_end: float | None = None

    for index, recovered in zip(
        recovery.word_indices,
        recovery.words,
        strict=True,
    ):
        if index < 0 or index >= len(words):
            return False

        recovered_start = float(
            recovered["start"]
        )
        recovered_end = float(
            recovered["end"]
        )

        if (
            recovered_start < recovery.region_start
            or recovered_end > recovery.region_end
            or recovered_end < recovered_start
        ):
            return False

        if (
            previous_recovered_end is not None
            and recovered_start < previous_recovered_end
        ):
            return False

        if _lexical_tokens(
            str(words[index]["text"])
        ) != _lexical_tokens(
            str(recovered["text"])
        ):
            return False

        previous_recovered_end = recovered_end

    return True


def apply_alignment_recoveries(
    words: list[dict[str, Any]],
    recoveries: list[LocalAlignmentRecovery],
) -> list[dict[str, Any]]:
    effective_words = [
        dict(word)
        for word in words
    ]

    replaced_indices: set[int] = set()

    for recovery in recoveries:
        if len(recovery.word_indices) != len(
            recovery.words
        ):
            raise ValueError(
                "Alignment recovery word count mismatch"
            )

        for index, recovered in zip(
            recovery.word_indices,
            recovery.words,
            strict=True,
        ):
            if index < 0 or index >= len(
                effective_words
            ):
                raise ValueError(
                    "Alignment recovery word index "
                    f"out of range: {index}"
                )

            if index in replaced_indices:
                raise ValueError(
                    "Overlapping alignment recoveries "
                    f"for word {index}"
                )

            recovered_start = float(
                recovered["start"]
            )
            recovered_end = float(
                recovered["end"]
            )

            if (
                recovered_start < recovery.region_start
                or recovered_end > recovery.region_end
            ):
                raise ValueError(
                    "Alignment recovery word is "
                    "outside recovery region: "
                    f"{index} "
                    f"{recovered_start}-{recovered_end} "
                    f"not within "
                    f"{recovery.region_start}-"
                    f"{recovery.region_end}"
                )

            original = effective_words[index]

            if _lexical_tokens(
                str(original["text"])
            ) != _lexical_tokens(
                str(recovered["text"])
            ):
                raise ValueError(
                    "Alignment recovery text mismatch "
                    f"for word {index}"
                )

            replacement = dict(original)
            replacement["start"] = recovered_start
            replacement["end"] = recovered_end

            effective_words[index] = replacement
            replaced_indices.add(index)

    return effective_words


def alignment_spans_overlap(
    comparison: AlignmentComparison,
) -> bool:
    return (
        comparison.original_start
        < comparison.recovered_end
        and comparison.recovered_start
        < comparison.original_end
    )

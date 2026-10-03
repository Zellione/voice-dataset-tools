from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .speaker_candidates import SpeakerCandidate


@dataclass(frozen=True)
class ReviewerBoundaryView:
    status: str
    near_source_start: bool
    near_source_end: bool


@dataclass(frozen=True)
class ReviewerAssignmentView:
    status: str
    voice_id: str | None


@dataclass(frozen=True)
class ReviewerAutomaticPipelineView:
    status: str | None
    review_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ReviewerEmbeddingScoreView:
    embedding_name: str
    score: float
    support: int


@dataclass(frozen=True)
class ReviewerSpeakerCandidateView:
    voice_id: str
    character: str | None
    score: float
    encoder_count: int
    embedding_scores: tuple[
        ReviewerEmbeddingScoreView,
        ...,
    ]


@dataclass(frozen=True)
class ReviewerTurnView:
    turn_id: str
    source_id: str
    position: int
    total: int

    start: float
    end: float
    duration: float

    transcript: str | None
    language: str | None

    region_ids: tuple[str, ...]
    representation_names: tuple[str, ...]

    review_status: str

    boundary: ReviewerBoundaryView
    assignment: ReviewerAssignmentView
    automatic_pipeline: ReviewerAutomaticPipelineView
    speaker_evidence: ReviewerSpeakerEvidenceView

    speaker_candidates: tuple[
        ReviewerSpeakerCandidateView,
        ...,
    ]

    alignment: ReviewerAlignmentView

    edge_recovery: ReviewerEdgeRecoveryView | None


@dataclass(frozen=True)
class ReviewerSpeakerEvidenceView:
    speaker: str | None
    known_speakers: tuple[str, ...]
    unresolved_word_indices: tuple[int, ...]


@dataclass(frozen=True)
class ReviewerAlignmentRecoveryView:
    status: str | None
    source_start: float | None
    source_end: float | None
    speaker: str | None
    region_ids: tuple[str, ...]
    matched_token_count: int | None
    candidate_token_count: int | None


@dataclass(frozen=True)
class ReviewerAlignmentView:
    status: str | None
    issue_word_indices: tuple[int, ...]
    recovery: ReviewerAlignmentRecoveryView | None


@dataclass(frozen=True)
class ReviewerEdgeRecoveryView:
    status: str | None
    method: str | None
    edge: str | None

    source_start: float | None
    source_end: float | None

    region_id: str | None
    conflicting_region_id: str | None
    speaker: str | None

    candidate_token: str | None
    whisper_token: str | None

    candidate_text: str | None
    whisper_text: str | None
    conflicting_whisper_text: str | None


def _speaker_candidate_view(
    candidate: SpeakerCandidate,
    voices: dict[str, dict[str, Any]],
) -> ReviewerSpeakerCandidateView:
    voice = voices.get(candidate.voice_id)

    character = None

    if isinstance(voice, dict):
        raw_character = voice.get("character")

        if isinstance(raw_character, str) and raw_character:
            character = raw_character

    embedding_scores = tuple(
        ReviewerEmbeddingScoreView(
            embedding_name=item.embedding_name,
            score=item.score,
            support=item.support,
        )
        for item in candidate.embedding_scores
    )

    return ReviewerSpeakerCandidateView(
        voice_id=candidate.voice_id,
        character=character,
        score=candidate.score,
        encoder_count=candidate.encoder_count,
        embedding_scores=embedding_scores,
    )


def build_reviewer_turn_view(
    turn: dict[str, Any],
    *,
    position: int,
    total: int,
    voices: dict[str, dict[str, Any]],
    speaker_candidates: list[SpeakerCandidate],
) -> ReviewerTurnView:
    start = float(turn["source_start"])
    end = float(turn["source_end"])

    regions = turn.get("source_regions")
    if not isinstance(regions, list):
        regions = []

    representations = turn.get("representations")
    if not isinstance(representations, dict):
        representations = {}

    review = turn.get("review")
    if not isinstance(review, dict):
        review = {}

    boundary_review = review.get("boundary")
    if not isinstance(boundary_review, dict):
        boundary_review = {}

    assignment = turn.get("assignment")
    if not isinstance(assignment, dict):
        assignment = {}

    metadata = turn.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    boundary_evidence = metadata.get(
        "boundary_evidence"
    )
    if not isinstance(boundary_evidence, dict):
        boundary_evidence = {}

    automatic_pipeline = metadata.get(
        "automatic_pipeline"
    )
    if not isinstance(automatic_pipeline, dict):
        automatic_pipeline = {}

    speaker_evidence = metadata.get(
        "speaker_evidence"
    )
    if not isinstance(speaker_evidence, dict):
        speaker_evidence = {}

    alignment_evidence = metadata.get(
        "alignment_evidence"
    )
    if not isinstance(alignment_evidence, dict):
        alignment_evidence = {}

    alignment_status = alignment_evidence.get(
        "status"
    )
    if not isinstance(alignment_status, str):
        alignment_status = None

    edge_data = metadata.get(
        "edge_evidence"
    )

    edge_recovery = None

    if isinstance(edge_data, dict):
        def optional_string(
            key: str,
        ) -> str | None:
            value = edge_data.get(key)

            if isinstance(value, str) and value:
                return value

            return None

        def optional_float(
            key: str,
        ) -> float | None:
            value = edge_data.get(key)

            if isinstance(value, (int, float)):
                return float(value)

            return None

        edge_recovery = ReviewerEdgeRecoveryView(
            status=optional_string("status"),
            method=optional_string("method"),
            edge=optional_string("edge"),
            source_start=optional_float(
                "source_start"
            ),
            source_end=optional_float(
                "source_end"
            ),
            region_id=optional_string(
                "region_id"
            ),
            conflicting_region_id=optional_string(
                "conflicting_region_id"
            ),
            speaker=optional_string("speaker"),
            candidate_token=optional_string(
                "candidate_token"
            ),
            whisper_token=optional_string(
                "whisper_token"
            ),
            candidate_text=optional_string(
                "candidate_text"
            ),
            whisper_text=optional_string(
                "whisper_text"
            ),
            conflicting_whisper_text=optional_string(
                "conflicting_whisper_text"
            ),
        )

    issue_word_indices = alignment_evidence.get(
        "issue_word_indices"
    )
    if not isinstance(issue_word_indices, list):
        issue_word_indices = []

    recovery_data = alignment_evidence.get(
        "recovery"
    )

    recovery_view = None

    if isinstance(recovery_data, dict):
        recovery_status = recovery_data.get(
            "status"
        )
        if not isinstance(recovery_status, str):
            recovery_status = None

        recovery_start = recovery_data.get(
            "source_start"
        )
        if not isinstance(
            recovery_start,
            (int, float),
        ):
            recovery_start = None
        else:
            recovery_start = float(
                recovery_start
            )

        recovery_end = recovery_data.get(
            "source_end"
        )
        if not isinstance(
            recovery_end,
            (int, float),
        ):
            recovery_end = None
        else:
            recovery_end = float(
                recovery_end
            )

        recovery_speaker = recovery_data.get(
            "speaker"
        )
        if not isinstance(
            recovery_speaker,
            str,
        ):
            recovery_speaker = None

        recovery_regions = recovery_data.get(
            "region_ids"
        )
        if not isinstance(
            recovery_regions,
            list,
        ):
            recovery_regions = []

        matched_token_count = recovery_data.get(
            "matched_token_count"
        )
        if not isinstance(
            matched_token_count,
            int,
        ):
            matched_token_count = None

        candidate_token_count = recovery_data.get(
            "candidate_token_count"
        )
        if not isinstance(
            candidate_token_count,
            int,
        ):
            candidate_token_count = None

        recovery_view = ReviewerAlignmentRecoveryView(
            status=recovery_status,
            source_start=recovery_start,
            source_end=recovery_end,
            speaker=recovery_speaker,
            region_ids=tuple(
                str(region_id)
                for region_id in recovery_regions
            ),
            matched_token_count=(
                matched_token_count
            ),
            candidate_token_count=(
                candidate_token_count
            ),
        )

    speaker = speaker_evidence.get("speaker")
    if not isinstance(speaker, str):
        speaker = None

    known_speakers = speaker_evidence.get(
        "known_speakers"
    )
    if not isinstance(known_speakers, list):
        known_speakers = []

    unresolved_word_indices = speaker_evidence.get(
        "unresolved_word_indices"
    )
    if not isinstance(
        unresolved_word_indices,
        list,
    ):
        unresolved_word_indices = []

    review_reasons = automatic_pipeline.get(
        "review_reasons"
    )
    if not isinstance(review_reasons, list):
        review_reasons = []

    auto_status = automatic_pipeline.get("status")
    if not isinstance(auto_status, str):
        auto_status = None

    assignment_status = assignment.get("status")
    if not isinstance(assignment_status, str):
        assignment_status = "unknown"

    assignment_voice_id = assignment.get("voice_id")
    if not isinstance(assignment_voice_id, str):
        assignment_voice_id = None

    return ReviewerTurnView(
        turn_id=str(turn["id"]),
        source_id=str(turn["source_id"]),
        position=position,
        total=total,
        start=start,
        end=end,
        duration=end - start,
        transcript=(
            turn.get("transcript")
            if isinstance(
                turn.get("transcript"),
                str,
            )
            else None
        ),
        language=(
            turn.get("language")
            if isinstance(
                turn.get("language"),
                str,
            )
            else None
        ),
        region_ids=tuple(
            str(region_id)
            for region_id in regions
        ),
        representation_names=tuple(
            str(name)
            for name in representations
        ),
        review_status=(
            str(review.get("status") or "pending")
        ),
        boundary=ReviewerBoundaryView(
            status=str(
                boundary_review.get("status")
                or "unknown"
            ),
            near_source_start=bool(
                boundary_evidence.get(
                    "near_source_start"
                )
            ),
            near_source_end=bool(
                boundary_evidence.get(
                    "near_source_end"
                )
            ),
        ),
        assignment=ReviewerAssignmentView(
            status=assignment_status,
            voice_id=assignment_voice_id,
        ),
        automatic_pipeline=ReviewerAutomaticPipelineView(
            status=auto_status,
            review_reasons=tuple(
                str(reason)
                for reason in review_reasons
            ),
        ),
        speaker_evidence=ReviewerSpeakerEvidenceView(
            speaker=speaker,
            known_speakers=tuple(
                str(known_speaker)
                for known_speaker in known_speakers
            ),
            unresolved_word_indices=tuple(
                int(index)
                for index in unresolved_word_indices
                if isinstance(index, int)
            ),
        ),
        alignment=ReviewerAlignmentView(
            status=alignment_status,
            issue_word_indices=tuple(
                int(index)
                for index in issue_word_indices
                if isinstance(index, int)
            ),
            recovery=recovery_view,
        ),
        edge_recovery=edge_recovery,
        speaker_candidates=tuple(
            _speaker_candidate_view(
                candidate,
                voices,
            )
            for candidate in speaker_candidates
        ),
    )

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


AssignmentStatus = Literal[
    "unknown",
    "assigned",
    "ignore",
]


@dataclass
class AudioRepresentation:
    path: str
    kind: str

    processor: str | None = None
    processor_version: str | None = None

    sample_rate: int | None = None
    channels: int | None = None

    purposes: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EmbeddingReference:
    encoder: str
    representation: str
    path: str

    dimension: int | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TranscriptHypothesis:
    text: str | None = None
    language: str | None = None

    model: str | None = None
    representation: str | None = None

    confidence: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict:
        return asdict(self)


ReconciliationStatus = Literal[
    "pending",
    "rejected",
]

RejectionReason = Literal[
    "non_speech",
    "unusable",
    "duplicate",
    "other",
]

REJECTION_REASONS: frozenset[str] = frozenset({
    "non_speech",
    "unusable",
    "duplicate",
    "other",
})


@dataclass
class RegionReconciliation:
    status: ReconciliationStatus = "pending"

    reason: RejectionReason | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.status == "rejected":
            if self.reason is None:
                raise ValueError(
                    "rejected reconciliation "
                    "requires reason"
                )

            if self.reason not in REJECTION_REASONS:
                raise ValueError(
                    "invalid rejection reason: "
                    f"{self.reason!r}"
                )

        elif self.reason is not None:
            raise ValueError(
                "pending reconciliation "
                "must not contain reason"
            )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SourceRecord:
    id: str

    media_path: str | None = None

    representations: dict[
        str,
        AudioRepresentation,
    ] = field(default_factory=dict)

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "record_type": "source",

            "id": self.id,
            "media_path": self.media_path,

            "representations": {
                name: representation.to_dict()
                for name, representation
                in self.representations.items()
            },

            "metadata": self.metadata,
        }


@dataclass
class CandidateRegion:
    id: str
    source_id: str

    source_start: float
    source_end: float

    detector: str
    detector_label: str | None = None

    representations: dict[
        str,
        AudioRepresentation,
    ] = field(default_factory=dict)

    transcripts: dict[
        str,
        TranscriptHypothesis,
    ] = field(default_factory=dict)

    embeddings: dict[
        str,
        EmbeddingReference,
    ] = field(default_factory=dict)

    reconciliation: RegionReconciliation = field(
        default_factory=RegionReconciliation
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict:
        return {
            "schema_version": 2,
            "record_type": "candidate_region",

            "id": self.id,
            "source_id": self.source_id,

            "source_start": self.source_start,
            "source_end": self.source_end,

            "detector": self.detector,
            "detector_label": self.detector_label,

            "representations": {
                name: representation.to_dict()
                for name, representation
                in self.representations.items()
            },

            "transcripts": {
                name: transcript.to_dict()
                for name, transcript
                in self.transcripts.items()
            },

            "embeddings": {
                name: embedding.to_dict()
                for name, embedding
                in self.embeddings.items()
            },

            "reconciliation":
                self.reconciliation.to_dict(),

            "metadata": self.metadata,
        }


@dataclass
class VoiceAssignment:
    status: AssignmentStatus = "unknown"

    voice_id: str | None = None

    method: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.status == "assigned":
            if not self.voice_id:
                raise ValueError(
                    "assigned voice requires voice_id"
                )

        elif self.voice_id is not None:
            raise ValueError(
                f"{self.status} assignment "
                "must not contain voice_id"
            )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TurnRecord:
    id: str
    source_id: str

    source_start: float
    source_end: float
    source_regions: list[str] = field(
        default_factory=list
    )

    language: str | None = None
    transcript: str | None = None

    representations: dict[
        str,
        AudioRepresentation,
    ] = field(default_factory=dict)

    embeddings: dict[
        str,
        EmbeddingReference,
    ] = field(default_factory=dict)

    assignment: VoiceAssignment = field(
        default_factory=VoiceAssignment
    )

    review: dict[str, Any] = field(
        default_factory=lambda: {
            "status": "pending"
        }
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict:
        return {
            "schema_version": 2,
            "record_type": "turn",

            "id": self.id,
            "source_id": self.source_id,

            "source_start": self.source_start,
            "source_end": self.source_end,
            "source_regions":
                self.source_regions,

            "language": self.language,
            "transcript": self.transcript,

            "representations": {
                name: representation.to_dict()
                for name, representation
                in self.representations.items()
            },

            "embeddings": {
                name: embedding.to_dict()
                for name, embedding
                in self.embeddings.items()
            },

            "assignment":
                self.assignment.to_dict(),

            "review": self.review,
            "metadata": self.metadata,
        }


@dataclass
class VoiceProfile:
    id: str

    character: str | None = None
    language: str | None = None

    aliases: list[str] = field(
        default_factory=list
    )

    ignored: bool = False
    notes: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "record_type": "voice_profile",

            "id": self.id,

            "character": self.character,
            "language": self.language,

            "aliases": self.aliases,
            "ignored": self.ignored,
            "notes": self.notes,

            "metadata": self.metadata,
        }

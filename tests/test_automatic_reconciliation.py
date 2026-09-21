from __future__ import annotations

from pathlib import Path

from voice_dataset.automatic_reconciliation import (
    reconcile_source_regions,
)
from voice_dataset.schema import SourceRecord
from voice_dataset.storage import DatasetStorage


def make_storage(
    tmp_path: Path,
) -> DatasetStorage:
    storage = DatasetStorage(
        tmp_path / "dataset"
    )

    storage.add_source(
        SourceRecord(
            id="source_001",
            media_path="/media/source.mkv",
        )
    )

    return storage


def add_region(
    storage: DatasetStorage,
    *,
    region_id: str,
    start: float,
    end: float,
    text: str | None,
    language: str | None = "English",
    reconciliation_status: str = "pending",
) -> None:
    storage.regions.append(
        {
            "schema_version": 2,
            "record_type": "candidate_region",
            "id": region_id,
            "source_id": "source_001",
            "source_start": start,
            "source_end": end,
            "representations": {},
            "transcripts": {
                "whisper": {
                    "text": text,
                    "language": language,
                },
            },
            "embeddings": {},
            "reconciliation": {
                "status": reconciliation_status,
                "reason": (
                    "non_speech"
                    if reconciliation_status == "rejected"
                    else None
                ),
                "notes": None,
            },
            "metadata": {},
        }
    )


def test_reconcile_source_regions_handles_real_region_states(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    add_region(
        storage,
        region_id="region_000001",
        start=1.0,
        end=2.0,
        text="  Keep this.  ",
    )

    add_region(
        storage,
        region_id="region_000002",
        start=3.0,
        end=4.0,
        text=None,
    )

    add_region(
        storage,
        region_id="region_000003",
        start=5.0,
        end=6.0,
        text="Do not create this.",
        reconciliation_status="rejected",
    )

    result = reconcile_source_regions(
        storage,
        "source_001",
    )

    assert result.created == 1
    assert result.skipped == 1
    assert result.unresolved == 1

    turns = storage.turns.load()

    assert len(turns) == 1

    turn = turns[0]

    assert turn["source_regions"] == [
        "region_000001",
    ]
    assert turn["source_start"] == 1.0
    assert turn["source_end"] == 2.0
    assert turn["transcript"] == "Keep this."
    assert turn["language"] == "English"

    assert turn["metadata"]["creation"] == {
        "method": (
            "automatic_transcript_reconciliation"
        ),
    }

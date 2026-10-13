from pathlib import Path

import pytest

from voice_dataset.schema import (
    AudioRepresentation,
    SourceRecord,
)
from voice_dataset.storage import DatasetStorage


def test_add_and_get_source(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)

    source = SourceRecord(
        id="arcane-s01e09-test",
        media_path="/media/arcane.mkv",
        representations={
            "center": AudioRepresentation(
                path="/scratch/arcane/center.wav",
                kind="center",
                sample_rate=48000,
                channels=1,
                duration=60.0,
                media_start=600.0,
                purposes=[
                    "speaker_embedding",
                    "boundary_analysis",
                    "context",
                ],
                stream_index=1,
                channel_mode="center",
            ),
            "speech": AudioRepresentation(
                path=(
                    "/scratch/arcane/"
                    "speech_estimate.wav"
                ),
                kind="separated_speech",
                processor="bandit",
                sample_rate=48000,
                channels=1,
                purposes=[
                    "asr",
                    "review",
                    "tts_candidate",
                ],
            ),
        },
        metadata={
            "episode": "S01E09",
        },
    )

    storage.add_source(source)

    result = storage.get_source(
        "arcane-s01e09-test"
    )

    assert result is not None
    assert result["record_type"] == "source"
    assert result["schema_version"] == 1

    assert result["id"] == "arcane-s01e09-test"
    assert result["media_path"] == "/media/arcane.mkv"

    assert (
        result["representations"]["center"]["kind"]
        == "center"
    )

    assert (
        result["representations"]["center"]["path"]
        == "/scratch/arcane/center.wav"
    )

    assert (
        result["representations"]["center"]["stream_index"]
        == 1
    )
    assert (
        result["representations"]["center"]["channel_mode"]
        == "center"
    )

    assert (
        result["representations"]["center"][
            "duration"
        ]
        == 60.0
    )

    assert (
        result["representations"]["center"][
            "media_start"
        ]
        == 600.0
    )

    assert (
        "context"
        in result["representations"]["center"][
            "purposes"
        ]
    )

    assert (
        result["representations"]["speech"][
            "processor"
        ]
        == "bandit"
    )

    assert result["metadata"] == {
        "episode": "S01E09",
    }


def test_get_unknown_source(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)

    assert (
        storage.get_source("does-not-exist")
        is None
    )


def test_duplicate_source_is_rejected(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)

    source = SourceRecord(
        id="source-test",
    )

    storage.add_source(source)

    before = storage.sources.path.read_bytes()

    with pytest.raises(
        ValueError,
        match="Record already exists",
    ):
        storage.add_source(source)

    assert storage.sources.path.read_bytes() == before


def test_update_source_preserves_other_fields(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)

    source = SourceRecord(
        id="source-1",
        media_path="/media/episode.mkv",
        representations={
            "center": AudioRepresentation(
                path="/scratch/center.wav",
                kind="center",
                purposes=["context"],
            ),
        },
        metadata={
            "episode": "S01E09",
        },
    )

    storage.add_source(source)

    def update(record):
        representation = record[
            "representations"
        ]["center"]

        representation["media_start"] = 600.0
        representation["stream_index"] = 1
        representation["channel_mode"] = "center"

        return record

    updated = storage.update_source(
        "source-1",
        update,
    )

    assert updated["media_path"] == (
        "/media/episode.mkv"
    )
    assert updated["metadata"] == {
        "episode": "S01E09",
    }

    representation = updated[
        "representations"
    ]["center"]

    assert representation["media_start"] == 600.0
    assert representation["stream_index"] == 1
    assert representation["channel_mode"] == "center"

    stored = storage.get_source("source-1")

    assert stored == updated


def test_dataset_storage_uses_project_voices(
    tmp_path: Path,
):
    dataset_root = tmp_path / "s01e01"
    project_root = tmp_path

    storage = DatasetStorage(
        dataset_root,
        project_root=project_root,
    )

    assert storage.root == dataset_root
    assert storage.project_root == project_root

    assert storage.sources.path == (
        dataset_root / "sources.jsonl"
    )
    assert storage.turns.path == (
        dataset_root / "turns.jsonl"
    )
    assert storage.regions.path == (
        dataset_root / "regions.jsonl"
    )

    assert storage.voices.path == (
        project_root / "voices.jsonl"
    )


def test_dataset_storage_without_project_keeps_local_voices(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)

    assert storage.project_root is None
    assert storage.voices.path == (
        tmp_path / "voices.jsonl"
    )


def test_project_scoped_voices_are_shared_between_datasets(
    tmp_path: Path,
):
    from voice_dataset.schema import VoiceProfile

    project_root = tmp_path / "arcane"

    episode_1 = DatasetStorage(
        project_root / "s01e01",
        project_root=project_root,
    )

    episode_2 = DatasetStorage(
        project_root / "s01e02",
        project_root=project_root,
    )

    assert episode_1.next_voice_id() == "voice_001"
    assert episode_2.next_voice_id() == "voice_001"

    episode_1.add_voice(
        VoiceProfile(
            id="voice_001",
            character="Vi",
            language="en",
        )
    )

    assert episode_2.get_voice("voice_001") == (
        episode_1.get_voice("voice_001")
    )

    assert episode_1.next_voice_id() == "voice_002"
    assert episode_2.next_voice_id() == "voice_002"

    episode_2.add_voice(
        VoiceProfile(
            id="voice_002",
            character="Vander",
            language="en",
        )
    )

    assert [
        voice["id"]
        for voice in episode_1.voices.load()
    ] == [
        "voice_001",
        "voice_002",
    ]

    assert [
        voice["id"]
        for voice in episode_2.voices.load()
    ] == [
        "voice_001",
        "voice_002",
    ]


def test_project_datasets_discovers_episode_directories(
    tmp_path: Path,
):
    project_root = tmp_path / "arcane"

    episode_1_root = project_root / "s01e01"
    episode_2_root = project_root / "s01e02"

    episode_1_root.mkdir(parents=True)
    episode_2_root.mkdir(parents=True)

    (episode_1_root / "turns.jsonl").write_text(
        "",
        encoding="utf-8",
    )

    (episode_2_root / "turns.jsonl").write_text(
        "",
        encoding="utf-8",
    )

    storage = DatasetStorage(
        episode_2_root,
        project_root=project_root,
    )

    datasets = storage.project_datasets()

    assert [
        dataset.root
        for dataset in datasets
    ] == [
        episode_1_root,
        episode_2_root,
    ]

    assert all(
        dataset.project_root == project_root
        for dataset in datasets
    )


def test_project_datasets_without_project_returns_self(
    tmp_path: Path,
):
    storage = DatasetStorage(tmp_path)

    datasets = storage.project_datasets()

    assert len(datasets) == 1
    assert datasets[0].root == tmp_path
    assert datasets[0].project_root is None


def test_for_dataset_detects_project_voice_catalog(
    tmp_path: Path,
):
    project_root = tmp_path / "arcane"
    dataset_root = project_root / "s01e02"

    dataset_root.mkdir(parents=True)

    (project_root / "voices.jsonl").write_text(
        "",
        encoding="utf-8",
    )

    storage = DatasetStorage.for_dataset(
        dataset_root
    )

    assert storage.root == dataset_root
    assert storage.project_root == project_root
    assert storage.voices.path == (
        project_root / "voices.jsonl"
    )


def test_for_dataset_keeps_standalone_dataset_local(
    tmp_path: Path,
):
    dataset_root = tmp_path / "standalone"
    dataset_root.mkdir()

    storage = DatasetStorage.for_dataset(
        dataset_root
    )

    assert storage.root == dataset_root
    assert storage.project_root is None
    assert storage.voices.path == (
        dataset_root / "voices.jsonl"
    )

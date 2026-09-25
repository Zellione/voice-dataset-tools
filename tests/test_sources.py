from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice_dataset.schema import (
    AudioRepresentation,
    SourceRecord,
)
from voice_dataset.sources import (
    probe_audio_representation,
    register_derived_representation,
    resolve_source_representation,
    set_representation_provenance,
    resolve_source_representation_for_purpose,
)
from voice_dataset.storage import DatasetStorage


def make_storage(
    tmp_path: Path,
) -> DatasetStorage:
    storage = DatasetStorage(tmp_path)

    audio_path = tmp_path / "center.wav"

    sample_rate = 48000
    duration = 2.0

    audio = np.zeros(
        round(sample_rate * duration),
        dtype=np.float32,
    )

    sf.write(
        audio_path,
        audio,
        sample_rate,
        subtype="PCM_16",
    )

    storage.add_source(
        SourceRecord(
            id="source-1",
            media_path="/media/episode.mkv",
            representations={
                "center": AudioRepresentation(
                    path=str(audio_path),
                    kind="center",
                    purposes=["context"],
                ),
            },
        )
    )

    return storage


def test_probe_audio_representation(
    tmp_path: Path,
):
    storage = make_storage(tmp_path)

    source = storage.get_source("source-1")

    assert source is not None

    path = Path(
        source["representations"]["center"]["path"]
    )

    result = probe_audio_representation(path)

    assert result == {
        "sample_rate": 48000,
        "channels": 1,
        "duration": 2.0,
    }


def test_set_representation_provenance(
    tmp_path: Path,
):
    storage = make_storage(tmp_path)

    result = set_representation_provenance(
        storage,
        "source-1",
        "center",
        media_start=600.0,
        stream_index=1,
        channel_mode="center",
    )

    representation = result[
        "representations"
    ]["center"]

    assert representation["media_start"] == 600.0
    assert representation["stream_index"] == 1
    assert representation["channel_mode"] == "center"

    assert representation["sample_rate"] == 48000
    assert representation["channels"] == 1
    assert representation["duration"] == 2.0

    stored = storage.get_source("source-1")

    assert stored == result


def test_set_representation_provenance_rejects_unknown_representation(
    tmp_path: Path,
):
    storage = make_storage(tmp_path)

    with pytest.raises(
        KeyError,
        match="Source representation does not exist",
    ):
        set_representation_provenance(
            storage,
            "source-1",
            "missing",
            media_start=600.0,
            stream_index=1,
            channel_mode="center",
        )


@pytest.mark.parametrize(
    (
        "media_start",
        "stream_index",
        "channel_mode",
        "message",
    ),
    [
        (
            -1.0,
            1,
            "center",
            "media_start must not be negative",
        ),
        (
            600.0,
            -1,
            "center",
            "stream_index must not be negative",
        ),
        (
            600.0,
            1,
            "invalid",
            "Unsupported channel_mode",
        ),
    ],
)
def test_set_representation_provenance_rejects_invalid_values(
    tmp_path: Path,
    media_start: float,
    stream_index: int,
    channel_mode: str,
    message: str,
):
    storage = make_storage(tmp_path)

    with pytest.raises(
        ValueError,
        match=message,
    ):
        set_representation_provenance(
            storage,
            "source-1",
            "center",
            media_start=media_start,
            stream_index=stream_index,
            channel_mode=channel_mode,
        )


def test_resolve_source_representation(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    representation, path = (
        resolve_source_representation(
            storage,
            "source-1",
            "center",
        )
    )

    assert representation["kind"] == "center"
    assert path == (
        tmp_path / "center.wav"
    ).resolve()


def test_resolve_source_representation_for_purpose(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    name, representation, path = (
        resolve_source_representation_for_purpose(
            storage,
            "source-1",
            "context",
        )
    )

    assert name == "center"
    assert representation["kind"] == "center"
    assert path == (
        tmp_path / "center.wav"
    ).resolve()


def test_resolve_source_representation_for_purpose_rejects_missing_purpose(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    with pytest.raises(
        ValueError,
        match="No source representation supports",
    ):
        resolve_source_representation_for_purpose(
            storage,
            "source-1",
            "asr",
        )


def test_resolve_source_representation_for_purpose_rejects_ambiguous_purpose(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    storage.update_source(
        "source-1",
        lambda record: {
            **record,
            "representations": {
                **record["representations"],
                "other": {
                    **record["representations"]["center"],
                },
            },
        },
    )

    with pytest.raises(
        ValueError,
        match=(
            "Multiple source representations support "
            "purpose 'context'"
        ),
    ):
        resolve_source_representation_for_purpose(
            storage,
            "source-1",
            "context",
        )


def test_resolve_relative_source_representation(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    storage.update_source(
        "source-1",
        lambda record: {
            **record,
            "representations": {
                **record["representations"],
                "center": {
                    **record[
                        "representations"
                    ]["center"],
                    "path": "center.wav",
                },
            },
        },
    )

    _, path = resolve_source_representation(
        storage,
        "source-1",
        "center",
    )

    assert path == (
        tmp_path / "center.wav"
    ).resolve()


def test_resolve_source_representation_rejects_unknown_source(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    with pytest.raises(
        KeyError,
        match="Source does not exist",
    ):
        resolve_source_representation(
            storage,
            "missing",
            "center",
        )


def test_resolve_source_representation_rejects_unknown_representation(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    with pytest.raises(
        KeyError,
        match=(
            "Source representation "
            "does not exist"
        ),
    ):
        resolve_source_representation(
            storage,
            "source-1",
            "missing",
        )


def test_resolve_source_representation_rejects_missing_file(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    storage.update_source(
        "source-1",
        lambda record: {
            **record,
            "representations": {
                **record["representations"],
                "center": {
                    **record[
                        "representations"
                    ]["center"],
                    "path": "missing.wav",
                },
            },
        },
    )

    with pytest.raises(
        ValueError,
        match=(
            "Source representation audio "
            "does not exist"
        ),
    ):
        resolve_source_representation(
            storage,
            "source-1",
            "center",
        )


def test_register_derived_representation(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    set_representation_provenance(
        storage,
        "source-1",
        "center",
        media_start=600.0,
        stream_index=1,
        channel_mode="center",
    )

    speech_path = tmp_path / "speech.wav"

    audio = np.zeros(
        48000 * 2,
        dtype=np.float32,
    )

    sf.write(
        speech_path,
        audio,
        48000,
        subtype="PCM_16",
    )

    result = register_derived_representation(
        storage,
        "source-1",
        "speech",
        path=speech_path,
        kind="separated_speech",
        parent_representation_name="center",
        processor="bandit-v2",
        processor_version="revision-123",
        purposes=[
            "asr",
            "review",
            "tts_candidate",
        ],
        metadata={
            "checkpoint_sha256": "abc123",
            "parameters": {
                "chunk": 30.0,
                "overlap": 2.0,
            },
        },
    )

    representation = result[
        "representations"
    ]["speech"]

    assert representation == {
        "path": "speech.wav",
        "kind": "separated_speech",
        "processor": "bandit-v2",
        "processor_version": "revision-123",
        "sample_rate": 48000,
        "channels": 1,
        "duration": 2.0,
        "media_start": 600.0,
        "stream_index": 1,
        "channel_mode": "center",
        "purposes": [
            "asr",
            "review",
            "tts_candidate",
        ],
        "metadata": {
            "checkpoint_sha256": "abc123",
            "parameters": {
                "chunk": 30.0,
                "overlap": 2.0,
            },
        },
    }


def test_register_derived_representation_is_idempotent(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    speech_path = tmp_path / "speech.wav"

    sf.write(
        speech_path,
        np.zeros(48000, dtype=np.float32),
        48000,
        subtype="PCM_16",
    )

    arguments = {
        "path": speech_path,
        "kind": "separated_speech",
        "parent_representation_name": "center",
        "processor": "bandit-v2",
        "processor_version": "revision-123",
        "purposes": ["asr"],
        "metadata": {
            "parameters": {
                "chunk": 30.0,
                "overlap": 2.0,
            },
        },
    }

    first = register_derived_representation(
        storage,
        "source-1",
        "speech",
        **arguments,
    )

    before = (
        storage.sources.path.read_bytes()
    )

    second = register_derived_representation(
        storage,
        "source-1",
        "speech",
        **arguments,
    )

    after = (
        storage.sources.path.read_bytes()
    )

    assert second == first
    assert after == before


def test_register_derived_representation_rejects_conflict(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    first_path = tmp_path / "speech-a.wav"
    second_path = tmp_path / "speech-b.wav"

    audio = np.zeros(
        48000,
        dtype=np.float32,
    )

    sf.write(
        first_path,
        audio,
        48000,
        subtype="PCM_16",
    )

    sf.write(
        second_path,
        audio,
        48000,
        subtype="PCM_16",
    )

    register_derived_representation(
        storage,
        "source-1",
        "speech",
        path=first_path,
        kind="separated_speech",
        parent_representation_name="center",
        processor="bandit-v2",
    )

    before = (
        storage.sources.path.read_bytes()
    )

    with pytest.raises(
        ValueError,
        match=(
            "Source representation already exists "
            "with different provenance"
        ),
    ):
        register_derived_representation(
            storage,
            "source-1",
            "speech",
            path=second_path,
            kind="separated_speech",
            parent_representation_name="center",
            processor="bandit-v2",
        )

    after = (
        storage.sources.path.read_bytes()
    )

    assert after == before


def test_register_derived_representation_rejects_external_path(
    tmp_path: Path,
) -> None:
    storage = make_storage(tmp_path)

    external_root = tmp_path.parent / (
        f"{tmp_path.name}-external"
    )
    external_root.mkdir()

    speech_path = external_root / "speech.wav"

    sf.write(
        speech_path,
        np.zeros(48000, dtype=np.float32),
        48000,
        subtype="PCM_16",
    )

    try:
        with pytest.raises(
            ValueError,
            match=(
                "Derived representation must be "
                "stored inside the dataset"
            ),
        ):
            register_derived_representation(
                storage,
                "source-1",
                "speech",
                path=speech_path,
                kind="separated_speech",
                parent_representation_name="center",
                processor="bandit-v2",
            )

        source = storage.get_source("source-1")

        assert source is not None
        assert (
            "speech"
            not in source["representations"]
        )
    finally:
        speech_path.unlink(missing_ok=True)
        external_root.rmdir()

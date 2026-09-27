from pathlib import Path

import numpy as np
import soundfile as sf

import voice_dataset.ingest as ingest_module
from voice_dataset.media import AudioStream
from voice_dataset.storage import DatasetStorage


def test_ingest_preserves_audio_position_and_stream_index(
    tmp_path: Path,
    monkeypatch,
):
    source = tmp_path / "episode.mkv"
    source.touch()

    output = tmp_path / "dataset"

    streams = [
        AudioStream(
            index=1,
            codec="aac",
            sample_rate=48000,
            channels=2,
            channel_layout="stereo",
            language="deu",
            title=None,
            is_default=True,
        ),
        AudioStream(
            index=4,
            codec="aac",
            sample_rate=48000,
            channels=6,
            channel_layout="5.1",
            language="eng",
            title="English",
            is_default=False,
        ),
    ]

    monkeypatch.setattr(
        ingest_module,
        "probe_audio_streams",
        lambda source: streams,
    )

    normalize_calls = []

    def fake_normalize_media(
        source,
        destination,
        stream_index,
        channel_mode="mono",
        start=None,
        duration=None,
    ):
        normalize_calls.append(
            {
                "stream_index": stream_index,
                "channel_mode": channel_mode,
            }
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        sf.write(
            destination,
            np.zeros(48000, dtype=np.float32),
            48000,
            subtype="PCM_16",
        )

    monkeypatch.setattr(
        ingest_module,
        "normalize_media",
        fake_normalize_media,
    )

    record = ingest_module.ingest(
        source=source,
        output=output,
        source_id="episode",
        language="eng",
    )

    assert normalize_calls == [
        {
            "stream_index": 4,
            "channel_mode": "center",
        }
    ]

    representation = record.representations["center"]

    assert representation.stream_index == 4
    assert representation.metadata["audio_stream"] == 1
    assert representation.metadata["source_language"] == "eng"

    stored = DatasetStorage(output).get_source(
        "episode"
    )

    assert stored is not None

    stored_representation = stored[
        "representations"
    ]["center"]

    assert stored_representation["stream_index"] == 4
    assert (
        stored_representation["metadata"]["audio_stream"]
        == 1
    )


    assert stored["metadata"]["ingest"] == {
        "requested_audio_stream": None,
        "requested_audio_language": "eng",
        "requested_channel": "auto",
        "requested_start": None,
        "requested_duration": None,
        "audio_stream": 1,
        "stream_index": 4,
        "channel_mode": "center",
        "source_language": "eng",
    }

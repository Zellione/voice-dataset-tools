from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

import voice_dataset.playback as playback
from voice_dataset.playback import (
    external_representation_path,
    preferred_context_representation,
    preferred_review_representation,
    representation_path,
)


def test_preferred_review_representation():
    turn = {
        "id": "turn_000001",
        "representations": {
            "raw": {
                "path": "turns/turn_000001/raw.wav",
                "purposes": [
                    "speaker_embedding",
                    "boundary_analysis",
                ],
            },
            "speech": {
                "path": "turns/turn_000001/speech.wav",
                "purposes": [
                    "asr",
                    "review",
                    "tts_candidate",
                ],
            },
        },
    }

    name, representation = (
        preferred_review_representation(turn)
    )

    assert name == "speech"
    assert (
        representation["path"]
        == "turns/turn_000001/speech.wav"
    )


def test_preferred_review_representation_missing():
    turn = {
        "id": "turn_000001",
        "representations": {
            "raw": {
                "path": "turns/turn_000001/raw.wav",
                "purposes": [
                    "speaker_embedding",
                ],
            },
        },
    }

    with pytest.raises(
        ValueError,
        match="has no audio representation for review",
    ):
        preferred_review_representation(turn)


def test_preferred_context_representation():
    source = {
        "id": "source-test",
        "representations": {
            "center": {
                "path": "/tmp/center.wav",
                "purposes": [
                    "boundary_analysis",
                    "context",
                ],
            },
            "speech": {
                "path": "/tmp/speech.wav",
                "purposes": [
                    "asr",
                    "review",
                ],
            },
        },
    }

    name, representation = (
        preferred_context_representation(source)
    )

    assert name == "center"
    assert representation["path"] == "/tmp/center.wav"


def test_preferred_context_representation_missing():
    source = {
        "id": "source-test",
        "representations": {},
    }

    with pytest.raises(
        ValueError,
        match="has no audio representation for context",
    ):
        preferred_context_representation(source)


def test_representation_path(
    tmp_path: Path,
):
    audio_path = (
        tmp_path
        / "turns"
        / "turn_000001"
        / "speech.wav"
    )
    audio_path.parent.mkdir(parents=True)
    audio_path.touch()

    result = representation_path(
        tmp_path,
        {
            "path": (
                "turns/turn_000001/speech.wav"
            )
        },
    )

    assert result == audio_path.resolve()


def test_representation_path_missing_file(
    tmp_path: Path,
):
    with pytest.raises(
        FileNotFoundError,
        match="does not exist",
    ):
        representation_path(
            tmp_path,
            {
                "path": (
                    "turns/turn_000001/speech.wav"
                )
            },
        )


def test_representation_path_refuses_escape(
    tmp_path: Path,
):
    outside = tmp_path.parent / "outside.wav"
    outside.touch()

    with pytest.raises(
        ValueError,
        match="outside the dataset",
    ):
        representation_path(
            tmp_path,
            {
                "path": "../outside.wav",
            },
        )


def test_external_representation_path(
    tmp_path: Path,
):
    audio_path = tmp_path / "source.wav"
    audio_path.touch()

    result = external_representation_path(
        {
            "path": str(audio_path),
        }
    )

    assert result == audio_path.resolve()


def test_external_representation_path_requires_absolute():
    with pytest.raises(
        ValueError,
        match="must be absolute",
    ):
        external_representation_path(
            {
                "path": "source.wav",
            }
        )


def test_external_representation_path_missing(
    tmp_path: Path,
):
    missing = tmp_path / "missing.wav"

    with pytest.raises(
        FileNotFoundError,
        match="does not exist",
    ):
        external_representation_path(
            {
                "path": str(missing),
            }
        )


def test_play_turn_context_clamps_start_and_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    audio_path = tmp_path / "source.wav"

    sample_rate = 1000
    audio = np.zeros(
        sample_rate * 3,
        dtype=np.float32,
    )

    sf.write(
        audio_path,
        audio,
        sample_rate,
    )

    played = {}

    def fake_play(
        audio,
        sample_rate,
    ):
        played["frames"] = len(audio)
        played["sample_rate"] = sample_rate

    monkeypatch.setattr(
        playback._playback_engine,
        "play",
        fake_play,
    )

    source = {
        "id": "source-test",
        "representations": {
            "center": {
                "path": str(audio_path),
                "purposes": ["context"],
            },
        },
    }

    turn = {
        "id": "turn_000001",
        "source_id": "source-test",
        "source_start": 0.5,
        "source_end": 2.5,
    }

    (
        name,
        path,
        actual_start,
        actual_end,
    ) = playback.play_turn_context(
        source,
        turn,
        padding=2.0,
    )

    assert name == "center"
    assert path == audio_path.resolve()
    assert actual_start == pytest.approx(0.0)
    assert actual_end == pytest.approx(3.0)

    assert played["sample_rate"] == sample_rate
    assert played["frames"] == sample_rate * 3


def test_play_turn_context_rejects_source_mismatch(
    tmp_path: Path,
):
    audio_path = tmp_path / "source.wav"
    audio_path.touch()

    source = {
        "id": "source-a",
        "representations": {
            "center": {
                "path": str(audio_path),
                "purposes": ["context"],
            },
        },
    }

    turn = {
        "source_id": "source-b",
        "source_start": 1.0,
        "source_end": 2.0,
    }

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        playback.play_turn_context(
            source,
            turn,
        )


def test_has_original_media_mapping():
    representation = {
        "media_start": 600.0,
        "stream_index": 1,
        "channel_mode": "center",
    }

    assert playback.has_original_media_mapping(
        representation
    )


def test_has_original_media_mapping_requires_all_fields():
    representation = {
        "media_start": 600.0,
        "stream_index": 1,
    }

    assert not playback.has_original_media_mapping(
        representation
    )


def test_play_turn_context_uses_original_media(
    tmp_path: Path,
    monkeypatch,
):
    media_path = tmp_path / "episode.mkv"
    media_path.touch()

    source = {
        "id": "source-1",
        "media_path": str(media_path),
        "representations": {
            "center": {
                "path": "/unused/center.wav",
                "kind": "center",
                "media_start": 600.0,
                "stream_index": 1,
                "channel_mode": "center",
                "purposes": ["context"],
            },
        },
    }

    turn = {
        "id": "turn-1",
        "source_id": "source-1",
        "source_start": 0.031,
        "source_end": 1.027,
    }

    extraction = {}
    played = []

    def fake_extract_media_audio_region(
        source,
        destination,
        start,
        end,
        *,
        stream_index,
        channel_mode,
    ):
        extraction.update(
            {
                "source": source,
                "destination": destination,
                "start": start,
                "end": end,
                "stream_index": stream_index,
                "channel_mode": channel_mode,
            }
        )
        destination.touch()

    def fake_play_file(path):
        played.append(path)

    monkeypatch.setattr(
        playback,
        "extract_media_audio_region",
        fake_extract_media_audio_region,
    )
    monkeypatch.setattr(
        playback,
        "play_file",
        fake_play_file,
    )

    name, path, start, end = (
        playback.play_turn_context(
            source,
            turn,
            padding=2.0,
        )
    )

    assert name == "original-media"
    assert path == media_path.resolve()

    assert start == pytest.approx(598.031)
    assert end == pytest.approx(603.027)

    assert extraction["source"] == (
        media_path.resolve()
    )
    assert extraction["start"] == pytest.approx(
        598.031
    )
    assert extraction["end"] == pytest.approx(
        603.027
    )
    assert extraction["stream_index"] == 1
    assert extraction["channel_mode"] == "center"

    assert len(played) == 1


def test_is_playing_false_when_engine_idle(
    monkeypatch,
):
    monkeypatch.setattr(
        playback._playback_engine,
        "is_playing",
        lambda: False,
    )

    assert playback.is_playing() is False


def test_is_playing_true_when_engine_active(
    monkeypatch,
):
    monkeypatch.setattr(
        playback._playback_engine,
        "is_playing",
        lambda: True,
    )

    assert playback.is_playing() is True


class FakeOutputStream:
    def __init__(
        self,
        *,
        samplerate,
        channels,
        dtype,
        latency,
        callback,
        finished_callback,
    ):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.latency = latency
        self.callback = callback
        self.finished_callback = finished_callback

        self.closed = False
        self.active = False

        self.start_count = 0
        self.abort_count = 0
        self.close_count = 0

    def start(self):
        self.start_count += 1
        self.active = True

    def abort(self):
        self.abort_count += 1
        self.active = False

    def close(self):
        self.close_count += 1
        self.closed = True
        self.active = False


def test_playback_engine_reuses_matching_stream(
    monkeypatch,
):
    created = []

    def make_stream(**kwargs):
        stream = FakeOutputStream(**kwargs)
        created.append(stream)
        return stream

    monkeypatch.setattr(
        playback.sd,
        "OutputStream",
        make_stream,
    )

    engine = playback.PlaybackEngine()

    audio = np.zeros(
        480,
        dtype=np.float32,
    )

    engine.play(
        audio,
        48000,
    )

    first_stream = created[0]

    engine.play(
        audio,
        48000,
    )

    assert len(created) == 1
    assert first_stream.start_count == 1
    assert first_stream.abort_count == 0
    assert first_stream.close_count == 0

    engine.close()

    assert first_stream.close_count == 1


def test_playback_engine_recreates_stream_for_new_format(
    monkeypatch,
):
    created = []

    def make_stream(**kwargs):
        stream = FakeOutputStream(**kwargs)
        created.append(stream)
        return stream

    monkeypatch.setattr(
        playback.sd,
        "OutputStream",
        make_stream,
    )

    engine = playback.PlaybackEngine()

    mono = np.zeros(
        480,
        dtype=np.float32,
    )

    stereo = np.zeros(
        (480, 2),
        dtype=np.float32,
    )

    engine.play(
        mono,
        48000,
    )

    first_stream = created[0]

    engine.play(
        stereo,
        48000,
    )

    second_stream = created[1]

    assert len(created) == 2
    assert first_stream.abort_count == 0
    assert first_stream.close_count == 1

    assert second_stream.channels == 2
    assert second_stream.samplerate == 48000

    engine.close()


def test_playback_engine_uses_high_latency(
    monkeypatch,
):
    created = []

    def make_stream(**kwargs):
        stream = FakeOutputStream(**kwargs)
        created.append(stream)
        return stream

    monkeypatch.setattr(
        playback.sd,
        "OutputStream",
        make_stream,
    )

    engine = playback.PlaybackEngine()

    engine.play(
        np.zeros(
            480,
            dtype=np.float32,
        ),
        48000,
    )

    assert len(created) == 1
    assert created[0].latency == "high"
    assert created[0].dtype == "float32"

    engine.close()


def test_playback_engine_stop_keeps_stream_active(
    monkeypatch,
):
    created = []

    def make_stream(**kwargs):
        stream = FakeOutputStream(**kwargs)
        created.append(stream)
        return stream

    monkeypatch.setattr(
        playback.sd,
        "OutputStream",
        make_stream,
    )

    engine = playback.PlaybackEngine()

    engine.play(
        np.zeros(
            480,
            dtype=np.float32,
        ),
        48000,
    )

    stream = created[0]

    assert engine.is_playing() is True

    engine.stop()

    assert stream.abort_count == 0
    assert stream.active is True
    assert engine.is_playing() is False

    engine.close()


def test_playback_engine_callback_writes_audio():
    engine = playback.PlaybackEngine()

    engine._audio = np.array(
        [
            [0.1],
            [0.2],
            [0.3],
            [0.4],
        ],
        dtype=np.float32,
    )
    engine._position = 0
    engine._finished.clear()

    outdata = np.zeros(
        (2, 1),
        dtype=np.float32,
    )

    status = SimpleNamespace(
        output_underflow=False,
    )

    engine._callback(
        outdata,
        2,
        None,
        status,
    )

    assert outdata[:, 0] == pytest.approx(
        [
            0.1,
            0.2,
        ]
    )

    assert engine._position == 2


def test_playback_engine_counts_output_underflows():
    engine = playback.PlaybackEngine()

    engine._audio = np.zeros(
        (4, 1),
        dtype=np.float32,
    )
    engine._position = 0
    engine._finished.clear()

    outdata = np.zeros(
        (2, 1),
        dtype=np.float32,
    )

    status = SimpleNamespace(
        output_underflow=True,
    )

    engine._callback(
        outdata,
        2,
        None,
        status,
    )

    diagnostics = engine.diagnostics()

    assert diagnostics["output_underflows"] == 1


def test_playback_engine_finished_callback_marks_idle():
    engine = playback.PlaybackEngine()

    engine._finished.clear()

    assert engine.is_playing() is True

    engine._on_finished()

    assert engine.is_playing() is False


def test_playback_engine_closes_stream_without_holding_lock(
    monkeypatch,
):
    import threading

    engine = playback.PlaybackEngine()
    close_checked = threading.Event()

    class LockCheckingStream:
        closed = False
        active = False

        def close(self):
            acquired = []

            def try_lock():
                got_lock = engine._lock.acquire(
                    timeout=0.5,
                )

                acquired.append(got_lock)

                if got_lock:
                    engine._lock.release()

            thread = threading.Thread(
                target=try_lock,
            )
            thread.start()
            thread.join(timeout=1.0)

            assert acquired == [True]

            self.closed = True
            close_checked.set()

    engine._stream = LockCheckingStream()
    engine._sample_rate = 48000
    engine._channels = 1

    engine.close()

    assert close_checked.is_set()


def test_playback_engine_replaces_stream_without_holding_lock(
    monkeypatch,
):
    import threading

    engine = playback.PlaybackEngine()
    created = []

    class LockCheckingStream(FakeOutputStream):
        def close(self):
            acquired = []

            def try_lock():
                got_lock = engine._lock.acquire(
                    timeout=0.5,
                )

                acquired.append(got_lock)

                if got_lock:
                    engine._lock.release()

            thread = threading.Thread(
                target=try_lock,
            )
            thread.start()
            thread.join(timeout=1.0)

            assert acquired == [True]

            super().close()

    def make_stream(**kwargs):
        stream = LockCheckingStream(**kwargs)
        created.append(stream)
        return stream

    monkeypatch.setattr(
        playback.sd,
        "OutputStream",
        make_stream,
    )

    engine.play(
        np.zeros(
            480,
            dtype=np.float32,
        ),
        48000,
    )

    engine.play(
        np.zeros(
            (480, 2),
            dtype=np.float32,
        ),
        48000,
    )

    assert len(created) == 2

    engine.close()

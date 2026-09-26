from pathlib import Path

import pytest

import voice_dataset.media as media


def test_normalize_media_uses_global_stream_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    def fake_run(
        command,
        check,
    ):
        calls.append(
            {
                "command": command,
                "check": check,
            }
        )

    monkeypatch.setattr(
        media.subprocess,
        "run",
        fake_run,
    )

    source = Path("/media/arcane.mkv")
    destination = tmp_path / "normalized.wav"

    media.normalize_media(
        source,
        destination,
        stream_index=7,
        channel_mode="center",
        start=598.031,
        duration=4.996,
    )

    assert len(calls) == 1

    command = calls[0]["command"]

    assert calls[0]["check"] is True

    map_index = command.index("-map")

    assert command[map_index + 1] == "0:7"

    assert command == [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        "598.031",
        "-i",
        str(source),
        "-t",
        "4.996",
        "-map",
        "0:7",
        "-vn",
        "-af",
        "pan=mono|c0=FC",
        "-ar",
        str(media.WORK_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]


def test_extract_media_audio_region(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    def fake_normalize_media(
        source,
        destination,
        stream_index,
        channel_mode="mono",
        start=None,
        duration=None,
    ):
        calls.append(
            {
                "source": source,
                "destination": destination,
                "stream_index": stream_index,
                "channel_mode": channel_mode,
                "start": start,
                "duration": duration,
            }
        )

    monkeypatch.setattr(
        media,
        "normalize_media",
        fake_normalize_media,
    )

    source = Path("/media/arcane.mkv")
    destination = tmp_path / "context.wav"

    media.extract_media_audio_region(
        source,
        destination,
        598.031,
        603.027,
        stream_index=1,
        channel_mode="center",
    )

    assert calls == [
        {
            "source": source,
            "destination": destination,
            "stream_index": 1,
            "channel_mode": "center",
            "start": 598.031,
            "duration": pytest.approx(4.996),
        }
    ]


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (
            -1.0,
            1.0,
            "must not be negative",
        ),
        (
            1.0,
            1.0,
            "must be greater than start",
        ),
        (
            2.0,
            1.0,
            "must be greater than start",
        ),
    ],
)
def test_extract_media_audio_region_rejects_invalid_range(
    tmp_path: Path,
    start: float,
    end: float,
    message: str,
):
    with pytest.raises(
        ValueError,
        match=message,
    ):
        media.extract_media_audio_region(
            Path("/media/source.mkv"),
            tmp_path / "context.wav",
            start,
            end,
            stream_index=1,
        )


def make_audio_stream(
    *,
    index: int,
    language: str | None = None,
    is_default: bool = False,
) -> media.AudioStream:
    return media.AudioStream(
        index=index,
        codec="aac",
        sample_rate=48000,
        channels=2,
        channel_layout="stereo",
        language=language,
        title=None,
        is_default=is_default,
    )


def test_select_audio_stream_uses_explicit_position():
    streams = [
        make_audio_stream(
            index=1,
            language="deu",
            is_default=True,
        ),
        make_audio_stream(
            index=4,
            language="eng",
        ),
    ]

    position, stream = media.select_audio_stream(
        streams,
        audio_stream=1,
        language="deu",
    )

    assert position == 1
    assert stream.index == 4


def test_select_audio_stream_uses_unique_language():
    streams = [
        make_audio_stream(
            index=1,
            language="deu",
            is_default=True,
        ),
        make_audio_stream(
            index=4,
            language="eng",
        ),
    ]

    position, stream = media.select_audio_stream(
        streams,
        language="eng",
    )

    assert position == 1
    assert stream.index == 4


def test_select_audio_stream_rejects_ambiguous_language():
    streams = [
        make_audio_stream(
            index=1,
            language="eng",
        ),
        make_audio_stream(
            index=4,
            language="eng",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Multiple audio streams match language",
    ):
        media.select_audio_stream(
            streams,
            language="eng",
        )


def test_select_audio_stream_uses_unique_default():
    streams = [
        make_audio_stream(
            index=1,
            language="deu",
        ),
        make_audio_stream(
            index=4,
            language="eng",
            is_default=True,
        ),
    ]

    position, stream = media.select_audio_stream(
        streams,
    )

    assert position == 1
    assert stream.index == 4


def test_select_audio_stream_uses_only_stream():
    streams = [
        make_audio_stream(
            index=7,
        ),
    ]

    position, stream = media.select_audio_stream(
        streams,
    )

    assert position == 0
    assert stream.index == 7


def test_select_audio_stream_rejects_ambiguous_streams():
    streams = [
        make_audio_stream(index=1),
        make_audio_stream(index=4),
    ]

    with pytest.raises(
        ValueError,
        match="Audio stream selection is ambiguous",
    ):
        media.select_audio_stream(streams)


def test_select_audio_stream_rejects_invalid_explicit_position():
    streams = [
        make_audio_stream(index=7),
    ]

    with pytest.raises(
        ValueError,
        match="Audio stream 1 does not exist",
    ):
        media.select_audio_stream(
            streams,
            audio_stream=1,
        )


def test_probe_audio_streams_reads_default_disposition(
    monkeypatch: pytest.MonkeyPatch,
):
    class Result:
        stdout = """
        {
          "streams": [
            {
              "index": 2,
              "codec_name": "aac",
              "sample_rate": "48000",
              "channels": 6,
              "channel_layout": "5.1",
              "disposition": {
                "default": 1
              },
              "tags": {
                "language": "eng",
                "title": "English"
              }
            },
            {
              "index": 5,
              "codec_name": "aac",
              "sample_rate": "48000",
              "channels": 2,
              "channel_layout": "stereo",
              "disposition": {
                "default": 0
              },
              "tags": {
                "language": "deu"
              }
            }
          ]
        }
        """

    def fake_run(*args, **kwargs):
        return Result()

    monkeypatch.setattr(
        media.subprocess,
        "run",
        fake_run,
    )

    streams = media.probe_audio_streams(
        Path("/media/example.mkv")
    )

    assert len(streams) == 2

    assert streams[0].index == 2
    assert streams[0].language == "eng"
    assert streams[0].title == "English"
    assert streams[0].is_default is True

    assert streams[1].index == 5
    assert streams[1].language == "deu"
    assert streams[1].is_default is False

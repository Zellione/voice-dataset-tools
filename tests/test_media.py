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

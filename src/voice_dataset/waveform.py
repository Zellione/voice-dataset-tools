from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class WaveformEnvelope:
    start: float
    end: float
    times: tuple[float, ...]
    upper: tuple[float, ...]
    lower: tuple[float, ...]


def load_waveform_envelope(
    path: Path,
    *,
    start: float,
    end: float,
    points: int = 800,
) -> WaveformEnvelope:
    if points <= 0:
        raise ValueError(
            "points must be positive"
        )

    if start < 0:
        raise ValueError(
            "start must not be negative"
        )

    if end <= start:
        raise ValueError(
            "end must be after start"
        )

    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(path)

    with sf.SoundFile(path) as audio:
        sample_rate = int(audio.samplerate)
        total_frames = int(audio.frames)

        if sample_rate <= 0:
            raise ValueError(
                "Audio has invalid sample rate"
            )

        duration = (
            total_frames
            / sample_rate
        )

        actual_start = min(
            float(start),
            duration,
        )
        actual_end = min(
            float(end),
            duration,
        )

        if actual_end <= actual_start:
            raise ValueError(
                "Waveform range is outside audio"
            )

        start_frame = max(
            0,
            int(
                np.floor(
                    actual_start
                    * sample_rate
                )
            ),
        )

        end_frame = min(
            total_frames,
            int(
                np.ceil(
                    actual_end
                    * sample_rate
                )
            ),
        )

        audio.seek(start_frame)

        samples = audio.read(
            end_frame - start_frame,
            dtype="float32",
            always_2d=True,
        )

    if samples.size == 0:
        raise ValueError(
            "Waveform range contains no samples"
        )

    mono = np.mean(
        samples,
        axis=1,
        dtype=np.float32,
    )

    bucket_count = min(
        points,
        len(mono),
    )

    edges = np.linspace(
        0,
        len(mono),
        bucket_count + 1,
        dtype=int,
    )

    times: list[float] = []
    peaks: list[float] = []

    for index in range(bucket_count):
        left = int(edges[index])
        right = int(edges[index + 1])

        if right <= left:
            continue

        bucket = mono[left:right]

        peak = float(
            np.max(
                np.abs(bucket)
            )
        )

        midpoint = (
            left + right
        ) / 2.0

        time = (
            start_frame
            + midpoint
        ) / sample_rate

        times.append(time)
        peaks.append(peak)

    if not peaks:
        raise ValueError(
            "Waveform contains no buckets"
        )

    maximum = max(peaks)

    if maximum > 0:
        peaks = [
            peak / maximum
            for peak in peaks
        ]

    return WaveformEnvelope(
        start=actual_start,
        end=actual_end,
        times=tuple(times),
        upper=tuple(peaks),
        lower=tuple(
            -peak
            for peak in peaks
        ),
    )

from dataclasses import dataclass

import numpy as np
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps


VAD_SAMPLE_RATE = 16000


@dataclass
class SpeechRegion:
    start: float
    end: float


@dataclass
class ClipBounds:
    start: float
    end: float


def detect_speech_regions(path: str) -> list[SpeechRegion]:
    audio = decode_audio(
        path,
        sampling_rate=VAD_SAMPLE_RATE,
    )

    options = VadOptions(
        threshold=0.5,
        min_speech_duration_ms=100,
        min_silence_duration_ms=150,
        speech_pad_ms=100,
    )

    timestamps = get_speech_timestamps(
        np.asarray(audio),
        vad_options=options,
        sampling_rate=VAD_SAMPLE_RATE,
    )

    return [
        SpeechRegion(
            start=timestamp["start"] / VAD_SAMPLE_RATE,
            end=timestamp["end"] / VAD_SAMPLE_RATE,
        )
        for timestamp in timestamps
    ]


def reconcile_boundaries(
    segments,
    speech_regions: list[SpeechRegion],
    audio_duration: float,
    edge_padding: float = 0.10,
) -> list[ClipBounds]:
    """
    Convert ASR segment boundaries into safer clip boundaries.

    Between adjacent ASR segments, the midpoint of the ASR gap is used.
    At the beginning/end of the source, VAD is used to recover speech
    extending beyond the ASR timestamps.
    """

    if not segments:
        return []

    bounds: list[ClipBounds] = []

    for index, segment in enumerate(segments):
        # Beginning of clip
        if index == 0:
            overlapping = [
                region
                for region in speech_regions
                if region.end >= segment.start
                and region.start <= segment.end
            ]

            if overlapping:
                start = min(
                    segment.start,
                    min(region.start for region in overlapping),
                )
            else:
                start = segment.start

            start = max(0.0, start - edge_padding)

        else:
            previous = segments[index - 1]
            start = (previous.end + segment.start) / 2.0

        # End of clip
        if index + 1 < len(segments):
            following = segments[index + 1]
            end = (segment.end + following.start) / 2.0

        else:
            overlapping = [
                region
                for region in speech_regions
                if region.end >= segment.start
                and region.start <= audio_duration
            ]

            if overlapping:
                end = max(
                    segment.end,
                    max(region.end for region in overlapping),
                )
            else:
                end = segment.end

            end = min(audio_duration, end + edge_padding)

        bounds.append(
            ClipBounds(
                start=max(0.0, start),
                end=min(audio_duration, end),
            )
        )

    return bounds

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice_dataset.waveform import (
    load_waveform_envelope,
)


def test_load_waveform_envelope(
    tmp_path: Path,
):
    path = tmp_path / "audio.wav"

    sample_rate = 1000

    samples = np.zeros(
        sample_rate * 2,
        dtype=np.float32,
    )

    samples[500:1000] = 0.5
    samples[1000:1500] = -1.0

    sf.write(
        path,
        samples,
        sample_rate,
    )

    waveform = load_waveform_envelope(
        path,
        start=0.25,
        end=1.75,
        points=100,
    )

    assert waveform.start == pytest.approx(
        0.25
    )
    assert waveform.end == pytest.approx(
        1.75
    )

    assert len(waveform.times) <= 100
    assert len(waveform.times) > 0

    assert len(waveform.upper) == len(
        waveform.times
    )
    assert len(waveform.lower) == len(
        waveform.times
    )

    assert max(waveform.upper) == pytest.approx(
        1.0
    )
    assert min(waveform.lower) == pytest.approx(
        -1.0
    )

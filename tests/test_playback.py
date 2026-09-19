from pathlib import Path

import pytest

from voice_dataset.playback import (
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

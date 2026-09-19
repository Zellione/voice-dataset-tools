import pytest

from voice_dataset.boundary_evidence import (
    calculate_boundary_evidence,
    select_boundary_representation,
)


def test_near_source_start():
    result = calculate_boundary_evidence(
        representation_name="center",
        representation={
            "duration": 60.0,
        },
        start=0.03096875,
        end=1.027,
    )

    assert result == {
        "representation": "center",
        "margin": 0.1,
        "near_source_start": True,
        "near_source_end": False,
    }


def test_near_source_end():
    result = calculate_boundary_evidence(
        representation_name="center",
        representation={
            "duration": 60.0,
        },
        start=58.0,
        end=59.95,
    )

    assert result == {
        "representation": "center",
        "margin": 0.1,
        "near_source_start": False,
        "near_source_end": True,
    }


def test_not_near_source_boundary():
    result = calculate_boundary_evidence(
        representation_name="center",
        representation={
            "duration": 60.0,
        },
        start=10.0,
        end=12.0,
    )

    assert result == {
        "representation": "center",
        "margin": 0.1,
        "near_source_start": False,
        "near_source_end": False,
    }


def test_missing_duration_is_rejected():
    with pytest.raises(
        ValueError,
        match="no known duration",
    ):
        calculate_boundary_evidence(
            representation_name="center",
            representation={},
            start=1.0,
            end=2.0,
        )


@pytest.mark.parametrize(
    (
        "start",
        "end",
        "message",
    ),
    [
        (
            -0.1,
            1.0,
            "Turn start must not be negative",
        ),
        (
            2.0,
            2.0,
            "Turn end must be after turn start",
        ),
        (
            2.0,
            1.0,
            "Turn end must be after turn start",
        ),
        (
            59.0,
            60.1,
            "Turn ends after source representation",
        ),
    ],
)
def test_invalid_turn_range(
    start: float,
    end: float,
    message: str,
):
    with pytest.raises(
        ValueError,
        match=message,
    ):
        calculate_boundary_evidence(
            representation_name="center",
            representation={
                "duration": 60.0,
            },
            start=start,
            end=end,
        )


def test_select_boundary_representation():
    source = {
        "representations": {
            "speech": {
                "duration": 60.0,
                "purposes": [
                    "asr",
                    "review",
                ],
            },
            "center": {
                "duration": 60.0,
                "purposes": [
                    "speaker_embedding",
                    "boundary_analysis",
                    "context",
                ],
            },
        },
    }

    name, representation = (
        select_boundary_representation(source)
    )

    assert name == "center"
    assert representation == source[
        "representations"
    ]["center"]


def test_select_boundary_representation_rejects_missing():
    with pytest.raises(
        ValueError,
        match="no boundary_analysis representation",
    ):
        select_boundary_representation(
            {
                "representations": {
                    "speech": {
                        "purposes": [
                            "asr",
                        ],
                    },
                },
            }
        )


def test_select_boundary_representation_rejects_multiple():
    with pytest.raises(
        ValueError,
        match=(
            "multiple boundary_analysis "
            "representations"
        ),
    ):
        select_boundary_representation(
            {
                "representations": {
                    "center": {
                        "purposes": [
                            "boundary_analysis",
                        ],
                    },
                    "raw": {
                        "purposes": [
                            "boundary_analysis",
                        ],
                    },
                },
            }
        )

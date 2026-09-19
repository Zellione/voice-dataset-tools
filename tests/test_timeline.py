import pytest

from voice_dataset.timeline import (
    media_time_to_representation_time,
    representation_media_start,
    representation_time_to_media_time,
)


def test_representation_media_start():
    representation = {
        "media_start": 600.0,
    }

    assert (
        representation_media_start(
            representation
        )
        == 600.0
    )


def test_representation_media_start_unknown():
    with pytest.raises(
        ValueError,
        match="no known media timeline mapping",
    ):
        representation_media_start({})


def test_representation_time_to_media_time():
    representation = {
        "media_start": 600.0,
    }

    result = (
        representation_time_to_media_time(
            representation,
            0.031,
        )
    )

    assert result == pytest.approx(
        600.031
    )


def test_media_time_to_representation_time():
    representation = {
        "media_start": 600.0,
    }

    result = (
        media_time_to_representation_time(
            representation,
            600.031,
        )
    )

    assert result == pytest.approx(
        0.031
    )


def test_media_time_before_representation():
    representation = {
        "media_start": 600.0,
    }

    with pytest.raises(
        ValueError,
        match=(
            "before the start of "
            "the representation"
        ),
    ):
        media_time_to_representation_time(
            representation,
            599.0,
        )


@pytest.mark.parametrize(
    ("function", "time"),
    [
        (
            representation_time_to_media_time,
            -1.0,
        ),
        (
            media_time_to_representation_time,
            -1.0,
        ),
    ],
)
def test_negative_time_is_rejected(
    function,
    time,
):
    representation = {
        "media_start": 600.0,
    }

    with pytest.raises(
        ValueError,
        match="must not be negative",
    ):
        function(
            representation,
            time,
        )


def test_negative_media_start_is_rejected():
    representation = {
        "media_start": -1.0,
    }

    with pytest.raises(
        ValueError,
        match="media_start must not be negative",
    ):
        representation_media_start(
            representation
        )

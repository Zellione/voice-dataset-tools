from pathlib import Path
import subprocess

import pytest

from voice_dataset import workers


def test_project_root_contains_project() -> None:
    root = workers.project_root()

    assert (
        root
        / "pyproject.toml"
    ).is_file()

    assert (
        root
        / "adapters"
        / "diarize.py"
    ).is_file()


def test_community_worker_paths() -> None:
    item = workers.worker(
        "community-1"
    )

    root = workers.project_root()

    assert item.python == (
        root
        / "tools"
        / "pyannote"
        / ".venv"
        / "bin"
        / "python"
    )

    assert item.adapter == (
        root
        / "adapters"
        / "diarize.py"
    )


def test_bandit_worker_paths() -> None:
    item = workers.worker(
        "bandit"
    )

    root = workers.project_root()

    assert item.python == (
        root
        / "tools"
        / "bandit"
        / ".venv"
        / "bin"
        / "python"
    )

    assert item.adapter == (
        root
        / "adapters"
        / "bandit.py"
    )


def test_unknown_worker_is_rejected() -> None:
    with pytest.raises(
        workers.WorkerError,
        match="Unknown worker",
    ):
        workers.worker(
            "does-not-exist"
        )


def test_validate_worker_rejects_missing_python(
    tmp_path: Path,
) -> None:
    item = workers.Worker(
        name="test",
        python=tmp_path / "missing-python",
        adapter=tmp_path / "adapter.py",
    )

    with pytest.raises(
        workers.WorkerError,
        match="Run ./scripts/bootstrap first",
    ):
        workers.validate_worker(item)


def test_run_worker_builds_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = tmp_path / "python"
    adapter = tmp_path / "adapter.py"

    python.touch()
    adapter.touch()

    item = workers.Worker(
        name="test",
        python=python,
        adapter=adapter,
    )

    observed: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        check: bool,
    ) -> None:
        assert check is True
        observed.append(command)

    monkeypatch.setattr(
        workers.subprocess,
        "run",
        fake_run,
    )

    workers.run_worker(
        item,
        [
            Path("/tmp/input.wav"),
            Path("/tmp/output.json"),
            "--min-speakers",
            "2",
        ],
    )

    assert observed == [
        [
            str(python),
            str(adapter),
            "/tmp/input.wav",
            "/tmp/output.json",
            "--min-speakers",
            "2",
        ]
    ]


def test_run_worker_wraps_process_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = tmp_path / "python"
    adapter = tmp_path / "adapter.py"

    python.touch()
    adapter.touch()

    item = workers.Worker(
        name="test",
        python=python,
        adapter=adapter,
    )

    def fake_run(
        command: list[str],
        *,
        check: bool,
    ) -> None:
        raise subprocess.CalledProcessError(
            7,
            command,
        )

    monkeypatch.setattr(
        workers.subprocess,
        "run",
        fake_run,
    )

    with pytest.raises(
        workers.WorkerError,
        match=(
            "test worker failed "
            "with exit code 7"
        ),
    ):
        workers.run_worker(
            item,
            [],
        )


def test_run_worker_can_capture_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = tmp_path / "python"
    adapter = tmp_path / "adapter.py"

    python.touch()
    adapter.touch()

    item = workers.Worker(
        name="test",
        python=python,
        adapter=adapter,
    )

    observed = []

    def fake_run(
        command,
        **kwargs,
    ):
        observed.append(
            (
                command,
                kwargs,
            )
        )

        return subprocess.CompletedProcess(
            command,
            0,
            stdout="worker output",
            stderr="worker warning",
        )

    monkeypatch.setattr(
        workers.subprocess,
        "run",
        fake_run,
    )

    workers.run_worker(
        item,
        ["argument"],
        capture_output=True,
    )

    assert len(observed) == 1

    command, kwargs = observed[0]

    assert command == [
        str(python),
        str(adapter),
        "argument",
    ]

    assert kwargs == {
        "check": True,
        "capture_output": True,
        "text": True,
    }

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


def test_run_workers_parallel_starts_all_before_waiting(
    monkeypatch,
    tmp_path,
):
    import voice_dataset.workers as workers

    events = []

    class FakeProcess:
        def __init__(
            self,
            command,
            **kwargs,
        ):
            self.command = command
            self.returncode = None
            events.append(
                ("start", command[-1])
            )

        def communicate(self):
            events.append(
                (
                    "wait",
                    self.command[-1],
                )
            )
            self.returncode = 0
            return (None, None)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self):
            return self.returncode

    monkeypatch.setattr(
        workers.subprocess,
        "Popen",
        FakeProcess,
    )

    fake_python = tmp_path / "python"
    fake_adapter = tmp_path / "adapter.py"

    fake_python.touch()
    fake_adapter.touch()

    first = workers.Worker(
        name="first",
        python=fake_python,
        adapter=fake_adapter,
    )

    second = workers.Worker(
        name="second",
        python=fake_python,
        adapter=fake_adapter,
    )

    workers.run_workers_parallel(
        [
            workers.WorkerInvocation(
                worker=first,
                arguments=["one"],
            ),
            workers.WorkerInvocation(
                worker=second,
                arguments=["two"],
            ),
        ]
    )

    assert events == [
        ("start", "one"),
        ("start", "two"),
        ("wait", "one"),
        ("wait", "two"),
    ]


def test_run_workers_parallel_raises_on_failure(
    monkeypatch,
    tmp_path,
):
    import voice_dataset.workers as workers

    class FakeProcess:
        next_returncodes = iter([0, 7])

        def __init__(
            self,
            command,
            **kwargs,
        ):
            self.returncode = None
            self.final_returncode = next(
                self.next_returncodes
            )

        def communicate(self):
            self.returncode = (
                self.final_returncode
            )
            return (None, None)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self):
            return self.returncode

    monkeypatch.setattr(
        workers.subprocess,
        "Popen",
        FakeProcess,
    )

    fake_python = tmp_path / "python"
    fake_adapter = tmp_path / "adapter.py"

    fake_python.touch()
    fake_adapter.touch()

    first = workers.Worker(
        name="first",
        python=fake_python,
        adapter=fake_adapter,
    )

    second = workers.Worker(
        name="second",
        python=fake_python,
        adapter=fake_adapter,
    )

    with pytest.raises(
        workers.WorkerError,
        match=(
            "second worker failed "
            "with exit code 7"
        ),
    ):
        workers.run_workers_parallel(
            [
                workers.WorkerInvocation(
                    worker=first,
                    arguments=["one"],
                ),
                workers.WorkerInvocation(
                    worker=second,
                    arguments=["two"],
                ),
            ]
        )

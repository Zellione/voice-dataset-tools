from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Sequence


class WorkerError(RuntimeError):
    pass


@dataclass(frozen=True)
class Worker:
    name: str
    python: Path
    adapter: Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def worker(
    name: str,
) -> Worker:
    root = project_root()

    workers = {
        "community-1": Worker(
            name="community-1",
            python=(
                root
                / "tools"
                / "pyannote"
                / ".venv"
                / "bin"
                / "python"
            ),
            adapter=(
                root
                / "adapters"
                / "diarize.py"
            ),
        ),
        "ecapa": Worker(
            name="ecapa",
            python=(
                root
                / "tools"
                / "ecapa"
                / ".venv"
                / "bin"
                / "python"
            ),
            adapter=(
                root
                / "adapters"
                / "ecapa.py"
            ),
        ),
        "wespeaker": Worker(
            name="wespeaker",
            python=(
                root
                / "tools"
                / "pyannote"
                / ".venv"
                / "bin"
                / "python"
            ),
            adapter=(
                root
                / "adapters"
                / "wespeaker.py"
            ),
        ),
        "bandit": Worker(
            name="bandit",
            python=(
                root
                / "tools"
                / "bandit"
                / ".venv"
                / "bin"
                / "python"
            ),
            adapter=(
                root
                / "adapters"
                / "bandit.py"
            ),
        ),
        "qwen3-asr": Worker(
            name="qwen3-asr",
            python=(
                root
                / "tools"
                / "qwen3-aligner"
                / ".venv"
                / "bin"
                / "python"
            ),
            adapter=(
                root
                / "adapters"
                / "qwen3_asr.py"
            ),
        ),
        "sat": Worker(
            name="sat",
            python=(
                root
                / "tools"
                / "sat"
                / ".venv"
                / "bin"
                / "python"
            ),
            adapter=(
                root
                / "adapters"
                / "sat.py"
            ),
        ),
    }

    try:
        return workers[name]
    except KeyError as exc:
        raise WorkerError(
            f"Unknown worker: {name}"
        ) from exc


def validate_worker(
    worker: Worker,
) -> None:
    if not worker.python.is_file():
        raise WorkerError(
            f"{worker.name} Python environment "
            f"does not exist: {worker.python}\n"
            "Run ./scripts/bootstrap first."
        )

    if not worker.adapter.is_file():
        raise WorkerError(
            f"{worker.name} adapter does not exist: "
            f"{worker.adapter}"
        )


@dataclass(frozen=True)
class WorkerInvocation:
    worker: Worker
    arguments: Sequence[str | Path]


def _worker_command(
    invocation: WorkerInvocation,
) -> list[str]:
    validate_worker(
        invocation.worker
    )

    return [
        str(invocation.worker.python),
        str(invocation.worker.adapter),
        *(
            str(argument)
            for argument
            in invocation.arguments
        ),
    ]


def run_workers_parallel(
    invocations: Sequence[WorkerInvocation],
    *,
    capture_output: bool = False,
) -> None:
    if not invocations:
        return

    processes: list[
        tuple[
            Worker,
            subprocess.Popen,
        ]
    ] = []

    try:
        for invocation in invocations:
            command = _worker_command(
                invocation
            )

            kwargs = {}

            if capture_output:
                kwargs.update({
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.PIPE,
                    "text": True,
                })

            process = subprocess.Popen(
                command,
                **kwargs,
            )

            processes.append(
                (
                    invocation.worker,
                    process,
                )
            )

        failures = []

        for worker_item, process in processes:
            stdout, stderr = (
                process.communicate()
            )

            if process.returncode != 0:
                failures.append(
                    (
                        worker_item,
                        process.returncode,
                        stdout,
                        stderr,
                    )
                )

        if failures:
            worker_item, returncode, _, _ = (
                failures[0]
            )

            raise WorkerError(
                f"{worker_item.name} worker failed "
                f"with exit code {returncode}"
            )

    except Exception:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()

        for _, process in processes:
            if process.poll() is None:
                process.wait()

        raise


def run_worker(
    worker: Worker,
    arguments: Sequence[str | Path],
    *,
    capture_output: bool = False,
) -> None:
    validate_worker(worker)

    command = [
        str(worker.python),
        str(worker.adapter),
        *(
            str(argument)
            for argument in arguments
        ),
    ]

    run_kwargs = {
        "check": True,
    }

    if capture_output:
        run_kwargs.update({
            "capture_output": True,
            "text": True,
        })

    try:
        subprocess.run(
            command,
            **run_kwargs,
        )
    except subprocess.CalledProcessError as exc:
        raise WorkerError(
            f"{worker.name} worker failed "
            f"with exit code {exc.returncode}"
        ) from exc

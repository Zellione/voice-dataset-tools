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


def run_worker(
    worker: Worker,
    arguments: Sequence[str | Path],
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

    try:
        subprocess.run(
            command,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise WorkerError(
            f"{worker.name} worker failed "
            f"with exit code {exc.returncode}"
        ) from exc

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from .schema import TurnRecord, VoiceProfile


class JsonlStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []

        records = []

        with self.path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid JSON in {self.path} "
                        f"at line {line_number}"
                    ) from exc

                records.append(record)

        return records

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )
            file.write("\n")

            file.flush()
            os.fsync(file.fileno())

    def replace(
        self,
        records: list[dict[str, Any]],
    ) -> None:
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
            text=True,
        )

        temp_path = Path(temp_name)

        try:
            with os.fdopen(
                fd,
                "w",
                encoding="utf-8",
            ) as file:
                for record in records:
                    file.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                        )
                    )
                    file.write("\n")

                file.flush()
                os.fsync(file.fileno())

            os.replace(
                temp_path,
                self.path,
            )

        except Exception:
            temp_path.unlink(
                missing_ok=True
            )
            raise

    def update(
        self,
        record_id: str,
        update_fn: Callable[
            [dict[str, Any]],
            dict[str, Any],
        ],
    ) -> dict[str, Any]:
        records = self.load()

        matches = [
            index
            for index, record in enumerate(records)
            if record.get("id") == record_id
        ]

        if not matches:
            raise KeyError(
                f"Record does not exist: {record_id}"
            )

        if len(matches) > 1:
            raise RuntimeError(
                f"Duplicate record id "
                f"in {self.path}: {record_id}"
            )

        index = matches[0]

        updated = update_fn(
            records[index].copy()
        )

        if updated.get("id") != record_id:
            raise ValueError(
                "Record update must not change id"
            )

        records[index] = updated

        self.replace(records)

        return updated


class DatasetStorage:
    def __init__(self, root: Path):
        self.root = root

        self.turns = JsonlStore(
            root / "turns.jsonl"
        )

        self.voices = JsonlStore(
            root / "voices.jsonl"
        )

    def next_turn_id(self) -> str:
        return self._next_id(
            self.turns.load(),
            prefix="turn",
            width=6,
        )

    def next_voice_id(self) -> str:
        return self._next_id(
            self.voices.load(),
            prefix="voice",
            width=3,
        )

    def add_turn(
        self,
        turn: TurnRecord,
    ) -> None:
        self._ensure_unique(
            self.turns,
            turn.id,
        )

        self.turns.append(
            turn.to_dict()
        )

    def add_voice(
        self,
        voice: VoiceProfile,
    ) -> None:
        self._ensure_unique(
            self.voices,
            voice.id,
        )

        self.voices.append(
            voice.to_dict()
        )

    def get_turn(
        self,
        turn_id: str,
    ) -> dict[str, Any] | None:
        return self._get(
            self.turns,
            turn_id,
        )

    def get_voice(
        self,
        voice_id: str,
    ) -> dict[str, Any] | None:
        return self._get(
            self.voices,
            voice_id,
        )

    @staticmethod
    def _get(
        store: JsonlStore,
        record_id: str,
    ) -> dict[str, Any] | None:
        matches = [
            record
            for record in store.load()
            if record.get("id") == record_id
        ]

        if not matches:
            return None

        if len(matches) > 1:
            raise RuntimeError(
                f"Duplicate record id "
                f"in {store.path}: {record_id}"
            )

        return matches[0]

    @staticmethod
    def _ensure_unique(
        store: JsonlStore,
        record_id: str,
    ) -> None:
        if any(
            record.get("id") == record_id
            for record in store.load()
        ):
            raise ValueError(
                f"Record already exists: {record_id}"
            )

    @staticmethod
    def _next_id(
        records: list[dict[str, Any]],
        prefix: str,
        width: int,
    ) -> str:
        highest = 0

        for record in records:
            record_id = record.get("id")

            if not isinstance(
                record_id,
                str,
            ):
                continue

            expected_prefix = (
                f"{prefix}_"
            )

            if not record_id.startswith(
                expected_prefix
            ):
                continue

            suffix = record_id[
                len(expected_prefix):
            ]

            if not suffix.isdigit():
                continue

            highest = max(
                highest,
                int(suffix),
            )

        return (
            f"{prefix}_"
            f"{highest + 1:0{width}d}"
        )

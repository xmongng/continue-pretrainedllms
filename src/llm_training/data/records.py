"""Read only the JSON fields needed by training without Arrow schema inference."""

from __future__ import annotations

import gzip
import json
from pathlib import Path


def _open_text(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_json_records(filename: str | Path, json_field: str | None = None):
    """Yield JSON objects while preserving metadata values without type inference."""

    path = Path(filename)
    name = path.name.lower()
    if name.endswith((".jsonl", ".jsonl.gz")):
        with _open_text(path) as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON at {path}:{line_number}: {error.msg}"
                    ) from error
                if not isinstance(row, dict):
                    raise TypeError(f"Expected a JSON object at {path}:{line_number}")
                yield row
        return

    with _open_text(path) as stream:
        payload = json.load(stream)
    if json_field is not None:
        if not isinstance(payload, dict) or json_field not in payload:
            raise ValueError(f"Missing JSON field {json_field!r} in {path}")
        payload = payload[json_field]
    rows = payload if isinstance(payload, list) else [payload]
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise TypeError(f"Expected a JSON object at {path}, item {index}")
        yield row


def training_text(row: dict, columns: list[str], location: str) -> tuple[str, str]:
    """Return the first populated configured text field and its column name."""

    for column in columns:
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return value, column
    present = {column: type(row[column]).__name__ for column in columns if column in row}
    raise ValueError(
        f"No nonempty training text in {location}; expected one of {columns!r}, "
        f"present types: {present!r}"
    )

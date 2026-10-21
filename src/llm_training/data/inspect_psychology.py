"""Stream and validate the confirmed psychology JSONL.GZ corpus."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from llm_training.config import load_config
from llm_training.data.dataset import discover_data_files


def inspect_dataset(files: list[str], text_column: str, source_column: str) -> dict:
    """Check every row without materializing the corpus in memory."""
    if not files:
        raise ValueError("No dataset shards were found")

    counts = Counter()
    total = 0
    chars = 0
    shortest = None
    longest = 0
    for filename in files:
        path = Path(filename)
        if not path.name.endswith(".jsonl.gz"):
            raise ValueError(f"Expected a .jsonl.gz shard: {path}")
        shard_rows = 0
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a JSON object at {path}:{line_number}")
                text = row.get(text_column)
                source = row.get(source_column)
                identifier = row.get("id")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Empty/non-string {text_column!r} at {path}:{line_number}")
                if not isinstance(source, str) or not source.strip():
                    raise ValueError(f"Empty/non-string {source_column!r} at {path}:{line_number}")
                if not isinstance(identifier, str) or not identifier.strip():
                    raise ValueError(f"Empty/non-string 'id' at {path}:{line_number}")
                declared_chars = row.get("n_chars")
                if (
                    isinstance(declared_chars, bool)
                    or not isinstance(declared_chars, int)
                    or declared_chars != len(text)
                ):
                    raise ValueError(f"n_chars mismatch at {path}:{line_number}")
                shard_rows += 1
                total += 1
                chars += len(text)
                shortest = len(text) if shortest is None else min(shortest, len(text))
                longest = max(longest, len(text))
                counts[source] += 1
        if shard_rows == 0:
            raise ValueError(f"Empty dataset shard: {path}")

    return {
        "shards": len(files),
        "documents": total,
        "text_column": text_column,
        "source_column": source_column,
        "characters": chars,
        "shortest_document_chars": shortest,
        "longest_document_chars": longest,
        "documents_by_source": dict(sorted(counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen3_1_7b_base_e01.json")
    args = parser.parse_args()
    data = load_config(args.config)["data"]
    files = discover_data_files(data["source"], data["cache_dir"])
    report = inspect_dataset(files, data["text_column"], data["source_column"])
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

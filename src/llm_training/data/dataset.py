"""Discover corpus shards, extract ZIP archives safely, and collate token batches."""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase

SUPPORTED_DATA_FILES = (".json", ".jsonl", ".json.gz", ".jsonl.gz")


def _is_supported_data_file(path: str | Path) -> bool:
    name = str(path).lower()
    return any(name.endswith(suffix) for suffix in SUPPORTED_DATA_FILES)


def _safe_archive_target(root: Path, member_name: str) -> Path:
    member = Path(member_name)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError(f"Unsafe path inside ZIP archive: {member_name}")

    target = (root / member).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Unsafe path inside ZIP archive: {member_name}") from error
    return target


def extract_json_archive(archive_path: Path, cache_dir: Path) -> list[str]:
    """Extract supported JSON files from a ZIP into a stable cache directory."""

    if not zipfile.is_zipfile(archive_path):
        raise ValueError(f"Dataset archive is not a valid ZIP file: {archive_path}")

    digest = hashlib.sha256()
    with archive_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    cache_key = digest.hexdigest()
    destination = cache_dir / f"{archive_path.stem}-{cache_key}"
    marker = destination / ".complete"

    if not marker.exists():
        destination.mkdir(parents=True, exist_ok=True)

        # Extract only dataset files and reject ZIP path traversal.
        with zipfile.ZipFile(archive_path) as archive:
            members = [
                member
                for member in archive.infolist()
                if not member.is_dir() and _is_supported_data_file(member.filename)
            ]
            if not members:
                raise ValueError("The ZIP archive contains no JSON or JSONL files")

            targets = [_safe_archive_target(destination, member.filename) for member in members]
            normalized = [str(target.relative_to(destination)).casefold() for target in targets]
            if len(normalized) != len(set(normalized)):
                raise ValueError("The ZIP archive contains duplicate dataset paths")

            for member, target in zip(members, targets):
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)

        marker.write_text("complete\n", encoding="utf-8")

    files = sorted(
        str(path)
        for path in destination.rglob("*")
        if path.is_file() and _is_supported_data_file(path)
    )
    if not files:
        raise ValueError(f"No dataset files found after extracting {archive_path}")
    return files


def discover_data_files(source: str | Path, cache_dir: str | Path) -> list[str]:
    """Find JSON files in a single file, directory, or ZIP archive."""

    source_path = Path(source).expanduser().resolve()
    cache_path = Path(cache_dir).expanduser().resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"Dataset source does not exist: {source_path}")
    if source_path.is_dir():
        files = sorted(
            str(path)
            for path in source_path.rglob("*")
            if path.is_file() and _is_supported_data_file(path)
        )
    elif source_path.suffix.lower() == ".zip":
        files = extract_json_archive(source_path, cache_path)
    elif _is_supported_data_file(source_path):
        files = [str(source_path)]
    else:
        raise ValueError("Dataset source must be JSON, JSONL, a directory, or a ZIP archive")

    if not files:
        raise ValueError(f"No JSON or JSONL files found in {source_path}")
    return files


class CausalLMCollator:
    """Pad batches and mask only padding positions in next-token labels."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        labels[:, 0] = -100
        batch["labels"] = labels
        return batch

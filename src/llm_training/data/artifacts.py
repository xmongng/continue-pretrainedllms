"""Versioned, reusable dataset artifacts for controlled CPT experiments."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk
from filelock import FileLock

from llm_training.data.dataset import discover_data_files
from llm_training.data.records import iter_json_records


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def packed_rows(documents, length):
    # Carry one context token across blocks: every subsequent token is a target once.
    buffer = []
    for row in documents:
        tokens = row["input_ids"]
        offset = 0
        while offset < len(tokens):
            take = min(length - len(buffer), len(tokens) - offset)
            buffer.extend(tokens[offset : offset + take])
            offset += take
            if len(buffer) == length:
                yield {"input_ids": buffer, "attention_mask": [1] * len(buffer)}
                buffer = buffer[-1:]
    if len(buffer) > 1:
        yield {"input_ids": buffer, "attention_mask": [1] * len(buffer)}


def length_summary(histogram):
    count = sum(histogram.values())

    def quantile(fraction):
        cumulative = 0
        for length, frequency in sorted(histogram.items()):
            cumulative += frequency
            if cumulative >= max(1, fraction * count):
                return length

    return {
        "min": min(histogram),
        "max": max(histogram),
        "mean": sum(k * v for k, v in histogram.items()) / count,
        "median": quantile(0.5),
        "p95": quantile(0.95),
    }


def prepare_artifacts(data, tokenizer):
    """Hash sources, freeze group splits, and persist packed Arrow datasets once."""
    if tokenizer.eos_token_id is None:
        raise ValueError("An EOS token is required")
    cache = Path(data["cache_dir"])
    files = discover_data_files(data["source"], cache)
    if data.get("max_shards"):
        files = files[: data["max_shards"]]
    source = Path(data["source"])
    root = source if source.is_dir() else Path(files[0]).parent
    inventory = [
        {
            "path": str(Path(f).relative_to(root))
            if Path(f).is_relative_to(root)
            else Path(f).name,
            "sha256": file_hash(f),
        }
        for f in files
    ]
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Audited tokenizer fingerprinting currently requires a fast tokenizer")
    tokenizer_state = json.loads(tokenizer.backend_tokenizer.to_str())
    tokenizer_state.pop("padding", None)
    tokenizer_state.pop("truncation", None)
    tokenizer_hash = hashlib.sha256(
        json.dumps(tokenizer_state, sort_keys=True).encode()
    ).hexdigest()
    recipe = {
        "format_version": 1,
        "files": inventory,
        "tokenizer_sha256": tokenizer_hash,
        "eos_token_id": tokenizer.eos_token_id,
        "settings": {
            k: data.get(k)
            for k in (
                "text_column",
                "json_field",
                "group_column",
                "source_column",
                "validation_fraction",
                "test_fraction",
                "seed",
                "max_length",
                "max_shards",
                "max_documents",
            )
        },
    }
    identity = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    parent = cache / "prepared"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / identity
    with FileLock(str(parent / (identity + ".lock"))):
        if not destination.exists():
            with tempfile.TemporaryDirectory(prefix="building-", dir=parent) as temporary:
                work = Path(temporary)
                build_artifacts(work, files, inventory, data, tokenizer, recipe, identity)
                work.rename(destination)
    packed = load_from_disk(str(destination / "dataset"))
    stats = json.loads((destination / "dataset_stats.json").read_text())
    return packed, {"id": identity, "path": str(destination), "statistics": stats}


def build_artifacts(work, files, inventory, data, tokenizer, recipe, identity):
    # Preserve source-file identity without re-cleaning or stripping code indentation.
    selected_files = []
    group_counts = Counter()
    text_column = data["text_column"]
    group_column = data.get("group_column")
    source_column = data.get("source_column")
    remaining = data.get("max_documents")
    for number, file in enumerate(files):
        row_count = 0
        rows = iter_json_records(file, data.get("json_field"))
        if remaining is not None:
            rows = islice(rows, remaining)
        for row_number, row in enumerate(rows, start=1):
            for column in filter(None, (text_column, group_column, source_column)):
                if column not in row:
                    raise ValueError(f"Missing column {column!r} in {file}, record {row_number}")
            text = row[text_column]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty/non-string text in {file}, record {row_number}")
            key = (
                str(row[group_column])
                if group_column
                else hashlib.sha256(text.encode()).hexdigest()
            )
            if group_column and (row[group_column] is None or not key.strip()):
                raise ValueError(f"Empty group identifier in {file}, record {row_number}")
            group_counts[key] += 1
            row_count += 1
        selected_files.append((number, file, row_count))
        if remaining is not None:
            remaining -= row_count
        if remaining == 0:
            break
    keys = sorted(
        group_counts, key=lambda k: hashlib.sha256(f"{data.get('seed', 42)}:{k}".encode()).digest()
    )
    if len(keys) < 3:
        raise ValueError("At least 3 distinct documents/groups required for train/val/test")
    # Ratios apply to groups; actual document/token ratios are reported, never hidden.
    n_val = max(1, round(len(keys) * data["validation_fraction"]))
    n_test = max(1, round(len(keys) * data["test_fraction"]))
    if n_val + n_test >= len(keys):
        raise ValueError("Not enough groups for the requested split fractions")
    assignment = {
        k: ("validation" if i < n_val else "test" if i < n_val + n_test else "train")
        for i, k in enumerate(keys)
    }
    document_counts = Counter()
    token_counts = Counter()
    source_counts = defaultdict(Counter)
    lengths = defaultdict(Counter)
    manifest_path = work / "split_manifest.jsonl"

    def token_rows():
        with manifest_path.open("w", encoding="utf-8") as manifest:
            for file_number, file, row_count in selected_files:
                rows = islice(iter_json_records(file, data.get("json_field")), row_count)
                for row_number, row in enumerate(rows):
                    text = row[text_column]
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    key = str(row[group_column]) if group_column else digest
                    split = assignment[key]
                    origin = (
                        str(row[source_column]) if source_column else inventory[file_number]["path"]
                    )
                    ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)[
                        "input_ids"
                    ]
                    ids.append(tokenizer.eos_token_id)
                    document_counts[split] += 1
                    token_counts[split] += len(ids)
                    lengths[split][len(ids)] += 1
                    source_counts[split][origin] += len(ids)
                    manifest.write(
                        json.dumps(
                            {
                                "file": file_number,
                                "row": row_number,
                                "text_sha256": digest,
                                "group": key,
                                "source": origin,
                                "split": split,
                                "tokens_with_eos": len(ids),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    yield {"input_ids": ids, "split": split}

    tokenized = Dataset.from_generator(token_rows, cache_dir=str(work / "intermediate"))
    packed = DatasetDict()
    stats = {}
    for split in ("train", "validation", "test"):
        selected = tokenized.filter(
            lambda row, name=split: row["split"] == name, load_from_cache_file=False
        )
        packed[split] = Dataset.from_generator(
            packed_rows,
            gen_kwargs={"documents": selected, "length": data["max_length"]},
            cache_dir=str(work / "intermediate"),
        )
        target_tokens = sum(len(row["input_ids"]) - 1 for row in packed[split])
        if target_tokens != token_counts[split] - 1:
            raise AssertionError("Packing lost or duplicated target tokens")
        stats[split] = {
            "documents": document_counts[split],
            "groups": sum(v == split for v in assignment.values()),
            "tokens_with_eos": token_counts[split],
            "target_tokens": target_tokens,
            "sequences": len(packed[split]),
            "length_with_eos": length_summary(lengths[split]),
            "tokens_by_source": dict(source_counts[split]),
            "source_token_fraction": {
                k: v / token_counts[split] for k, v in source_counts[split].items()
            },
            "dropped_tokens": 0,
        }
    packed.save_to_disk(str(work / "dataset"))
    write_json(work / "dataset_stats.json", stats)
    write_json(work / "identity.json", {"id": identity, **recipe})
    # Intermediate Arrow files are build scratch; only packed data and audit artifacts remain.
    shutil.rmtree(work / "intermediate")

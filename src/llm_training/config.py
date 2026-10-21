"""Load and validate experiment configuration files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SUPPORTED_PRECISIONS = {"auto", "bf16", "fp16", "fp32"}


def _require_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name)
    if not isinstance(section, dict):
        raise TypeError(f"Configuration section '{name}' must be a JSON object")
    return section


def _resolve_path(project_root: Path, value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{field}' must be a nonempty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return str(path.resolve())


def _load_tree(config_path: Path, visited: set[Path]) -> dict[str, Any]:
    if config_path in visited:
        raise ValueError("Cyclic config inheritance")
    visited.add(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {config_path} at line {error.lineno}, "
            f"column {error.colno}: {error.msg}"
        ) from error

    if not isinstance(config, dict):
        raise TypeError("The configuration root must be a JSON object")

    if "extends" in config:
        parent = (config_path.parent / config.pop("extends")).resolve()
        if parent == config_path:
            raise ValueError("A config cannot extend itself")
        base = _load_tree(parent, visited)
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key].update(value)
            else:
                base[key] = value
        config = base
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    """Load inherited JSON configuration and resolve project-relative paths."""

    config_path = Path(path).expanduser().resolve()
    config = _load_tree(config_path, set())

    model = _require_section(config, "model")
    data = _require_section(config, "data")
    training = _require_section(config, "training")
    runtime = _require_section(config, "runtime")
    project_root = config_path.parent.parent

    if not isinstance(model.get("name_or_path"), str) or not model["name_or_path"].strip():
        raise ValueError("'model.name_or_path' must be a nonempty string")

    data["source"] = _resolve_path(
        project_root,
        os.environ.get("LLM_DATA_SOURCE") or data.get("source"),
        "data.source",
    )
    data["cache_dir"] = _resolve_path(
        project_root,
        os.environ.get("LLM_DATA_CACHE_DIR") or data.get("cache_dir", "data/cache"),
        "data.cache_dir",
    )

    text_column = data.get("text_column", "text")
    if not isinstance(text_column, str) or not text_column.strip():
        raise ValueError("'data.text_column' must be a nonempty string")
    data["text_column"] = text_column

    validation_fraction = data.get("validation_fraction", 0.01)
    if not isinstance(validation_fraction, (int, float)) or isinstance(validation_fraction, bool):
        raise TypeError("'data.validation_fraction' must be a number")
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("'data.validation_fraction' must be between 0 and 1")
    data["validation_fraction"] = float(validation_fraction)
    test_fraction = data.setdefault("test_fraction", 0.01)
    if (
        isinstance(test_fraction, bool)
        or not isinstance(test_fraction, (int, float))
        or not 0 < test_fraction < 1 - validation_fraction
    ):
        raise ValueError("test_fraction must be positive and val + test must be < 1")
    for key in ("group_column", "source_column"):
        data.setdefault(key, None)

    for key in ("max_shards", "max_documents"):
        value = data.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"'data.{key}' must be a positive integer or null")

    max_length = data.get("max_length")
    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 2:
        raise ValueError("'data.max_length' must be an integer of at least 2")

    for unused in ("preprocessing_num_workers", "tokenize_batch_size", "packing_batch_size"):
        data.pop(unused, None)

    training["output_dir"] = _resolve_path(
        project_root,
        training.get("output_dir"),
        "training.output_dir",
    )

    precision = runtime.get("precision", "auto")
    if not isinstance(precision, str) or precision not in SUPPORTED_PRECISIONS:
        choices = ", ".join(sorted(SUPPORTED_PRECISIONS))
        raise ValueError(f"'runtime.precision' must be one of: {choices}")
    runtime["precision"] = precision
    runtime["allow_tf32"] = bool(runtime.get("allow_tf32", True))

    config["config_path"] = str(config_path)
    return config

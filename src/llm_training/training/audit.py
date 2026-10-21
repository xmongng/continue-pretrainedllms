"""Portable experiment records; this module deliberately needs only the standard library."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_plan(config, statistics):
    """Match the single-device Trainer's rounding, including partial accumulation."""
    training = config["training"]
    batch = training["per_device_train_batch_size"]
    accumulation = training["gradient_accumulation_steps"]
    sequences = statistics["train"]["sequences"]
    batches = (
        sequences // batch
        if training.get("dataloader_drop_last", False)
        else math.ceil(sequences / batch)
    )
    updates = max(1, math.ceil(batches / accumulation))
    explicit_steps = training.get("max_steps", -1)
    steps = (
        explicit_steps
        if explicit_steps > 0
        else math.ceil(training.get("num_train_epochs", 3) * updates)
    )
    warmup = training.get("warmup_steps", 0) or math.ceil(
        training.get("warmup_ratio", 0) * steps
    )
    return {
        "world_size": 1,
        "train_sequences": sequences,
        "train_target_tokens_per_epoch": statistics["train"]["target_tokens"],
        "microbatches_per_epoch": batches,
        "optimizer_steps_per_epoch": updates,
        "expected_total_optimizer_steps": steps,
        "max_steps_overrides_epochs": explicit_steps > 0,
        "effective_batch_sequences": batch * accumulation,
        "full_step_target_tokens": batch * accumulation * (config["data"]["max_length"] - 1),
        "warmup_optimizer_steps": warmup,
        "note": "Final microbatch/accumulation group and final packed sequence may be smaller.",
    }


def gpu_sample():
    fields = [
        "index", "name", "uuid", "driver_version", "utilization.gpu", "utilization.memory",
        "memory.used", "memory.total", "power.draw", "temperature.gpu",
    ]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        rows = []
        for values in csv.reader(result.stdout.splitlines()):
            row = {}
            for key, value in zip(fields, values):
                value = value.strip()
                if key not in ("name", "uuid", "driver_version"):
                    try:
                        value = float(value)
                    except ValueError:
                        value = None
                row[key] = value
            rows.append(row)
        return {"gpus": rows}
    except (OSError, subprocess.SubprocessError) as error:
        return {"gpus": [], "gpu_sample_error": str(error)}


class ExperimentRecorder:
    """Keep each session separate so resume never erases the history of a failure."""

    def __init__(self, output, interval_seconds=30):
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        self.output = Path(output)
        self.root = self.output / "audit"
        self.session = self.root / "sessions" / self.session_id
        self.session.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.interval = interval_seconds
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.phase_totals = defaultdict(lambda: {"seconds": 0.0, "calls": 0})
        self.previous = None

    def append(self, name, record):
        record = {
            **record,
            "schema_version": 1,
            "utc": utc_now(),
            "session_id": self.session_id,
            "session_elapsed_seconds": time.monotonic() - self.started,
        }
        line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        with self.lock:
            with (self.session / name).open("a", encoding="utf-8") as stream:
                stream.write(line)

    def event(self, event, **fields):
        self.append("events.jsonl", {"event": event, **fields})

    def snapshot(self, config, artifact, contract, extra_sources=()):
        save_json(self.session / "config.json", config)
        save_json(self.session / "contract.json", contract)
        save_json(self.session / "dataset_stats.json", artifact["statistics"])
        save_json(self.session / "training_plan.json", training_plan(config, artifact["statistics"]))
        artifact_dir = Path(artifact["path"])
        manifest = artifact_dir / "split_manifest.jsonl"
        identity = artifact_dir / "identity.json"
        if identity.is_file():
            shutil.copy2(identity, self.session / "data_identity.json")
        save_json(self.session / "dataset_reference.json", {
            "artifact_id": artifact["id"], "artifact_path": str(artifact_dir),
            "split_manifest_path": str(manifest),
            "split_manifest_sha256": sha256(manifest) if manifest.is_file() else None,
        })
        package_versions = sorted(
            {f"{dist.metadata['Name']}=={dist.version}" for dist in distributions() if dist.metadata['Name']}
        )
        (self.session / "packages.txt").write_text("\n".join(package_versions) + "\n")
        save_json(self.session / "environment.json", {
            "python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "world_size": os.environ.get("WORLD_SIZE", "1"), **gpu_sample(),
        })
        code_root = Path(__file__).resolve().parents[1]
        sources = [(p, Path("llm_training") / p.relative_to(code_root))
                   for p in code_root.rglob("*.py")]
        sources += [(Path(p), Path("launcher") / Path(p).name) for p in extra_sources]
        hashes = {}
        for source, relative in sources:
            if source.is_file():
                target = self.session / "source" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                hashes[str(relative)] = sha256(source)
        config_dir = Path(config["config_path"]).parent
        for source in config_dir.glob("*.json"):
            target = self.session / "source" / "configs" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[str(Path("configs") / source.name)] = sha256(source)
        save_json(self.session / "source_sha256.json", hashes)

    def phase(self, name, seconds):
        self.phase_totals[name]["seconds"] += seconds
        self.phase_totals[name]["calls"] += 1

    def progress(self, step, tokens, max_steps, **fields):
        now = time.monotonic()
        record = {"step": step, "target_tokens_seen": tokens, "max_steps": max_steps, **fields}
        # Evaluation can log at the same step. Only optimizer progress advances the window.
        if self.previous is not None and step > self.previous[0]:
            previous_step, previous_tokens, previous_time = self.previous
            seconds = now - previous_time
            rate = (tokens - previous_tokens) / max(seconds, 1e-9)
            record.update(
                interval_seconds=seconds,
                interval_optimizer_steps=step - previous_step,
                interval_target_tokens=tokens - previous_tokens,
                interval_target_tokens_per_second=rate,
                eta_seconds=max(0, max_steps - step) * seconds / (step - previous_step),
            )
        if self.previous is None or step > self.previous[0]:
            self.previous = (step, tokens, now)
        record["phase_totals"] = dict(self.phase_totals)
        self.append("progress.jsonl", record)

    def sample_system(self):
        record = gpu_sample()
        record["cpu_load_average"] = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
        record["output_disk_free_bytes"] = shutil.disk_usage(self.output).free
        for name in ("status", "io"):
            path = Path("/proc/self") / name
            if path.exists():
                record[f"process_{name}"] = path.read_text()
        memory = Path("/proc/meminfo")
        if memory.exists():
            record["host_meminfo"] = memory.read_text()
        self.append("system_metrics.jsonl", record)

    def _sample_loop(self):
        while not self.stop_event.is_set():
            try:
                self.sample_system()
            except Exception as error:
                self.event("system_sampling_error", error=repr(error))
            self.stop_event.wait(self.interval)

    def start(self):
        self.thread = threading.Thread(target=self._sample_loop, name="experiment-monitor", daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
        save_json(self.session / "timing_totals.json", dict(self.phase_totals))


def export_metrics(output):
    """CSV for plots; metrics.jsonl remains the canonical, checkpoint-aligned history."""
    output = Path(output)
    source = output / "metrics.jsonl"
    if not source.exists():
        return
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    keys = sorted({key for row in rows for key, value in row.items()
                   if not isinstance(value, (dict, list))})
    destination = output / "audit" / "metrics.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def bundle_audit(output):
    """Export logs and provenance, excluding model weights and optimizer states."""
    output = Path(output)
    paths = sorted((output / "audit").rglob("*"))
    paths += [output / name for name in
              ("run.json", "baseline.json", "summary.json", "metrics.jsonl",
               "validation_metrics.json", "test_metrics.json")]
    destination = output / "audit_bundle.zip"
    temporary = output / "audit_bundle.zip.tmp"
    hashes = {}
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.is_file() and not path.name.endswith(".tmp"):
                relative = str(path.relative_to(output))
                content = path.read_bytes()
                archive.writestr(relative, content)
                hashes[relative] = hashlib.sha256(content).hexdigest()
        archive.writestr("bundle_manifest.json", json.dumps({
            "created_utc": utc_now(), "sha256": hashes,
            "note": "Snapshot of available logs; a running job may write further records.",
        }, indent=2))
    temporary.replace(destination)
    return destination

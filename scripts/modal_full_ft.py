"""Full-parameter Qwen3-1.7B training on one Modal A100 80GB.

Run with: modal run scripts/modal_full_ft.py
Run the same command again to resume the latest complete checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import modal

PROJECT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path("/root/configs/qwen3_1_7b_modal_a100_full_ft.json")
DATA_MOUNT = Path("/data")
RUNS_MOUNT = Path("/runs")
CORPUS = DATA_MOUNT / "corpus"
DATASET_REPO = "iraprp/chayqwen"
EXPECTED_SHARDS = 253
DEFAULT_RUN_ID = "qwen3-1.7b-a10080-e02-b16-ga2-v2"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        "transformers==4.57.6",
        "datasets==4.4.1",
        "accelerate==1.10.1",
        "huggingface_hub==0.36.0",
        "hf_xet",
        "safetensors>=0.4,<1",
        "filelock>=3.12",
        "numpy>=1.24",
    )
    .env({
        "HF_HOME": "/data/hf-cache", "PYTHONPATH": "/root",
        "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": "0",
        "LLM_LAUNCHER_SOURCE": "/root/launcher_snapshot.py",
    })
    .add_local_dir(str(PROJECT / "src" / "llm_training"), "/root/llm_training")
    .add_local_dir(str(PROJECT / "configs"), "/root/configs")
    .add_local_dir(str(PROJECT / "tests"), "/root/tests")
    .add_local_file(str(PROJECT / "scripts" / "modal_notebook.py"), "/root/scripts/modal_notebook.py")
    .add_local_file(str(PROJECT / "scripts" / "modal_full_ft.py"), "/root/launcher_snapshot.py")
)
app = modal.App("qwen3-1-7b-full-ft", image=image)
data_volume = modal.Volume.from_name("qwen3-1-7b-corpus", create_if_missing=True)
runs_volume = modal.Volume.from_name("qwen3-1-7b-full-ft-runs", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")  # Must contain HF_TOKEN.


def configure_data_paths() -> None:
    import os

    os.environ["LLM_DATA_SOURCE"] = str(CORPUS / "data")
    os.environ["LLM_DATA_CACHE_DIR"] = str(DATA_MOUNT / "cache")


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", run_id):
        raise ValueError("run_id must be 1-100 letters, digits, dots, underscores or hyphens")
    return run_id


def load_modal_config(run_id: str = DEFAULT_RUN_ID) -> dict:
    from llm_training.config import load_config

    configure_data_paths()
    config = load_config(CONFIG_PATH)
    config["training"]["output_dir"] = str(RUNS_MOUNT / validate_run_id(run_id))
    config["training"]["run_name"] = run_id
    model_pin = DATA_MOUNT / "model_source.json"
    if not model_pin.is_file():
        raise RuntimeError("Run prepare_data first to pin the model revision")
    pin = json.loads(model_pin.read_text())
    if pin["repo_id"] != config["model"]["name_or_path"]:
        raise ValueError("Model cache is pinned to a different repository")
    config["model"]["revision"] = pin["revision"]
    config["provenance"] = {
        "corpus": json.loads((DATA_MOUNT / "corpus_source.json").read_text()),
        "model": pin,
    }
    return config


@app.function(
    cpu=8,
    memory=32768,
    timeout=24 * 60 * 60,
    volumes={DATA_MOUNT: data_volume},
    secrets=[hf_secret],
)
def prepare_data() -> dict:
    """Pin/download the 253 corpus shards and build reusable packed Arrow data."""
    from huggingface_hub import HfApi, snapshot_download
    from llm_training.config import load_config
    from llm_training.data.artifacts import prepare_artifacts
    from llm_training.training.train import load_tokenizer
    from llm_training.training.audit import save_json

    configure_data_paths()
    pin_path = DATA_MOUNT / "corpus_source.json"
    if pin_path.exists():
        pin = json.loads(pin_path.read_text(encoding="utf-8"))
        if pin["repo_id"] != DATASET_REPO:
            raise ValueError("Corpus volume is pinned to a different repository")
    else:
        revision = HfApi().repo_info(DATASET_REPO, repo_type="dataset").sha
        pin = {"repo_id": DATASET_REPO, "revision": revision}
        save_json(pin_path, pin)
        data_volume.commit()

    model_pin_path = DATA_MOUNT / "model_source.json"
    if not model_pin_path.exists():
        model_config = load_config(CONFIG_PATH)["model"]
        revision = HfApi().model_info(
            model_config["name_or_path"], revision=model_config.get("revision", "main")
        ).sha
        save_json(model_pin_path, {"repo_id": model_config["name_or_path"], "revision": revision})
        data_volume.commit()
    config = load_modal_config()
    recipe_hash = hashlib.sha256(json.dumps(
        {"data": config["data"], "model": config["model"]}, sort_keys=True
    ).encode()).hexdigest()
    ready_path = DATA_MOUNT / "prepared_dataset.json"
    if ready_path.exists():
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        if ready.get("recipe_sha256") == recipe_hash and ready.get("revision") == pin["revision"]:
            artifact_dir = DATA_MOUNT / "cache" / "prepared" / ready["artifact_id"]
            if (len(list((CORPUS / "data").glob("train-*.jsonl*"))) == EXPECTED_SHARDS
                    and (artifact_dir / "dataset").is_dir()):
                return ready

    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        revision=pin["revision"],
        allow_patterns=["data/train-*.jsonl.gz", "data/train-*.jsonl"],
        local_dir=str(CORPUS),
    )
    shards = sorted((CORPUS / "data").glob("train-*.jsonl*"))
    if len(shards) != EXPECTED_SHARDS:
        raise RuntimeError(f"Expected {EXPECTED_SHARDS} corpus shards, found {len(shards)}")
    data_volume.commit()

    tokenizer = load_tokenizer(config["model"])
    datasets, artifact = prepare_artifacts(config["data"], tokenizer)
    ready = {
        "repo_id": DATASET_REPO,
        "revision": pin["revision"],
        "recipe_sha256": recipe_hash,
        "artifact_id": artifact["id"],
        "train_sequences": len(datasets["train"]),
        "validation_sequences": len(datasets["validation"]),
    }
    save_json(ready_path, ready)
    data_volume.commit()
    return ready


@app.function(
    gpu="A100-80GB",
    cpu=8,
    memory=32768,
    timeout=24 * 60 * 60,
    volumes={DATA_MOUNT: data_volume, RUNS_MOUNT: runs_volume},
    secrets=[hf_secret],
)
def train_full(run_id: str = DEFAULT_RUN_ID) -> dict:
    """Train every model weight; stop cleanly before Modal's 24-hour limit."""
    import time

    import torch
    from transformers import TrainerCallback

    from llm_training.training.train import create_trainer, run_training

    configure_data_paths()
    if not (DATA_MOUNT / "prepared_dataset.json").is_file():
        raise RuntimeError("Run prepare_data before training")
    if (not torch.cuda.is_available()
            or "A100" not in torch.cuda.get_device_name(0)
            or torch.cuda.get_device_properties(0).total_memory < 79_000_000_000):
        raise RuntimeError("This configuration requires one A100 80GB GPU")

    config = load_modal_config(run_id)
    output = Path(config["training"]["output_dir"])
    run_path = output / "run.json"
    if (output / "summary.json").is_file() and run_path.is_file():
        if json.loads(run_path.read_text()).get("status") == "completed":
            return json.loads((output / "summary.json").read_text(encoding="utf-8"))
    resume = run_path.is_file() and json.loads(run_path.read_text()).get("stage") != "initialization"
    started = time.monotonic()

    class SessionCallback(TrainerCallback):
        def __init__(self):
            self.last_commit = time.monotonic()

        def on_train_begin(self, args, state, control, **kwargs):
            torch.cuda.reset_peak_memory_stats()
            return control

        def commit(self):
            began = time.monotonic()
            runs_volume.commit()
            self.last_commit = time.monotonic()
            if trainer.recorder is not None:
                duration = self.last_commit - began
                trainer.recorder.phase("volume_commit_wall", duration)
                trainer.recorder.event("volume_committed", duration_seconds=duration)

        def on_log(self, args, state, control, **kwargs):
            if time.monotonic() - self.last_commit >= 300:
                self.commit()
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step <= 3 or state.global_step % 1000 == 0:
                gib = 1024 ** 3
                print(json.dumps({
                    "step": state.global_step,
                    "gpu_allocated_gib": round(torch.cuda.memory_allocated() / gib, 2),
                    "gpu_peak_allocated_gib": round(torch.cuda.max_memory_allocated() / gib, 2),
                    "gpu_reserved_gib": round(torch.cuda.memory_reserved() / gib, 2),
                }), flush=True)
            if time.monotonic() - started >= 18 * 60 * 60 and state.global_step < state.max_steps:
                control.should_evaluate = True
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_save(self, args, state, control, **kwargs):
            # Added after AuditCallback, whose on_save writes the COMPLETE marker.
            self.commit()
            return control

    try:
        trainer = create_trainer(config)
    except BaseException as error:
        import traceback

        from llm_training.training.audit import save_json, utc_now

        failure = {"status": "failed", "stage": "initialization", "utc": utc_now(),
                   "config": config, "error": repr(error), "traceback": traceback.format_exc()}
        save_json(output / "audit" / "initialization_failures" / f"{time.time_ns()}.json", failure)
        if not run_path.exists():
            save_json(run_path, failure)
        runs_volume.commit()
        raise
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    if trainable != total:
        raise RuntimeError(f"Expected full fine-tuning, got {trainable}/{total} trainable parameters")
    data_volume.commit()  # Persist the model/tokenizer cache for later sessions.
    trainer.add_callback(SessionCallback())
    try:
        run_training(config, trainer, resume=resume)
    finally:
        runs_volume.commit()
    return json.loads((output / "summary.json").read_text(encoding="utf-8"))


@app.function(cpu=2, memory=4096, timeout=20 * 60)
def check_pipeline() -> dict:
    """Exercise a tiny local model on CPU; download neither corpus nor Qwen weights."""
    import os
    import subprocess
    import sys

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"}
    command = (
        "import sys, unittest; "
        "suite = unittest.defaultTestLoader.discover('/root/tests'); "
        "result = unittest.TextTestRunner(verbosity=2).run(suite); "
        "sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)"
    )
    subprocess.run(
        [sys.executable, "-B", "-c", command],
        env=env, check=True,
    )
    return {"status": "passed", "device": "cpu", "external_downloads": False}


@app.function(cpu=1, memory=1024, timeout=120, volumes={RUNS_MOUNT: runs_volume})
def run_status(run_id: str = DEFAULT_RUN_ID) -> dict:
    runs_volume.reload()
    output = RUNS_MOUNT / validate_run_id(run_id)
    result = {"output_dir": str(output)}
    for name in ("run.json", "summary.json", "baseline.json"):
        path = output / name
        if path.is_file():
            value = json.loads(path.read_text())
            result[name] = value if name != "run.json" else {
                key: value.get(key) for key in ("status", "resumed_from", "audit_session_id", "error")
            }
    sessions = sorted((output / "audit" / "sessions").glob("*"))
    if sessions:
        result["latest_session"] = sessions[-1].name
        for name in ("training_plan.json", "trainer_plan.json"):
            path = sessions[-1] / name
            if path.is_file():
                result[name] = json.loads(path.read_text())
        progress = sessions[-1] / "progress.jsonl"
        if progress.exists():
            with progress.open("rb") as stream:
                stream.seek(max(0, progress.stat().st_size - 65536))
                lines = stream.read().splitlines()
            if lines:
                try:
                    result["latest_progress"] = json.loads(lines[-1])
                except json.JSONDecodeError:
                    result["latest_progress"] = "Log write in progress; try again shortly."
    return result


@app.function(cpu=1, memory=2048, timeout=600, volumes={RUNS_MOUNT: runs_volume})
def export_logs(run_id: str = DEFAULT_RUN_ID) -> str:
    from llm_training.training.audit import bundle_audit

    runs_volume.reload()
    output = RUNS_MOUNT / validate_run_id(run_id)
    if not (output / "run.json").exists():
        raise FileNotFoundError(f"No run found at {output}")
    archive = bundle_audit(output)
    runs_volume.commit()
    return "/" + str(archive.relative_to(RUNS_MOUNT))


@app.function(
    gpu="A100-80GB", cpu=8, memory=32768, timeout=6 * 60 * 60,
    volumes={DATA_MOUNT: data_volume, RUNS_MOUNT: runs_volume}, secrets=[hf_secret],
)
def evaluate_test(run_id: str = DEFAULT_RUN_ID) -> dict:
    from llm_training.evaluation.evaluate import evaluate_run

    try:
        return evaluate_run(load_modal_config(run_id), split="test")
    finally:
        runs_volume.commit()


@app.local_entrypoint()
def main(prepare_only: bool = False, sessions: int = 1, run_id: str = DEFAULT_RUN_ID,
         check_only: bool = False, status_only: bool = False, export_only: bool = False,
         test_only: bool = False):
    """Prepare on CPU, then run up to sessions resumable GPU sessions."""
    if sessions < 1:
        raise ValueError("sessions must be at least 1")
    validate_run_id(run_id)
    if check_only:
        print(check_pipeline.remote())
        return
    if status_only:
        print(json.dumps(run_status.remote(run_id), indent=2))
        return
    if export_only:
        remote_path = export_logs.remote(run_id)
        print(f"modal volume get qwen3-1-7b-full-ft-runs {remote_path} ./audit_bundle.zip")
        return
    if test_only:
        print(json.dumps(evaluate_test.remote(run_id), indent=2))
        return
    print("Prepared corpus:", prepare_data.remote())
    if prepare_only:
        return
    for session in range(1, sessions + 1):
        summary = train_full.remote(run_id)
        print(f"Training session {session}:", summary)
        if summary["status"] == "completed":
            break

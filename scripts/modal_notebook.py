"""Run directly inside a Modal Notebook with a mounted persistent Volume."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unittest
from pathlib import Path
from urllib.parse import urlparse

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
DEFAULT_RUN = "qwen3-1.7b-notebook-a10080-e02-b16-ga2-v1"


def dataset_repository(url):
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "huggingface.co":
        raise ValueError("Use a Hugging Face dataset URL: https://huggingface.co/datasets/owner/name")
    parts = parsed.path.strip("/").split("/")
    if len(parts) != 3 or parts[0] != "datasets" or not all(parts[1:]):
        raise ValueError("Use the dataset homepage URL, without /tree or /resolve")
    return "/".join(parts[1:])


def configure(volume, run_id):
    from llm_training.config import load_config

    os.environ["LLM_DATA_SOURCE"] = str(volume / "corpus" / "data")
    os.environ["LLM_DATA_CACHE_DIR"] = str(volume / "cache")
    config = load_config(PROJECT / "configs/qwen3_1_7b_modal_a100_full_ft.json")
    config["training"].update(output_dir=str(volume / "runs" / run_id), run_name=run_id)
    pin_path = volume / "corpus_pin.json"
    if pin_path.exists():
        pin = json.loads(pin_path.read_text())
        if pin["model_repo"] != config["model"]["name_or_path"]:
            raise ValueError("The pinned model differs; use a new Volume for a different model")
        config["model"]["revision"] = pin["model_revision"]
        config["provenance"] = pin
    return config


def prepare(volume, run_id, data_url):
    from huggingface_hub import HfApi, snapshot_download

    from llm_training.data.artifacts import prepare_artifacts
    from llm_training.training.audit import save_json, training_plan
    from llm_training.training.train import load_tokenizer

    repo = dataset_repository(data_url)
    config = configure(volume, run_id)
    pin_path = volume / "corpus_pin.json"
    if pin_path.exists():
        pin = json.loads(pin_path.read_text())
        if pin["data_repo"] != repo:
            raise ValueError("This Volume is pinned to another dataset; use a new Volume")
    else:
        api = HfApi()
        pin = {
            "data_repo": repo, "data_url": data_url,
            "data_revision": api.repo_info(repo, repo_type="dataset").sha,
            "model_repo": config["model"]["name_or_path"],
            "model_revision": api.model_info(
                config["model"]["name_or_path"], revision=config["model"].get("revision", "main")
            ).sha,
        }
        save_json(pin_path, pin)
    snapshot_download(
        repo_id=repo, repo_type="dataset", revision=pin["data_revision"],
        allow_patterns=["data/train-*.jsonl.gz", "data/train-*.jsonl"],
        local_dir=str(volume / "corpus"),
    )
    shards = list((volume / "corpus/data").glob("train-*.jsonl*"))
    if not shards:
        raise ValueError("Dataset must contain data/train-*.jsonl[.gz] shards")
    if repo == "iraprp/chayqwen" and len(shards) != 253:
        raise ValueError(f"Expected 253 psychology shards, found {len(shards)}")
    config = configure(volume, run_id)
    tokenizer = load_tokenizer(config["model"])
    _, artifact = prepare_artifacts(config["data"], tokenizer)
    report = {"artifact_id": artifact["id"], "artifact_path": artifact["path"],
              "statistics": artifact["statistics"], "plan": training_plan(config, artifact["statistics"])}
    save_json(volume / "prepared_notebook.json", report)
    print(json.dumps(report, indent=2))


def train(volume, run_id):
    import torch
    from transformers import TrainerCallback

    from llm_training.training.audit import save_json, utc_now
    from llm_training.training.train import create_trainer, run_training

    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or "A100" not in torch.cuda.get_device_name(0)
            or torch.cuda.get_device_properties(0).total_memory < 79_000_000_000):
        raise RuntimeError("Select one A100 80GB in the Modal Notebook compute settings")
    if not (volume / "prepared_notebook.json").is_file():
        raise RuntimeError("Run the data preparation cell first")
    config = configure(volume, run_id)
    output = Path(config["training"]["output_dir"])
    run_path = output / "run.json"
    prior = json.loads(run_path.read_text()) if run_path.exists() else {}
    if prior.get("status") == "completed":
        print((output / "summary.json").read_text())
        return
    resume = bool(prior) and prior.get("stage") != "initialization"
    if not prior and output.exists() and any(output.iterdir()):
        raise FileExistsError("Output already contains files without a run record; use a new run ID")
    started = time.monotonic()

    class SessionLimit(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            torch.cuda.reset_peak_memory_stats()
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if time.monotonic() - started >= 18 * 60 * 60 and state.global_step < state.max_steps:
                control.should_evaluate = True
                control.should_save = True
                control.should_training_stop = True
            return control

    try:
        trainer = create_trainer(config)
    except BaseException as error:
        import traceback

        failure = {"status": "failed", "stage": "initialization", "utc": utc_now(),
                   "config": config, "error": repr(error), "traceback": traceback.format_exc()}
        save_json(output / "audit/initialization_failures" / f"{time.time_ns()}.json", failure)
        if not prior:
            save_json(run_path, failure)
        raise
    trainer.add_callback(SessionLimit())
    run_training(config, trainer, resume=resume)
    print((output / "summary.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "check", "train", "export", "test"):
        actions.add_argument("--" + name, action="store_true")
    parser.add_argument("--volume", default="/mnt/qwen3-cpt")
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    parser.add_argument("--data-url", default="https://huggingface.co/datasets/iraprp/chayqwen")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", args.run_id):
        parser.error("Invalid run ID")
    volume = Path(args.volume).resolve()
    if not volume.is_dir():
        parser.error(f"Attach the persistent Modal Volume at {volume} before running cells")
    os.environ["HF_HOME"] = str(volume / "hf-cache")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.check else "0"
    os.environ["LLM_LAUNCHER_SOURCE"] = str(Path(__file__).resolve())
    if args.check:
        suite = unittest.defaultTestLoader.discover(str(PROJECT / "tests"))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
    elif args.prepare:
        prepare(volume, args.run_id, args.data_url)
    elif args.train:
        train(volume, args.run_id)
    elif args.export:
        from llm_training.training.audit import bundle_audit

        output = volume / "runs" / args.run_id
        if not (output / "run.json").exists():
            raise FileNotFoundError("No run record found")
        print(bundle_audit(output))
    else:
        from llm_training.evaluation.evaluate import evaluate_run

        evaluate_run(configure(volume, args.run_id), split="test")


if __name__ == "__main__":
    main()

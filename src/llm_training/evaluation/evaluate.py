"""Explicit validation/test evaluation of a selected completed run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_training.config import load_config
from llm_training.data.artifacts import write_json
from llm_training.training.train import create_trainer
from llm_training.training.audit import ExperimentRecorder, export_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", help="Default: best checkpoint from summary.json")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    evaluate_run(load_config(args.config), args.split, args.model)


def evaluate_run(config, split="validation", model=None):
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    output = Path(config["training"]["output_dir"])
    summary = json.loads((output / "summary.json").read_text())
    if summary["status"] != "completed":
        raise ValueError("Evaluate a completed run after selecting the checkpoint")
    model = model or summary["best_checkpoint"]
    if not model:
        raise ValueError("No best checkpoint recorded")
    if split == "test" and (output / "test_metrics.json").exists():
        raise FileExistsError("Test already evaluated. Keep the recorded final assessment.")
    trainer = create_trainer(config, model_source=model)
    if trainer.dataset_artifact["id"] != summary["artifact_id"]:
        raise ValueError("Evaluation corpus/tokenizer differs from the recorded run")
    dataset = trainer.test_dataset if split == "test" else trainer.eval_dataset
    recorder = ExperimentRecorder(output)
    trainer.recorder = recorder
    try:
        contract = json.loads((output / "run.json").read_text())["contract"]
        recorder.snapshot(config, trainer.dataset_artifact, contract)
        recorder.event("assessment_started", split=split, model=model)
        recorder.start()
        metrics = trainer.evaluate(eval_dataset=dataset, metric_key_prefix=split)
        result = {"model": model, "artifact_id": summary["artifact_id"],
                  "audit_session_id": recorder.session_id, **metrics}
        write_json(output / f"{split}_metrics.json", result)
        recorder.event("assessment_finished", split=split, metrics=metrics)
        print(json.dumps(metrics, indent=2))
        return result
    except BaseException as error:
        recorder.event("assessment_failed", split=split, error=repr(error))
        raise
    finally:
        recorder.close()
        export_metrics(output)
        trainer.recorder = None


if __name__ == "__main__":
    main()

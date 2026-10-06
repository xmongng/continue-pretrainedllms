"""Explicit validation/test evaluation of a selected completed run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_training.config import load_config
from llm_training.data.artifacts import write_json
from llm_training.training.train import create_trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", help="Default: best checkpoint from summary.json")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(config["training"]["output_dir"])
    summary = json.loads((output / "summary.json").read_text())
    model = args.model or summary["best_checkpoint"]
    if not model:
        raise ValueError("No best checkpoint recorded")
    if args.split == "test" and (output / "test_metrics.json").exists():
        raise FileExistsError("Test already evaluated. Keep the recorded final assessment.")
    trainer = create_trainer(config, model_source=model)
    if trainer.dataset_artifact["id"] != summary["artifact_id"]:
        raise ValueError("Evaluation corpus/tokenizer differs from the recorded run")
    dataset = trainer.test_dataset if args.split == "test" else trainer.eval_dataset
    metrics = trainer.evaluate(eval_dataset=dataset, metric_key_prefix=args.split)
    write_json(output / f"{args.split}_metrics.json", {"model": model, **metrics})
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

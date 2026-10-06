"""Print a compact experiment comparison without generating duplicate result files."""

import argparse
import csv
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/qwen3-1.7b-base")
    args = parser.parse_args()
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "run",
            "status",
            "artifact_id",
            "epochs_budget",
            "steps",
            "target_tokens_seen",
            "baseline_loss",
            "best_val_loss",
            "best_val_perplexity",
            "best_checkpoint",
        ]
    )
    for path in sorted(Path(args.root).glob("*/summary.json")):
        summary = json.loads(path.read_text())
        run = json.loads((path.parent / "run.json").read_text())
        baseline = json.loads((path.parent / "baseline.json").read_text())
        writer.writerow(
            [
                path.parent.name,
                summary["status"],
                summary["artifact_id"],
                run["contract"]["config"]["training"].get("num_train_epochs"),
                summary["last_step"],
                summary["target_tokens_seen"],
                baseline["baseline_loss"],
                summary["validation"]["eval_loss"],
                summary["validation"]["eval_perplexity"],
                summary["best_checkpoint"],
            ]
        )


if __name__ == "__main__":
    main()

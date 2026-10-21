"""Count corpus tokens with the training tokenizer, without loading model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_training.config import load_config
from llm_training.data.dataset import discover_data_files
from llm_training.data.records import iter_json_records, training_text
from llm_training.training.train import load_tokenizer


def count_shard(filename, tokenizer, text_columns, batch_size):
    documents = 0
    tokens = 0
    batch = []
    for line_number, row in enumerate(iter_json_records(filename), 1):
        text, _ = training_text(row, text_columns, f"{filename}:{line_number}")
        batch.append(text)
        documents += 1
        if len(batch) == batch_size:
            encoded = tokenizer(
                batch,
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
            )
            tokens += sum(map(len, encoded["input_ids"]))
            batch.clear()
    if batch:
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )
        tokens += sum(map(len, encoded["input_ids"]))
    return {"documents": documents, "text_tokens": tokens, "tokens_with_eos": tokens + documents}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen3_1_7b_base_e01.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", help="Default: data.cache_dir/token_counts.json")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    config = load_config(args.config)
    data = config["data"]
    files = discover_data_files(data["source"], data["cache_dir"])
    if any(not filename.endswith(".jsonl.gz") for filename in files):
        raise ValueError("This counting command expects psychology JSONL.GZ shards")
    tokenizer = load_tokenizer(config["model"])
    text_columns = [data["text_column"], *data.get("text_column_fallbacks", [])]
    # This command intentionally does not initialize a Trainer or load model weights.
    report = {
        "model": config["model"]["name_or_path"],
        "revision": config["model"].get("revision", "main"),
        "resolved_tokenizer_commit": tokenizer.init_kwargs.get("_commit_hash"),
        "source": data["source"],
        "text_column": data["text_column"],
        "text_column_fallbacks": data["text_column_fallbacks"],
        "eos_token_id": tokenizer.eos_token_id,
        "documents": 0,
        "text_tokens": 0,
        "tokens_with_eos": 0,
        "shards": [],
    }
    for index, filename in enumerate(files, 1):
        counts = count_shard(filename, tokenizer, text_columns, args.batch_size)
        report["shards"].append({"file": Path(filename).name, **counts})
        for key in ("documents", "text_tokens", "tokens_with_eos"):
            report[key] += counts[key]
        print(
            f"[{index}/{len(files)}] {Path(filename).name}: "
            f"{counts['tokens_with_eos']:,} tokens; total={report['tokens_with_eos']:,}",
            flush=True,
        )
    output = Path(args.output) if args.output else Path(data["cache_dir"]) / "token_counts.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "shards"}, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

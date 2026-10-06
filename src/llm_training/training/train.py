"""Full fine-tuning with causal next-token prediction."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import traceback
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

from llm_training.config import load_config
from llm_training.data.artifacts import file_hash, prepare_artifacts, write_json
from llm_training.data.dataset import CausalLMCollator
from llm_training.training.monitoring import AuditCallback, CPTTrainer


def resolve_precision(runtime_config: dict[str, Any]) -> dict[str, Any]:
    """Resolve automatic precision against the available accelerator."""

    precision = runtime_config.get("precision", "auto")
    if precision == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            precision = "bf16"
        elif torch.cuda.is_available():
            precision = "fp16"
        else:
            precision = "fp32"

    if precision == "bf16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 was requested, but the CUDA device does not support BF16")
    if precision == "fp16" and not torch.cuda.is_available():
        raise ValueError("FP16 training currently requires a CUDA device in this pipeline")

    supports_tf32 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    tf32 = bool(runtime_config.get("allow_tf32", True) and supports_tf32)
    return {
        "name": precision,
        # Autocast uses reduced precision; master weights and Adam states stay FP32.
        "dtype": torch.float32,
        "bf16": precision == "bf16",
        "fp16": precision == "fp16",
        "tf32": tf32,
    }


def load_tokenizer(
    model_config: dict[str, Any],
    tokenizer_source: str | None = None,
):
    """Load a tokenizer and configure safe dynamic padding."""

    source = tokenizer_source or model_config["name_or_path"]
    tokenizer_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
        "use_fast": bool(model_config.get("use_fast_tokenizer", True)),
    }
    if tokenizer_source is None:
        tokenizer_kwargs["revision"] = model_config.get("revision", "main")

    tokenizer = AutoTokenizer.from_pretrained(source, **tokenizer_kwargs)
    if tokenizer.eos_token_id is None:
        raise ValueError("The selected tokenizer does not define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(
    model_config: dict[str, Any],
    precision: dict[str, Any],
    model_source: str | None = None,
):
    """Load a causal base model without quantization for full fine-tuning."""

    source = model_source or model_config["name_or_path"]
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
        "dtype": precision["dtype"],
        "low_cpu_mem_usage": True,
    }
    if model_source is None:
        model_kwargs["revision"] = model_config.get("revision", "main")
    attention = model_config.get("attn_implementation")
    if attention:
        model_kwargs["attn_implementation"] = attention

    model = AutoModelForCausalLM.from_pretrained(source, **model_kwargs)

    # Full fine-tuning updates every model parameter.
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def create_trainer(
    config: dict[str, Any],
    model_source: str | None = None,
) -> CPTTrainer:
    """Build the tokenizer, model, datasets, arguments, and Trainer."""

    set_seed(int(config["training"].get("seed", 42)))
    for option in ("deepspeed", "fsdp", "ignore_data_skip", "label_smoothing_factor"):
        if config["training"].get(option):
            raise ValueError(f"{option} is outside this single-device audited runner")
    precision = resolve_precision(config["runtime"])
    tokenizer = load_tokenizer(config["model"], tokenizer_source=model_source)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 or torch.cuda.device_count() > 1:
        raise ValueError("This audited runner supports one device. Set CUDA_VISIBLE_DEVICES.")
    datasets, artifact = prepare_artifacts(config["data"], tokenizer)
    model = load_model(config["model"], precision, model_source=model_source)
    model.config.pad_token_id = tokenizer.pad_token_id
    limit = getattr(model.config, "max_position_embeddings", None)
    if limit and config["data"]["max_length"] > limit:
        raise ValueError(f"max_length exceeds model context limit {limit}")

    training_values = dict(config["training"])
    save_total_limit = training_values.setdefault("save_total_limit", 2)
    if isinstance(save_total_limit, bool) or not isinstance(save_total_limit, int):
        raise TypeError("save_total_limit must be an integer")
    if save_total_limit < 2:
        raise ValueError("save_total_limit must be at least 2 to retain best and latest")
    training_values.update(
        bf16=precision["bf16"],
        fp16=precision["fp16"],
        tf32=precision["tf32"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_only_model=False,
        prediction_loss_only=False,
        logging_nan_inf_filter=False,
        optim="adamw_torch",
    )
    training_args = TrainingArguments(**training_values)

    if training_args.gradient_checkpointing:
        model.config.use_cache = False

    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    print(
        json.dumps(
            {
                "precision": precision["name"],
                "train_sequences": len(datasets["train"]),
                "validation_sequences": len(datasets["validation"]),
                "trainable_parameters": trainable,
                "total_parameters": total,
            },
            indent=2,
        )
    )

    trainer = CPTTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=CausalLMCollator(tokenizer, pad_to_multiple_of=None),
    )
    trainer.dataset_artifact = artifact
    trainer.test_dataset = datasets["test"]
    trainer.add_callback(AuditCallback(trainer))
    return trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen3_1_7b_base_e01.json")
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const=True,
        default=None,
        help="Resume the latest checkpoint or a supplied checkpoint path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    output_dir = Path(config["training"]["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume_from_checkpoint:
        entries = {path.name for path in output_dir.iterdir()}
        run_path = output_dir / "run.json"
        existing_run = json.loads(run_path.read_text()) if run_path.is_file() else {}
        retryable = (
            entries == {"run.json"}
            and existing_run.get("status") == "failed"
            and existing_run.get("stage") == "initialization"
        )
        if not retryable:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Use a new directory or resume a complete checkpoint."
            )
    try:
        trainer = create_trainer(config)
    except Exception as error:
        if not args.resume_from_checkpoint:
            output_dir.mkdir(parents=True, exist_ok=True)
            previous = []
            run_path = output_dir / "run.json"
            if run_path.exists():
                previous = json.loads(run_path.read_text()).get("initialization_failures", [])
            failure = {
                "config": config,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
            write_json(
                run_path,
                {
                    "status": "failed",
                    "stage": "initialization",
                    **failure,
                    "initialization_failures": [*previous, failure],
                },
            )
        raise
    run_training(config, trainer, args.resume_from_checkpoint)


def run_training(config, trainer, resume=None):
    """Audit configuration/data identity before any continuation; retain full checkpoints."""
    output = Path(config["training"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    code_root = Path(__file__).resolve().parents[1]
    model_path = Path(config["model"]["name_or_path"])
    contract = {
        "config": {k: v for k, v in config.items() if k != "config_path"},
        "artifact_id": trainer.dataset_artifact["id"],
        "model_commit": getattr(trainer.model.config, "_commit_hash", None),
        "packages": {
            name: version(name) for name in ("torch", "transformers", "datasets", "accelerate")
        },
        "device": str(trainer.args.device),
        "python": platform.python_version(),
        "code_sha256": {
            str(p.relative_to(code_root)): file_hash(p) for p in sorted(code_root.rglob("*.py"))
        },
        "local_model_sha256": {
            str(p.relative_to(model_path)): file_hash(p)
            for p in sorted(model_path.rglob("*"))
            if p.is_file() and p.suffix in (".safetensors", ".bin", ".json")
        }
        if model_path.is_dir()
        else None,
    }
    checkpoint = None
    prior_initialization_failures = []
    if resume:
        existing_run = json.loads((output / "run.json").read_text())
        if existing_run["contract"] != contract:
            raise ValueError("Resume rejected: data, config, model revision or environment changed")
        if resume is True:
            candidates = [p for p in output.glob("checkpoint-*") if (p / "COMPLETE").exists()]
            if not candidates:
                raise ValueError("No complete checkpoint available")
            checkpoint = max(candidates, key=lambda p: int(p.name.split("-")[-1]))
        else:
            checkpoint = Path(resume).resolve()
        if checkpoint.parent.resolve() != output.resolve():
            raise ValueError("Resume must use a checkpoint from this run")
        required = (
            "COMPLETE",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "trainer_state.json",
            "accounting.json",
            "config.json",
        )
        for name in required:
            if not (checkpoint / name).is_file():
                raise ValueError(f"Incomplete resume checkpoint: missing {name}")
        if trainer.args.fp16 and not (checkpoint / "scaler.pt").is_file():
            raise ValueError("FP16 checkpoint is missing scaler.pt")
        if existing_run.get("status") == "completed":
            raise ValueError("This run is already complete; there is nothing to resume")
        accounting = json.loads((checkpoint / "accounting.json").read_text())
        trainer.target_tokens_seen = accounting["target_tokens_seen"]
        trainer.elapsed_before = accounting["elapsed_seconds"]
        # Remove logs from work performed after the restored checkpoint, avoiding duplicate history.
        log = output / "metrics.jsonl"
        if log.exists():
            checkpoint_boundary = int(accounting["metrics_bytes"])
            if checkpoint_boundary > log.stat().st_size:
                raise ValueError("Checkpoint log boundary exceeds the current metrics file")
            with log.open("r+b") as stream:
                stream.truncate(checkpoint_boundary)
    elif (output / "run.json").exists():
        existing_run = json.loads((output / "run.json").read_text())
        if existing_run.get("status") != "failed" or existing_run.get("stage") != "initialization":
            raise FileExistsError("Run already exists; use resume or a fresh output directory")
        prior_initialization_failures = existing_run.get("initialization_failures", [])

    run_record = {
        "contract": contract,
        "artifact_path": trainer.dataset_artifact["path"],
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "status": "running",
        "resumed_from": str(checkpoint) if checkpoint else None,
    }
    if prior_initialization_failures:
        run_record["initialization_failures"] = prior_initialization_failures
    write_json(output / "run.json", run_record)
    try:
        if checkpoint is None:
            baseline = trainer.evaluate(metric_key_prefix="baseline")
            write_json(output / "baseline.json", baseline)
        trainer.started_at = time.monotonic()
        result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        metrics = trainer.evaluate()
        completed = trainer.state.global_step >= trainer.state.max_steps
        summary = {
            "status": "completed" if completed else "stopped",
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "last_step": trainer.state.global_step,
            "target_tokens_seen": trainer.target_tokens_seen,
            "elapsed_seconds": trainer.elapsed(),
            "training_session_metrics": result.metrics,
            "validation": metrics,
            "artifact_id": trainer.dataset_artifact["id"],
        }
        write_json(output / "summary.json", summary)
        run = json.loads((output / "run.json").read_text())
        run["status"] = summary["status"]
        write_json(output / "run.json", run)
    except BaseException as error:
        run = json.loads((output / "run.json").read_text())
        run.update(
            status="failed",
            step=trainer.state.global_step,
            error=repr(error),
            traceback=traceback.format_exc(),
        )
        write_json(output / "run.json", run)
        raise


if __name__ == "__main__":
    main()

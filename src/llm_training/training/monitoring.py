"""Token-weighted loss, live diagnostics, and complete checkpoint accounting."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Trainer, TrainerCallback

from llm_training.data.artifacts import write_json


def summed_nll(logits, labels):
    targets = labels[:, 1:].contiguous()
    losses = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
        ignore_index=-100,
    )
    return losses.reshape(targets.shape).sum(-1), targets.ne(-100).sum(-1)


def token_metrics(prediction):
    sums = np.asarray(prediction.predictions, dtype=np.float64).sum(axis=0)
    loss = float(sums[0] / sums[1])
    if not math.isfinite(loss):
        raise FloatingPointError("Non-finite evaluation loss")
    return {
        "loss": loss,
        "perplexity": math.exp(loss) if loss < 709 else None,
        "target_tokens": int(sums[1]),
    }


class CPTTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, compute_metrics=token_metrics, compute_loss_func=self.token_loss, **kwargs
        )
        self.target_tokens_seen = 0
        self.elapsed_before = 0.0
        self.started_at = time.monotonic()
        self.model_accepts_loss_kwargs = False

    def token_loss(self, outputs, labels, num_items_in_batch=None):
        nll, count = summed_nll(outputs.logits, labels)
        total = nll.sum()
        if not torch.isfinite(total):
            raise FloatingPointError(f"Non-finite training loss at step {self.state.global_step}")
        self.target_tokens_seen += int(count.sum().item())
        denominator = num_items_in_batch if num_items_in_batch is not None else count.sum()
        return total / denominator

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels")
        with torch.no_grad(), self.compute_loss_context_manager():
            output = model(**inputs)
            nll, count = summed_nll(output.logits, labels)
        # Gather just two numbers per sequence; never retain vocabulary-sized logits.
        values = torch.stack((nll.double(), count.double()), dim=1)
        return None, values, torch.zeros(len(labels), device=labels.device)

    def elapsed(self):
        return self.elapsed_before + time.monotonic() - self.started_at


class AuditCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.last_saved_step = -1

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        norms = [p.grad.detach().norm() for p in model.parameters() if p.grad is not None]
        if norms and not torch.isfinite(torch.stack(norms)).all():
            raise FloatingPointError(f"Non-finite gradient at step {state.global_step}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        trainer = self.trainer
        record = {
            "step": state.global_step,
            "epoch": state.epoch,
            "target_tokens_seen": trainer.target_tokens_seen,
            "elapsed_seconds": trainer.elapsed(),
            **(logs or {}),
        }
        record["target_tokens_per_second_wall"] = trainer.target_tokens_seen / max(
            trainer.elapsed(), 1e-9
        )
        if torch.cuda.is_available():
            record["gpu_allocated_bytes"] = torch.cuda.memory_allocated()
            record["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        with (Path(args.output_dir) / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is None or abs(state.epoch - round(state.epoch)) > 1e-8:
            return control
        if self.last_saved_step == state.global_step:
            return control
        control.should_evaluate = True
        control.should_save = True
        control.should_log = True
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= state.max_steps:
            control.should_evaluate = True
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        self.last_saved_step = state.global_step
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        metrics_path = Path(args.output_dir) / "metrics.jsonl"
        write_json(
            checkpoint / "accounting.json",
            {
                "target_tokens_seen": self.trainer.target_tokens_seen,
                "elapsed_seconds": self.trainer.elapsed(),
                "global_step": state.global_step,
                # Resume truncates to this exact byte, including same-step logs only
                # when they were already durable at the checkpoint boundary.
                "metrics_bytes": metrics_path.stat().st_size if metrics_path.exists() else 0,
            },
        )
        # The marker is written last, so auto-resume ignores an interrupted save.
        (checkpoint / "COMPLETE").write_text("complete\n")

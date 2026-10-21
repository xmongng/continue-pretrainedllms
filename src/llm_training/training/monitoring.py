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
from torch.utils.checkpoint import checkpoint

from llm_training.data.artifacts import write_json


def summed_nll(logits, labels, chunk_tokens=512):
    """Exact shifted NLL without copying the whole batch x sequence x vocabulary tensor.

    Recompute each small cross-entropy during backward instead of retaining all its
    FP32 softmax intermediates. The model still materializes its output logits.
    """
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")

    def chunk_loss(scores, targets):
        return F.cross_entropy(scores.float(), targets, reduction="sum", ignore_index=-100)

    totals = []
    for sequence, target_row in zip(logits, labels):
        chunks = []
        for start in range(0, len(target_row) - 1, chunk_tokens):
            end = min(start + chunk_tokens, len(target_row) - 1)
            scores = sequence[start:end]
            targets = target_row[start + 1:end + 1]
            if torch.is_grad_enabled() and scores.requires_grad:
                loss = checkpoint(chunk_loss, scores, targets, use_reentrant=False)
            else:
                loss = chunk_loss(scores, targets)
            chunks.append(loss)
        totals.append(torch.stack(chunks).sum())
    return torch.stack(totals), labels[:, 1:].ne(-100).sum(-1)


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
        self.recorder = None

    def get_batch_samples(self, *args, **kwargs):
        started = time.monotonic()
        try:
            return super().get_batch_samples(*args, **kwargs)
        finally:
            if self.recorder is not None:
                # Includes Accelerate's device transfer and target counting, not pure disk I/O.
                self.recorder.phase("batch_fetch_host", time.monotonic() - started)

    def training_step(self, *args, **kwargs):
        started = time.monotonic()
        try:
            return super().training_step(*args, **kwargs)
        finally:
            if self.recorder is not None:
                # CUDA is asynchronous: this is host wall time, not a GPU kernel timer.
                self.recorder.phase("training_step_host", time.monotonic() - started)

    def evaluate(self, *args, **kwargs):
        started = time.monotonic()
        prefix = kwargs.get("metric_key_prefix", "eval")
        if self.recorder is not None:
            self.recorder.event("evaluation_started", step=self.state.global_step, prefix=prefix)
        try:
            metrics = super().evaluate(*args, **kwargs)
        except BaseException as error:
            if self.recorder is not None:
                self.recorder.event("evaluation_failed", prefix=prefix, error=repr(error))
            raise
        else:
            if self.recorder is not None:
                self.recorder.event("evaluation_finished", step=self.state.global_step,
                                    prefix=prefix, metrics=metrics,
                                    duration_seconds=time.monotonic() - started)
            return metrics
        finally:
            if self.recorder is not None:
                self.recorder.phase("evaluation_wall", time.monotonic() - started)

    def _save_checkpoint(self, *args, **kwargs):
        started = time.monotonic()
        succeeded = False
        try:
            result = super()._save_checkpoint(*args, **kwargs)
            succeeded = True
            return result
        finally:
            if self.recorder is not None:
                duration = time.monotonic() - started
                self.recorder.phase("checkpoint_write_wall", duration)
                self.recorder.event("checkpoint_write_finished", step=self.state.global_step,
                                    duration_seconds=duration, succeeded=succeeded)

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

    def on_train_begin(self, args, state, control, **kwargs):
        recorder = self.trainer.recorder
        if recorder is not None:
            recorder.previous = (state.global_step, self.trainer.target_tokens_seen, time.monotonic())
            plan = {
                "actual_total_optimizer_steps": state.max_steps,
                "warmup_optimizer_steps": args.get_warmup_steps(state.max_steps),
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_batch_sequences": args.train_batch_size * args.gradient_accumulation_steps,
                "starting_step": state.global_step,
            }
            write_json(recorder.session / "trainer_plan.json", plan)
            recorder.event("training_started", **plan)
            print(json.dumps({"training_plan": plan}), flush=True)
        return control

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        # AMP's GradScaler handles transient FP16 overflows by skipping the
        # optimizer update and reducing its scale. Raising here prevents that
        # recovery. A norm reduction is also not an exact elementwise
        # finiteness check for the gradients.
        if args.fp16:
            return control
        finite = [torch.isfinite(p.grad.detach()).all()
                  for p in model.parameters() if p.grad is not None]
        # One scalar synchronization instead of one per parameter tensor.
        if finite and not torch.stack(finite).all():
            raise FloatingPointError(f"Non-finite gradient at step {state.global_step}")
        return control

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
        if trainer.recorder is not None:
            record["session_id"] = trainer.recorder.session_id
            from llm_training.training.audit import utc_now

            record["utc"] = utc_now()
        record["target_tokens_per_second_wall"] = trainer.target_tokens_seen / max(
            trainer.elapsed(), 1e-9
        )
        if torch.cuda.is_available():
            record["gpu_allocated_bytes"] = torch.cuda.memory_allocated()
            record["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        with (Path(args.output_dir) / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if trainer.recorder is not None:
            trainer.recorder.progress(
                state.global_step, trainer.target_tokens_seen, state.max_steps,
                metrics=logs or {},
            )

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
        if self.trainer.recorder is not None:
            self.trainer.recorder.event("checkpoint_complete", step=state.global_step,
                                        path=str(checkpoint))

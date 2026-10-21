"""Offline CPU regression tests using a locally generated tiny causal LM."""

import copy
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ML_AVAILABLE = all(importlib.util.find_spec(name) is not None
                   for name in ("torch", "transformers", "datasets", "accelerate"))


@unittest.skipUnless(ML_AVAILABLE, "Install project dependencies to run offline model tests")
class TrainingTests(unittest.TestCase):
    def test_chunked_loss_matches_reference_values_and_gradients(self):
        import torch
        import torch.nn.functional as functional

        from llm_training.training.monitoring import summed_nll

        torch.manual_seed(4)
        reference_logits = torch.randn(3, 9, 17, requires_grad=True)
        actual_logits = reference_logits.detach().clone().requires_grad_()
        labels = torch.randint(0, 17, (3, 9))
        labels[:, 0] = -100
        labels[1, -3:] = -100
        labels[2, :] = -100
        losses = functional.cross_entropy(
            reference_logits[:, :-1].contiguous().view(-1, 17),
            labels[:, 1:].contiguous().view(-1), reduction="none", ignore_index=-100,
        ).reshape(3, 8).sum(-1)
        actual, counts = summed_nll(actual_logits, labels, chunk_tokens=3)
        torch.testing.assert_close(actual, losses)
        self.assertEqual(counts.tolist(), [8, 5, 0])
        losses.sum().backward()
        actual.sum().backward()
        torch.testing.assert_close(actual_logits.grad, reference_logits.grad)
        with torch.no_grad():
            evaluated, _ = summed_nll(actual_logits, labels, chunk_tokens=2)
        torch.testing.assert_close(evaluated, losses.detach())

    def test_nonfinite_gradients_are_rejected(self):
        import torch
        from types import SimpleNamespace

        from llm_training.training.monitoring import AuditCallback

        model = torch.nn.Linear(2, 2)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        model.bias.grad[0] = float("inf")
        with self.assertRaises(FloatingPointError):
            AuditCallback(None).on_pre_optimizer_step(
                SimpleNamespace(fp16=False), SimpleNamespace(global_step=1), None, model=model,
            )

    def test_train_stop_resume_matches_uninterrupted_run_and_audits(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, TrainerCallback

        from llm_training.config import load_config
        from llm_training.training.train import create_trainer, run_training

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model"
            backend = Tokenizer(WordLevel(
                {"[UNK]": 0, "[EOS]": 1, "[PAD]": 2, "paper": 3, "psychology": 4,
                 **{f"word{i}": i + 5 for i in range(30)}}, unk_token="[UNK]",
            ))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]",
            )
            tokenizer.save_pretrained(model_path)
            torch.manual_seed(42)
            GPT2LMHeadModel(GPT2Config(
                vocab_size=len(tokenizer), n_positions=32, n_embd=16, n_layer=1, n_head=2,
                resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
                eos_token_id=1, pad_token_id=2,
            )).save_pretrained(model_path)
            corpus = root / "corpus.jsonl"
            corpus.write_text("\n".join(json.dumps({"text": f"paper psychology word{i} " * 4})
                                         for i in range(30)) + "\n")
            raw = {
                "model": {"name_or_path": str(model_path)},
                "data": {"source": str(corpus), "cache_dir": str(root / "cache"),
                         "text_column": "text", "max_length": 16,
                         "validation_fraction": 0.1, "test_fraction": 0.1, "seed": 42},
                "training": {"output_dir": str(root / "resumed"), "max_steps": 2,
                             "learning_rate": 2e-5, "warmup_ratio": 0.01,
                             "per_device_train_batch_size": 2, "per_device_eval_batch_size": 2,
                             "gradient_accumulation_steps": 2, "eval_strategy": "steps",
                             "eval_steps": 1, "save_strategy": "steps", "save_steps": 1,
                             "logging_steps": 1, "report_to": "none", "use_cpu": True,
                             "disable_tqdm": True, "seed": 42},
                "runtime": {"precision": "fp32"},
            }
            config_dir = root / "configs"
            config_dir.mkdir()
            path = config_dir / "test.json"
            path.write_text(json.dumps(raw))
            config = load_config(path)

            class StopAfterOne(TrainerCallback):
                def on_step_end(self, args, state, control, **kwargs):
                    if state.global_step == 1:
                        control.should_training_stop = True
                        control.should_save = True
                        control.should_evaluate = True
                    return control

            # The test must never depend on nvidia-smi availability on the host.
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), patch(
                "llm_training.training.audit.gpu_sample", return_value={"gpus": []}
            ):
                first = create_trainer(config)
                first.add_callback(StopAfterOne())
                run_training(config, first)
                output = Path(config["training"]["output_dir"])
                self.assertEqual(json.loads((output / "summary.json").read_text())["status"], "stopped")
                self.assertTrue((output / "checkpoint-1" / "COMPLETE").is_file())
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write('{"step":999,"loss":999}\n')
                second = create_trainer(config)
                run_training(config, second, resume=True)
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["status"], "completed")
                self.assertEqual(summary["last_step"], 2)
                self.assertFalse(any(json.loads(line).get("step") == 999
                                     for line in (output / "metrics.jsonl").read_text().splitlines()))
                sessions = sorted((output / "audit/sessions").glob("*"))
                self.assertEqual(len(sessions), 2)
                for session in sessions:
                    for name in ("config.json", "environment.json", "packages.txt", "trainer_plan.json",
                                 "dataset_stats.json", "events.jsonl", "progress.jsonl", "timing_totals.json"):
                        self.assertTrue((session / name).is_file(), name)
                self.assertTrue((output / "audit/metrics.csv").is_file())
                uninterrupted = copy.deepcopy(config)
                uninterrupted["training"]["output_dir"] = str(root / "uninterrupted")
                third = create_trainer(uninterrupted)
                run_training(uninterrupted, third)
                self.assertEqual(second.target_tokens_seen, third.target_tokens_seen)
                for actual, expected in zip(second.model.parameters(), third.model.parameters()):
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()

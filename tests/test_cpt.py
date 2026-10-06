"""Offline regression tests: data boundaries, loss, and an actual interrupted run."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    TrainerCallback,
    set_seed,
)

from llm_training.data.artifacts import packed_rows, prepare_artifacts
from llm_training.data.dataset import CausalLMCollator
from llm_training.training.monitoring import summed_nll
from llm_training.training.train import create_trainer, run_training


def fixture(root):
    set_seed(42)
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "a": 2, "b": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[EOS]"
    )
    model_path = root / "base"
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=4,
            n_positions=32,
            n_embd=16,
            n_layer=1,
            n_head=1,
            eos_token_id=1,
            pad_token_id=1,
        )
    )
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    archive = root / "corpus.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(
            "data.jsonl",
            "".join(
                json.dumps({"text": "a b " * (i + 2), "repo": f"repo-{i // 2}", "source": "code"})
                + "\n"
                for i in range(20)
            ),
        )
    return tokenizer, {
        "model": {"name_or_path": str(model_path), "trust_remote_code": False},
        "data": {
            "source": str(archive),
            "cache_dir": str(root / "cache"),
            "text_column": "text",
            "group_column": "repo",
            "source_column": "source",
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
            "max_length": 16,
            "seed": 42,
        },
        "training": {
            "output_dir": str(root / "run"),
            "max_steps": 4,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "learning_rate": 0.001,
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "eval_steps": 2,
            "save_steps": 2,
            "logging_steps": 1,
            "seed": 42,
            "report_to": "none",
            "use_cpu": True,
            "disable_tqdm": True,
            "dataloader_pin_memory": False,
        },
        "runtime": {"precision": "fp32", "allow_tf32": False},
    }


class StopAtTwo(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 2:
            control.should_training_stop = True
        return control


class CPTTests(unittest.TestCase):
    def test_packing_has_each_target_once(self):
        rows = list(packed_rows([{"input_ids": list(range(19))}], 6))
        self.assertEqual([x for row in rows for x in row["input_ids"][1:]], list(range(1, 19)))

    def test_group_manifest_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            tokenizer, config = fixture(Path(directory))
            dataset, artifact = prepare_artifacts(config["data"], tokenizer)
            manifest = Path(artifact["path"]) / "split_manifest.jsonl"
            groups = {}
            for line in manifest.read_text().splitlines():
                row = json.loads(line)
                groups.setdefault(row["group"], set()).add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
            for split, stats in artifact["statistics"].items():
                self.assertEqual(stats["target_tokens"], stats["tokens_with_eos"] - 1)
                self.assertGreater(len(dataset[split]), 0)
            _, cached = prepare_artifacts(config["data"], tokenizer)
            self.assertEqual(artifact["id"], cached["id"])

    def test_padding_eos_and_weighted_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            tokenizer, _ = fixture(Path(directory))
            batch = CausalLMCollator(tokenizer, None)(
                [
                    {"input_ids": [2, 1], "attention_mask": [1, 1]},
                    {"input_ids": [2, 3, 1], "attention_mask": [1, 1, 1]},
                ]
            )
            self.assertEqual(batch["labels"].tolist(), [[-100, 1, -100], [-100, 3, 1]])
            nll, count = summed_nll(torch.zeros(2, 3, 4), batch["labels"])
            self.assertEqual(count.tolist(), [1, 2])
            self.assertAlmostEqual(
                float(nll.sum() / count.sum()), float(torch.log(torch.tensor(4.0))), 6
            )

    def test_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            _, config = fixture(Path(directory))
            first = create_trainer(config)
            first.add_callback(StopAtTwo())
            run_training(config, first)
            checkpoint = Path(config["training"]["output_dir"]) / "checkpoint-2"
            self.assertTrue((checkpoint / "COMPLETE").exists())
            resumed = create_trainer(config)
            run_training(config, resumed, True)
            self.assertEqual(resumed.state.global_step, 4)
            reference = copy.deepcopy(config)
            reference["training"]["output_dir"] = str(Path(directory) / "reference")
            full = create_trainer(reference)
            run_training(reference, full)
            self.assertEqual(resumed.target_tokens_seen, full.target_tokens_seen)
            for left, right in zip(resumed.model.parameters(), full.model.parameters()):
                torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-6)
            evaluated = create_trainer(config, model_source=resumed.state.best_model_checkpoint)
            self.assertEqual(evaluated.dataset_artifact["id"], resumed.dataset_artifact["id"])
            result = evaluated.evaluate(
                eval_dataset=evaluated.test_dataset, metric_key_prefix="test"
            )
            self.assertGreater(result["test_target_tokens"], 0)
            incomplete = Path(config["training"]["output_dir"]) / "checkpoint-999"
            incomplete.mkdir()
            (incomplete / "config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Incomplete resume checkpoint"):
                run_training(config, create_trainer(config), str(incomplete))
            with self.assertRaisesRegex(ValueError, "already complete"):
                run_training(config, create_trainer(config), True)
            changed = copy.deepcopy(config)
            changed["training"]["learning_rate"] = 0.1
            with self.assertRaisesRegex(ValueError, "Resume rejected"):
                run_training(changed, create_trainer(changed), True)


if __name__ == "__main__":
    unittest.main()

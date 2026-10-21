"""Validate the direct Modal Notebook inputs without network or GPU access."""

import importlib.util
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/modal_notebook.py"
spec = importlib.util.spec_from_file_location("direct_notebook_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class NotebookTests(unittest.TestCase):
    def test_dataset_url_is_converted_to_repo_id(self):
        self.assertEqual(runner.dataset_repository("https://huggingface.co/datasets/iraprp/chayqwen"),
                         "iraprp/chayqwen")
        for url in ("https://example.com/datasets/owner/name",
                    "https://huggingface.co/datasets/owner/name/tree/main",
                    "https://huggingface.co/owner/model"):
            with self.assertRaises(ValueError):
                runner.dataset_repository(url)

    def test_direct_runner_uses_volume_paths_and_pinned_model(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            volume = Path(directory).resolve()
            pin = {"model_repo": "Qwen/Qwen3-1.7B-Base", "model_revision": "fixed-commit",
                   "data_repo": "iraprp/chayqwen", "data_revision": "data-commit"}
            (volume / "corpus_pin.json").write_text(json.dumps(pin))
            config = runner.configure(volume, "my-run")
            self.assertEqual(config["model"]["revision"], "fixed-commit")
            self.assertEqual(config["training"]["output_dir"], str(volume / "runs/my-run"))
            self.assertEqual(config["data"]["source"], str(volume / "corpus/data"))
            self.assertEqual(config["training"]["per_device_train_batch_size"], 16)
            self.assertEqual(config["training"]["gradient_accumulation_steps"], 2)
            source = volume / "my-existing-data"
            source.mkdir()
            pin.update(data_kind="local", data_path=str(source))
            (volume / "corpus_pin.json").write_text(json.dumps(pin))
            self.assertEqual(runner.configure(volume, "my-run")["data"]["source"], str(source))

    def test_local_directory_requires_a_real_nonempty_parameter(self):
        with self.assertRaises(ValueError):
            runner.local_data_directory("")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                runner.local_data_directory(str(Path(directory) / "missing"))
            self.assertEqual(runner.local_data_directory(directory), Path(directory).resolve())

    def test_local_preparation_does_not_download_dataset_and_remembers_source(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            root = Path(directory).resolve()
            volume = root / "run-volume"
            volume.mkdir()
            source = root / "another-volume" / "papers"
            source.mkdir(parents=True)
            (source / "part-01.jsonl").write_text('{"full_paper":"paper","source":"local"}\n')
            artifact = {"id": "fixed-artifact", "path": str(volume / "cache/prepared/fixed-artifact"),
                        "statistics": {"train": {"sequences": 16, "target_tokens": 100}}}
            hub = types.ModuleType("huggingface_hub")
            api = Mock()
            api.model_info.return_value.sha = "pinned-model-commit"
            hub.HfApi = Mock(return_value=api)
            hub.snapshot_download = Mock()
            artifacts = types.ModuleType("llm_training.data.artifacts")
            artifacts.prepare_artifacts = Mock(return_value=(None, artifact))
            train = types.ModuleType("llm_training.training.train")
            train.load_tokenizer = Mock(return_value=object())
            with patch.dict("sys.modules", {
                "huggingface_hub": hub,
                "llm_training.data.artifacts": artifacts,
                "llm_training.training.train": train,
            }):
                runner.prepare(volume, "my-run", data_path=str(source))
                runner.prepare(volume, "my-run", data_path=str(source))
            hub.snapshot_download.assert_not_called()
            api.repo_info.assert_not_called()
            self.assertEqual(api.model_info.call_count, 1)
            self.assertEqual(runner.configure(volume, "my-run")["data"]["source"], str(source))
            self.assertEqual(artifacts.prepare_artifacts.call_args.args[0]["source"], str(source))


if __name__ == "__main__":
    unittest.main()

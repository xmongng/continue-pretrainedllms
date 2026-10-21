"""Validate the direct Modal Notebook inputs without network or GPU access."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()

"""Audit, packing and step accounting checks that run without ML dependencies."""

import ast
import gzip
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from llm_training.training.audit import (
    ExperimentRecorder,
    bundle_audit,
    export_metrics,
    training_plan,
)
from llm_training.data.records import iter_json_records, training_text


class AuditTests(unittest.TestCase):
    def test_jsonl_reader_ignores_inconsistent_metadata_types(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "part.jsonl.gz"
            rows = [
                {"full_text": "first", "published": "2025-01-02T00:00:00"},
                {"full_text": "second", "published": ""},
                {"full_text": "third", "published": None},
            ]
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
            self.assertEqual(list(iter_json_records(path)), rows)

    def test_training_text_supports_explicit_schema_fallback(self):
        columns = ["full_text", "full_paper"]
        self.assertEqual(
            training_text({"full_text": "new"}, columns, "row 1"), ("new", "full_text")
        )
        self.assertEqual(
            training_text({"full_paper": "core"}, columns, "row 2"), ("core", "full_paper")
        )
        with self.assertRaisesRegex(ValueError, "expected one of"):
            training_text({"full_text": "", "title": "metadata"}, columns, "row 3")

    def test_step_rounding_and_explicit_budget(self):
        config = {
            "data": {"max_length": 2048},
            "training": {
                "per_device_train_batch_size": 16,
                "gradient_accumulation_steps": 2,
                "num_train_epochs": 2,
                "warmup_ratio": 0.01,
            },
        }
        stats = {"train": {"sequences": 35, "target_tokens": 70000}}
        plan = training_plan(config, stats)
        self.assertEqual(plan["microbatches_per_epoch"], 3)
        self.assertEqual(plan["expected_total_optimizer_steps"], 4)
        self.assertEqual(plan["full_step_target_tokens"], 65504)
        self.assertEqual(plan["warmup_optimizer_steps"], 1)
        config["training"]["max_steps"] = 7
        self.assertEqual(training_plan(config, stats)["expected_total_optimizer_steps"], 7)
        config["training"]["max_steps"] = -1
        config["training"]["dataloader_drop_last"] = True
        self.assertEqual(training_plan(config, stats)["expected_total_optimizer_steps"], 2)

    def test_sessions_keep_history_and_eta_ignores_same_step_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            first = ExperimentRecorder(directory)
            first.event("session_started", step=0)
            first.previous = (0, 0, 100.0)
            with patch("llm_training.training.audit.time.monotonic", return_value=110.0):
                first.progress(2, 200, 10)
            with patch("llm_training.training.audit.time.monotonic", return_value=115.0):
                first.progress(2, 200, 10, metrics={"eval_loss": 2})
            with patch("llm_training.training.audit.time.monotonic", return_value=120.0):
                first.progress(4, 400, 10)
            records = [
                json.loads(line)
                for line in (first.session / "progress.jsonl").read_text().splitlines()
            ]
            self.assertEqual(records[0]["eta_seconds"], 40)
            self.assertNotIn("eta_seconds", records[1])
            self.assertEqual(records[2]["interval_seconds"], 10)
            first.close()
            second = ExperimentRecorder(directory)
            self.assertNotEqual(first.session, second.session)
            self.assertTrue((first.session / "events.jsonl").exists())
            second.close()

    def test_system_sampling_and_csv_export(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ExperimentRecorder(directory)
            with patch("llm_training.training.audit.gpu_sample", return_value={"gpus": []}):
                recorder.sample_system()
            record = json.loads((recorder.session / "system_metrics.jsonl").read_text())
            self.assertIn("output_disk_free_bytes", record)
            self.assertIn("utc", record)
            (Path(directory) / "metrics.jsonl").write_text(
                '{"step":1,"loss":2.0}\n{"step":2,"eval_loss":1.9}\n'
            )
            export_metrics(directory)
            self.assertIn("eval_loss", (recorder.root / "metrics.csv").read_text())
            recorder.close()

    def test_packing_preserves_all_targets_at_boundaries(self):
        source = Path(__file__).resolve().parents[1] / "src/llm_training/data/artifacts.py"
        if not source.exists():  # Layout inside the Modal image.
            source = Path("/root/llm_training/data/artifacts.py")
        tree = ast.parse(source.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "packed_rows"
        )
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        documents = [{"input_ids": list(range(7))}, {"input_ids": list(range(7, 23))}]
        for length in (2, 4, 8, 23, 24):
            rows = list(namespace["packed_rows"](documents, length))
            targets = [token for row in rows for token in row["input_ids"][1:]]
            self.assertEqual(targets, list(range(1, 23)))
            self.assertTrue(all(len(row["input_ids"]) == length for row in rows[:-1]))

    def test_bundle_excludes_weights_and_checksums_exact_contents(self):
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            recorder = ExperimentRecorder(output)
            recorder.event("session_started", step=0)
            recorder.close()
            (output / "run.json").write_text('{"status":"running"}')
            checkpoint = output / "checkpoint-1"
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_bytes(b"not included")
            archive = bundle_audit(output)
            with zipfile.ZipFile(archive) as bundle:
                names = bundle.namelist()
                self.assertFalse(any("safetensors" in name for name in names))
                manifest = json.loads(bundle.read("bundle_manifest.json"))
                for name, expected in manifest["sha256"].items():
                    self.assertEqual(hashlib.sha256(bundle.read(name)).hexdigest(), expected)

    def test_snapshot_records_manifest_hash_and_does_not_dump_token_environment(self):
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "configs"
            config_dir.mkdir()
            config_path = config_dir / "run.json"
            config_path.write_text("{}")
            artifact_dir = root / "prepared"
            artifact_dir.mkdir()
            manifest = artifact_dir / "split_manifest.jsonl"
            manifest.write_text('{"split":"train"}\n')
            (artifact_dir / "identity.json").write_text('{"id":"fixed"}')
            config = {
                "config_path": str(config_path),
                "data": {"max_length": 2048},
                "training": {
                    "per_device_train_batch_size": 16,
                    "gradient_accumulation_steps": 2,
                    "num_train_epochs": 2,
                },
            }
            artifact = {
                "path": str(artifact_dir),
                "id": "fixed",
                "statistics": {"train": {"sequences": 32, "target_tokens": 65000}},
            }
            recorder = ExperimentRecorder(root / "output")
            with (
                patch.dict("os.environ", {"HF_TOKEN": "never-store-this-token"}),
                patch("llm_training.training.audit.gpu_sample", return_value={"gpus": []}),
            ):
                recorder.snapshot(config, artifact, {"artifact_id": "fixed"})
            reference = json.loads((recorder.session / "dataset_reference.json").read_text())
            self.assertEqual(
                reference["split_manifest_sha256"],
                hashlib.sha256(manifest.read_bytes()).hexdigest(),
            )
            environment = (recorder.session / "environment.json").read_text()
            self.assertNotIn("never-store-this-token", environment)
            for relative, digest in json.loads(
                (recorder.session / "source_sha256.json").read_text()
            ).items():
                self.assertEqual(
                    hashlib.sha256(
                        (recorder.session / "source" / relative).read_bytes()
                    ).hexdigest(),
                    digest,
                )
            recorder.close()


if __name__ == "__main__":
    unittest.main()

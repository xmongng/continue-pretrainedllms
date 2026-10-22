import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/watch_checkpoints.py'
spec = importlib.util.spec_from_file_location('checkpoint_review', SCRIPT)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class CheckpointReviewTests(unittest.TestCase):
    def test_regression_requires_three_evaluations(self):
        history = [{'eval_loss': x} for x in (2.0, 1.9, 1.94, 1.95, 1.96)]
        self.assertEqual(review.assess(history, 2.0), 'review_validation_regression')
        self.assertEqual(review.assess(history[:3], 2.0), 'no_domain_loss_alert')

    def test_archive_only_complete_checkpoint_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'checkpoint-5000'
            checkpoint.mkdir()
            state = {'global_step': 5000, 'max_steps': 80000, 'epoch': 0.12,
                     'log_history': [{'step': 5000, 'eval_loss': 1.9}]}
            (checkpoint / 'trainer_state.json').write_text(json.dumps(state))
            (checkpoint / 'model.safetensors').write_bytes(b'fixture weights')
            (checkpoint / 'optimizer.pt').write_bytes(b'not archived')
            review.poll(root, 5000, 0.02, 3)
            self.assertFalse((root / 'checkpoint_review/step-5000.json').exists())
            (checkpoint / 'COMPLETE').touch()
            review.poll(root, 5000, 0.02, 3)
            archive = root / 'checkpoint_review/models/checkpoint-5000'
            self.assertTrue((archive / 'ARCHIVE_COMPLETE.json').exists())
            self.assertFalse((archive / 'optimizer.pt').exists())
            review.poll(root, 5000, 0.02, 3)

    def test_new_best_does_not_create_extra_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'checkpoint-1000'
            checkpoint.mkdir()
            (checkpoint / 'COMPLETE').touch()
            state = {'global_step': 1000, 'max_steps': 80000, 'epoch': 0.02,
                     'log_history': [{'step': 1000, 'eval_loss': 1.9}]}
            (checkpoint / 'trainer_state.json').write_text(json.dumps(state))
            review.poll(root, 5000, 0.02, 3)
            record = json.loads((root / 'checkpoint_review/step-1000.json').read_text())
            self.assertTrue(record['is_best'])
            self.assertFalse(record['model_archived'])
            self.assertFalse((root / 'checkpoint_review/models/checkpoint-1000').exists())


if __name__ == '__main__':
    unittest.main()

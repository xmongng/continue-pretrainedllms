# Checkpoint review during training

Run `python -u scripts/watch_checkpoints.py --output /path/to/run` alongside
the existing training process. It polls every 30 seconds. Start it again after
each session restart. It cannot recover checkpoints already removed by Trainer.

Training already evaluates domain validation every 1000 steps and at epoch/end
boundaries. Trainer retains the best and latest full resumable checkpoints.
The observer reads those evaluation results from completed checkpoint states;
it does not run additional GPU inference or access the test set.

Every observed completed checkpoint gets `checkpoint_review/step-N.json`.
Model snapshots are copied into `checkpoint_review/models` whenever validation
sets a new best (including ties), every 5000 steps, and at epoch/final boundaries.
Snapshots include model/tokenizer files and SHA-256 checksums, but exclude
optimizer state: use them for evaluation or SFT, not training resume.
Archives are never rotated; with improving loss this can retain every checkpoint.
Budget disk space accordingly (FP32 1.7B weights are roughly 6.8 GB per snapshot).

`above_baseline` flags loss more than 0.02 above baseline.
`review_validation_regression` flags three successive evaluations more than
0.02 above the best preceding those evaluations. These are configurable review
thresholds, not statistical tests. No automatic stop or claim of general quality.
Inspect errors in the watcher console, including insufficient disk space.

Domain loss cannot detect all forgetting or predict SFT quality. Compare selected
archives against the original model on a fixed independent general validation
set, and compare identical pilot SFT runs on held-out downstream tasks. Exact-text
split checks do not exclude near-duplicate or DOI-level leakage.

This script changes neither training configuration nor the hashed training source
or launcher, so it can accompany the existing run without altering its resume
contract. Existing package/runtime requirements still apply when resuming.

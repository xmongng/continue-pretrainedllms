"""Monitor completed CPT checkpoints without modifying the training/resume contract."""

import argparse
import hashlib
import json
import math
import shutil
import tempfile
import time
from pathlib import Path


def assess(history, baseline, margin=0.02, patience=3):
    values = [float(row['eval_loss']) for row in history]
    latest = values[-1]
    if not all(math.isfinite(x) for x in values):
        return 'nonfinite_validation'
    if len(values) >= patience + 1:
        reference = min(values[:-patience])
        if all(x > reference + margin for x in values[-patience:]):
            return 'review_validation_regression'
    if baseline is not None and latest > baseline + margin:
        return 'above_baseline'
    return 'no_domain_loss_alert'


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def archive_model(checkpoint, destination):
    if (destination / 'ARCHIVE_COMPLETE.json').exists():
        return
    files = [p for p in checkpoint.iterdir() if p.is_file() and (
        p.suffix in ('.json', '.safetensors', '.txt', '.model', '.tiktoken')
        or p.name == 'pytorch_model.bin'
    )]
    if not any(p.suffix == '.safetensors' or p.name == 'pytorch_model.bin' for p in files):
        raise ValueError(f'No model weights: {checkpoint}')
    required = sum(p.stat().st_size for p in files)
    if shutil.disk_usage(destination.parent).free < required + 5 * 1024**3:
        raise RuntimeError('Insufficient space for model archive plus 5 GiB reserve')
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.archiving-') as tmp:
        work = Path(tmp)
        hashes = {}
        for source in files:
            target = work / source.name
            shutil.copy2(source, target)
            digest = hashlib.sha256()
            with target.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b''):
                    digest.update(block)
            hashes[source.name] = digest.hexdigest()
        if not (checkpoint / 'COMPLETE').exists():
            raise RuntimeError('Source checkpoint rotated during archive; retry next checkpoint')
        save_json(work / 'ARCHIVE_COMPLETE.json', {
            'source': str(checkpoint), 'sha256': hashes,
            'purpose': 'model for evaluation/SFT; not optimizer-resumable',
        })
        work.rename(destination)


def poll(output, every, margin, patience):
    root = output / 'checkpoint_review'
    root.mkdir(exist_ok=True)
    archives = root / 'models'
    archives.mkdir(exist_ok=True)
    baseline_path = output / 'baseline.json'
    baseline = json.loads(baseline_path.read_text()).get('baseline_loss') if baseline_path.exists() else None
    checkpoints = sorted(output.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
    for checkpoint in checkpoints:
        if not (checkpoint / 'COMPLETE').exists():
            continue
        state = json.loads((checkpoint / 'trainer_state.json').read_text())
        step = int(state['global_step'])
        report_path = root / f'step-{step}.json'
        if report_path.exists():
            continue
        history = [row for row in state['log_history'] if 'eval_loss' in row]
        if not history or history[-1]['step'] != step:
            raise ValueError(f'No matching validation metrics at step {step}')
        status = assess(history, baseline, margin, patience)
        latest = float(history[-1]['eval_loss'])
        is_best = latest <= min(float(row['eval_loss']) for row in history)
        epoch = state.get('epoch')
        epoch_end = epoch is not None and abs(epoch - round(epoch)) < 1e-8
        keep = is_best or step % every == 0 or epoch_end or step >= state['max_steps']
        if keep:
            archive_model(checkpoint, archives / checkpoint.name)
        record = {
            'step': step, 'epoch': epoch, 'eval_loss': latest,
            'baseline_loss': baseline, 'status': status, 'model_archived': keep,
            'archive': str(archives / checkpoint.name) if keep else None,
            'scope': 'domain validation only; general capability and SFT quality untested',
        }
        save_json(report_path, record)
        print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--every', type=int, default=5000)
    parser.add_argument('--margin', type=float, default=0.02)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if not args.output.is_dir() or args.every < 1 or args.patience < 1 or args.margin < 0:
        parser.error('Require existing output, positive every/patience and nonnegative margin')
    while True:
        try:
            poll(args.output, args.every, args.margin, args.patience)
        except (OSError, ValueError, RuntimeError) as error:
            print(f'[checkpoint-review ERROR] {error}', flush=True)
            if args.once:
                raise
        if args.once:
            return
        time.sleep(30)


if __name__ == '__main__':
    main()

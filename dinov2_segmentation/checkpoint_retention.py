"""Retain each progress generation and durably enqueue selected generations."""
import json
import os
from pathlib import Path

from .validate_progress_checkpoints import _atomic_json, capture_progress_checkpoint


def retain_progress(path: Path, state: dict, monitor_interval: int) -> Path:
    path = Path(path)
    step = int(state['curriculum_step'])
    archive = path.parent / 'checkpoint_archive'
    archive.mkdir(exist_ok=True)
    destination = archive / f'checkpoint_step_{step:09d}.pt'
    # Never replace an existing generation for the same step.
    if destination.exists():
        if not os.path.samefile(path, destination):
            raise RuntimeError(f'Conflicting retained checkpoint: {destination}')
    else:
        os.link(path, destination)
    _atomic_json({
        'archive': str(destination), 'curriculum_step': step,
        'epoch': int(state['epoch']),
        'next_batch_index': int(state['next_batch_index']),
        'epoch_complete': bool(state['epoch_complete']),
        'monitor_due': bool(monitor_interval and step % monitor_interval == 0),
    }, destination.with_suffix('.json'))
    if monitor_interval and step % monitor_interval == 0:
        capture_progress_checkpoint(path)
    return destination


def validate_resume_configuration(source: dict, target: dict) -> dict:
    source, target = dict(source), dict(target)
    operational = {'checkpoint_interval_steps'}
    # These two samplers select the same quota-controlled population.  Moving
    # from the global ordering to WSI-local ordering changes only the target
    # batch arrangement, so it may resume model/optimizer/scheduler state at a
    # saved batch boundary.  Dataset, quotas and all optimization parameters
    # remain strict resume invariants.
    balanced_sampling = {'slide_stratified', 'wsi_local_stratified'}
    if (source.get('sampling_mode') in balanced_sampling
            and target.get('sampling_mode') in balanced_sampling):
        operational.update({'sampling_mode', 'sampling_locality_tile_size'})
    if (source.get('graph_feature_policy') == 'staged_consistent'
            and target.get('graph_feature_policy') == 'staged_consistent'):
        source.setdefault('cache_frozen_dino_prefix', False)
        target.setdefault('cache_frozen_dino_prefix', False)
        operational.add('cache_frozen_dino_prefix')
    differences = {k: {'source': source.get(k), 'target': target.get(k)}
                   for k in source.keys() | target.keys()
                   if source.get(k) != target.get(k)}
    if set(source) != set(target) or set(differences) - operational:
        raise ValueError('Resume checkpoint configuration differs')
    return differences


def queue_epoch_validation(path: Path, state: dict) -> Path:
    """Retain the exact epoch-end generation before publishing its request."""
    path = Path(path)
    epoch, step = int(state['epoch']), int(state['curriculum_step'])
    if not state['epoch_complete']:
        raise ValueError('Epoch validation requires an epoch-complete checkpoint')
    archive = path.parent / 'checkpoint_archive'
    archive.mkdir(exist_ok=True)
    destination = archive / f'checkpoint_epoch_{epoch:03d}_step_{step:09d}.pt'
    if destination.exists():
        if not os.path.samefile(path, destination):
            raise RuntimeError(f'Conflicting epoch checkpoint: {destination}')
    else:
        os.link(path, destination)
    request = path.parent / 'full_validation' / f'candidate_epoch_{epoch:03d}_step_{step:09d}.json'
    _atomic_json({'epoch': epoch, 'curriculum_step': step, 'path': str(destination),
                  'reason': 'epoch_end'}, request)
    return request


def completed_epoch_validation_state(run_dir, history, *, start_epoch, min_delta, patience):
    """Recompute early stopping from the contiguous completed prefix; never wait.

    Candidate sweeps cannot count as extra epochs. Missing/failed/out-of-order
    epoch results do not increment patience or cause a false early stop.
    """
    best = reference = -1.0
    misses = 0
    last_evaluated = -1
    for row in history:
        epoch = int(row['epoch'])
        metrics = row.get('val')
        if metrics is None:
            step = int(row['curriculum_step_end'])
            request = Path(run_dir) / 'full_validation' / f'candidate_epoch_{epoch:03d}_step_{step:09d}.json'
            result = request.with_suffix('.done.json')
            if not request.exists() or not result.exists():
                break
            job = json.loads(request.read_text())
            result = json.loads(result.read_text())
            if (result.get('status') != 'complete' or result.get('curriculum_step') != step
                    or result.get('checkpoint_path') != job['path']):
                break
            metrics = result['metrics']
        score = float(metrics['tumor_dice'])
        best = max(best, score)
        if epoch >= start_epoch and patience > 0:
            if score > reference + min_delta:
                reference, misses = score, 0
            else:
                misses += 1
        else:
            reference = max(reference, score)
        last_evaluated = epoch
        if patience > 0 and epoch >= start_epoch and misses >= patience:
            break
    return dict(best_dice=best, reference_dice=reference,
                epochs_without_improvement=misses, last_evaluated_epoch=last_evaluated,
                stopped_early=bool(patience > 0 and last_evaluated >= start_epoch and misses >= patience))

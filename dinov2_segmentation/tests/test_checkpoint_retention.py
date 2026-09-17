import json
import os
from pathlib import Path
import pytest

from dinov2_segmentation.checkpoint_retention import retain_progress, validate_resume_configuration
from dinov2_segmentation.validate_progress_checkpoints import _discover_progress, _discover_spool


def test_all_generations_retained_but_only_even_thousands_queued(tmp_path):
    source = tmp_path / 'checkpoint_progress.pt'
    (tmp_path / 'checkpoint_queue_writer.json').write_text('{}')
    for step in (1000, 2000, 3000, 4000):
        temp = tmp_path / 'new.pt'
        temp.write_bytes(str(step).encode())
        os.replace(temp, source)
        retain_progress(source, {'curriculum_step': step, 'epoch': 2,
                                'next_batch_index': step, 'epoch_complete': False}, 2000)
    assert len(list((tmp_path / 'checkpoint_archive').glob('*.pt'))) == 4
    assert sorted(p.read_bytes() for p in _discover_spool(tmp_path)) == [b'2000', b'4000']
    assert not list(_discover_progress(tmp_path))
    # A late consumer still sees the old immutable generation after overwrites.
    assert (tmp_path / 'checkpoint_archive/checkpoint_step_000001000.pt').read_bytes() == b'1000'
    retain_progress(source, {'curriculum_step': 4000, 'epoch': 2,
                            'next_batch_index': 4000, 'epoch_complete': False}, 2000)
    assert len(_discover_spool(tmp_path)) == 2


def test_resume_allows_only_save_frequency_change():
    before = {'checkpoint_interval_steps': 20000, 'batch_size': 32, 'decoder_lr': 1e-4}
    after = dict(before, checkpoint_interval_steps=1000)
    assert set(validate_resume_configuration(before, after)) == {'checkpoint_interval_steps'}
    for key, value in [('batch_size', 64), ('decoder_lr', 2e-4)]:
        with pytest.raises(ValueError):
            validate_resume_configuration(before, dict(after, **{key: value}))


def test_resume_allows_balanced_wsi_local_reordering_only():
    before = {
        'checkpoint_interval_steps': 1000,
        'batch_size': 32,
        'sampling_mode': 'slide_stratified',
        'sampling_locality_tile_size': 4096,
    }
    after = dict(before, sampling_mode='wsi_local_stratified',
                 sampling_locality_tile_size=2048)
    assert set(validate_resume_configuration(before, after)) == {
        'sampling_mode', 'sampling_locality_tile_size',
    }
    with pytest.raises(ValueError):
        validate_resume_configuration(before, dict(after, batch_size=64))
    with pytest.raises(ValueError):
        validate_resume_configuration(
            dict(before, sampling_mode='uniform'), after
        )


def test_retention_refuses_conflicting_generation(tmp_path):
    source = tmp_path / 'checkpoint_progress.pt'
    state = {'curriculum_step': 1000, 'epoch': 0, 'next_batch_index': 1000, 'epoch_complete': False}
    source.write_bytes(b'first')
    retained = retain_progress(source, state, 2000)
    replacement = tmp_path / 'new.pt'
    replacement.write_bytes(b'other'); os.replace(replacement, source)
    with pytest.raises(RuntimeError):
        retain_progress(source, state, 2000)
    assert retained.read_bytes() == b'first'

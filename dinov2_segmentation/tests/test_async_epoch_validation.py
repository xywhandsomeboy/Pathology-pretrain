"""Independent epoch validation must never run or wait inside the trainer."""
from datetime import timedelta
import json
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dinov2_segmentation.checkpoint_retention import queue_epoch_validation,completed_epoch_validation_state
from dinov2_segmentation.tests.test_step_curriculum import _CurriculumSystem,_trainer_args
from dinov2_segmentation.tests.test_parallel_training import _prepare_sources
from dinov2_segmentation.distributed_execution import DistributedExecution


def exercise(root,mode,execution):
    from dinov2_segmentation import train_joint
    train_joint.JointSegmentationSystem=_CurriculumSystem
    original_loader=train_joint._loader
    original_epoch=train_joint._run_epoch
    def loader(*a,**kw):
        assert kw['training'],'Trainer constructed a validation loader'
        return original_loader(*a,**kw)
    def epoch(*a,**kw):
        assert kw.get('optimizer') is not None,'Trainer ran validation'
        return original_epoch(*a,**kw)
    train_joint._loader=loader;train_joint._run_epoch=epoch
    try:
        args=_trainer_args(root,mode,root/'run',('--epochs','2','--async-full-validation'))
        train_joint.main(args=args,execution=execution)
    finally:
        train_joint._loader=original_loader;train_joint._run_epoch=original_epoch


def check(root):
    run=root/'run';history=json.loads((run/'history.json').read_text())
    assert len(history)==2 and all(r['val'] is None for r in history)
    assert all(r['validation_status']=='queued_independent_process' for r in history)
    requests=sorted((run/'full_validation').glob('candidate_epoch_*.json'))
    assert len(requests)==2
    for i,p in enumerate(requests):
        request=json.loads(p.read_text());checkpoint=torch.load(request['path'],map_location='cpu',weights_only=False)
        assert request['epoch']==i and checkpoint['epoch']==i and checkpoint['epoch_complete']
        assert checkpoint['curriculum_step']==5*(i+1)
    assert not (run/'checkpoint_best.pt').exists()
    assert (run/'complete').exists() # Neither queued result was needed to continue.


def test_serial_two_epochs_never_construct_or_run_validation(tmp_path,monkeypatch):
    from dinov2_segmentation import train_joint
    torch.set_num_threads(1);_prepare_sources(tmp_path,train_count=5)
    monkeypatch.setattr(train_joint,'JointSegmentationSystem',_CurriculumSystem)
    with DistributedExecution('cpu') as execution:exercise(tmp_path,'serial',execution)
    check(tmp_path)


def worker(rank,root,rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2,timeout=timedelta(seconds=60))
    execution=DistributedExecution('cpu',rank=rank,world_size=2,owns_process_group=True)
    try:exercise(Path(root),'ddp',execution)
    finally:execution.close()


def test_ddp_two_epochs_do_not_wait_for_validation(tmp_path):
    _prepare_sources(tmp_path,train_count=10)
    mp.spawn(worker,args=(str(tmp_path),(tmp_path/'rendezvous').as_uri()),nprocs=2,join=True)
    check(tmp_path)


def test_out_of_order_results_do_not_advance_early_stopping(tmp_path):
    history=[{'epoch':0,'val':{'tumor_dice':.9}}]
    for epoch in (1,2):
        state={'epoch':epoch,'curriculum_step':epoch*100,'epoch_complete':True}
        source=tmp_path/f'epoch{epoch}.pt';source.write_bytes(str(epoch).encode())
        request=queue_epoch_validation(source,state)
        history.append({'epoch':epoch,'curriculum_step_end':epoch*100,'val':None})
        if epoch==2:
            job=json.loads(request.read_text())
            request.with_suffix('.done.json').write_text(json.dumps({'status':'complete','curriculum_step':epoch*100,'checkpoint_path':job['path'],'metrics':{'tumor_dice':.8}}))
    result=completed_epoch_validation_state(tmp_path,history,start_epoch=1,min_delta=.001,patience=2)
    assert result['last_evaluated_epoch']==0 and result['epochs_without_improvement']==0
    request=tmp_path/'full_validation/candidate_epoch_001_step_000000100.json';job=json.loads(request.read_text())
    request.with_suffix('.done.json').write_text(json.dumps({'status':'complete','curriculum_step':100,'checkpoint_path':job['path'],'metrics':{'tumor_dice':.85}}))
    result=completed_epoch_validation_state(tmp_path,history,start_epoch=1,min_delta=.001,patience=2)
    assert result['stopped_early'] and result['last_evaluated_epoch']==2

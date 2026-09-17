"""Real joint-loop, checkpoint, phase and Gloo DDP tests for V4/V5."""
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest

from dinov2_segmentation.tests.test_step_curriculum import _FourBlockStage1, _trainer_args
from dinov2_segmentation.tests.test_parallel_training import TinyGNN, _prepare_sources
from dinov2_segmentation.distributed_execution import DistributedExecution
from dinov2_segmentation.models.model_v4 import GlobalLocalSegmentationModelV4
from dinov2_segmentation.models.model_v5 import GlobalLocalSegmentationModelV5


class AuxStage1(_FourBlockStage1):
    def forward(self, images):
        nodes, dense=super().forward(images)
        return nodes,dense[:,None,:]


class AuxSystem(nn.Module):
    def __init__(self,decoder_version,**kwargs):
        super().__init__();self.decoder_version=decoder_version
        self.model_version='joint_test_'+decoder_version
        self.stage1=AuxStage1();self.stage2=TinyGNN()
        self.stage2_runtime=SimpleNamespace(context_edge_mode='distance',num_layers=1,use_edge_attr=False)
        common=dict(num_classes=2,token_dim=3,context_dim=3,channels=8,diffusion_channels=8,diffusion_steps=16)
        if decoder_version=='v4':
            self.decoder=GlobalLocalSegmentationModelV4(**common,detail_depth=2,fusion_depth=2,num_heads=2,window_size=2,drop_path_rate=0.)
        else:
            self.decoder=GlobalLocalSegmentationModelV5(**common,high_resolution_depth=1,correction_depth=1,max_upsample_stages=4)

    def decode(self,images,dense,contexts):
        from dinov2_segmentation.joint_model import JointSegmentationSystem
        return JointSegmentationSystem.decode(self,images,dense,contexts)


def check_result(directory):
    history=json.loads((directory/'history.json').read_text())
    train=history[-1]['train'];valid=history[-1]['val']
    assert train['diffusion_aux_loss']>0
    assert train['loss']==pytest.approx(train['segmentation_loss']+.1*train['diffusion_aux_loss'],rel=1e-5)
    assert valid['loss']==valid['segmentation_loss']
    assert 'diffusion_aux_loss' not in valid  # Not a misleading zero evaluation.
    assert 0<=valid['boundary_f1']<=1
    c=torch.load(directory/'checkpoint_last.pt',map_location='cpu',weights_only=False)
    assert c['curriculum_step']==5
    assert c['configuration']['diffusion_loss_weight']==.1
    assert any('diffusion.' in key for key in c['model'])
    assert c['optimizer']['state']


@pytest.mark.parametrize('version',['v4','v5'])
def test_serial_auxiliary_checkpoint_resume_without_replayed_updates(tmp_path,monkeypatch,version):
    from dinov2_segmentation import train_joint
    torch.set_num_threads(1);_prepare_sources(tmp_path,train_count=5)
    monkeypatch.setattr(train_joint,'JointSegmentationSystem',AuxSystem)
    directory=tmp_path/version
    args=_trainer_args(tmp_path,'serial',directory,('--decoder-version',version))
    original=train_joint._atomic_torch_save
    class Saved(Exception):pass
    def save(state,path):
        original(state,path)
        if Path(path).name=='checkpoint_progress.pt':raise Saved()
    monkeypatch.setattr(train_joint,'_atomic_torch_save',save)
    with DistributedExecution('cpu') as execution:
        with pytest.raises(Saved):train_joint.main(args=args,execution=execution)
    monkeypatch.setattr(train_joint,'_atomic_torch_save',original)
    resumed=_trainer_args(tmp_path,'serial',directory,('--decoder-version',version,'--resume',str(directory/'checkpoint_progress.pt')))
    with DistributedExecution('cpu') as execution:train_joint.main(args=resumed,execution=execution)
    check_result(directory)


def ddp_worker(rank,root,rendezvous,version):
    from dinov2_segmentation import train_joint
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2,timeout=timedelta(seconds=90))
    execution=DistributedExecution('cpu',rank=rank,world_size=2,owns_process_group=True)
    train_joint.JointSegmentationSystem=AuxSystem
    try:
        args=_trainer_args(Path(root),'ddp',Path(root)/version,('--decoder-version',version))
        train_joint.main(args=args,execution=execution)
    finally:execution.close()


@pytest.mark.parametrize('version',['v4','v5'])
def test_two_rank_ddp_auxiliary_with_all_unfreeze_phases(tmp_path,version):
    _prepare_sources(tmp_path,train_count=10)
    mp.spawn(ddp_worker,args=(str(tmp_path),(tmp_path/'rendezvous').as_uri(),version),nprocs=2,join=True)
    check_result(tmp_path/version)


class ScalarNoise(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.tensor(.3))
    def forward(self,x):return x*self.weight


def synthetic_output(predicted,noise):
    return {'predicted_noise':predicted,'noise':noise,'reconstruction':-predicted,
            'clean':torch.zeros_like(predicted),'sqrt_alpha':predicted.new_full((predicted.shape[0],1,1,1),.6)}


def loss_equivalence_worker(rank,root,rendezvous):
    from dinov2_segmentation.diffusion_losses import diffusion_auxiliary_loss
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2,timeout=timedelta(seconds=60))
    try:
        x=torch.arange(3*3*8*8).reshape(3,3,8,8).float()/100
        noise=torch.sin(x);target=torch.zeros(3,8,8,dtype=torch.long);target[:,:,4:]=1
        section=slice(0,1) if rank==0 else slice(1,3)
        ddp=torch.nn.parallel.DistributedDataParallel(ScalarNoise())
        predicted=ddp(x[section])
        loss,_=diffusion_auxiliary_loss(synthetic_output(predicted,noise[section]),target[section],distributed=True)
        loss.backward()
        reference=ScalarNoise()
        ref_loss,_=diffusion_auxiliary_loss(synthetic_output(reference(x),noise),target)
        ref_loss.backward()
        torch.testing.assert_close(loss,ref_loss)
        torch.testing.assert_close(ddp.module.weight.grad,reference.weight.grad)
    finally:dist.destroy_process_group()


def test_ddp_auxiliary_loss_and_gradient_match_serial_unequal_shards(tmp_path):
    mp.spawn(loss_equivalence_worker,args=(str(tmp_path),(tmp_path/'loss_rendezvous').as_uri()),nprocs=2,join=True)

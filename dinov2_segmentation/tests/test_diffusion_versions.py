"""Functional checks of V4/V5 and boundary supervision, CPU only."""
from pathlib import Path
import sys
from unittest import mock
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'dinov2_stage2_2_FmH2ST'))
from dinov2_segmentation.models.model_v4 import GlobalLocalSegmentationModelV4
from dinov2_segmentation.models.model_v5 import GlobalLocalSegmentationModelV5
from dinov2_segmentation.diffusion_losses import (boundary_weights, boundary_counts,
                                                 boundary_scores, diffusion_auxiliary_loss)


def make_model(version):
    common=dict(num_classes=2, token_dim=16, context_dim=12, channels=8,
                diffusion_channels=8, diffusion_steps=16)
    if version=='v4':
        return GlobalLocalSegmentationModelV4(**common, detail_depth=2, fusion_depth=2,
                                               num_heads=2, window_size=2, drop_path_rate=0.)
    return GlobalLocalSegmentationModelV5(**common, high_resolution_depth=1,
                                          correction_depth=1, max_upsample_stages=4)


def inputs():
    return torch.randn(1,3,16,16),torch.randn(1,4,16,requires_grad=True),torch.randn(1,12)


@pytest.mark.parametrize('version',['v4','v5'])
def test_segmentation_is_base_exact_and_eval_consumes_no_diffusion_rng(version):
    torch.set_num_threads(1);model=make_model(version).eval(); x=inputs()
    before=torch.random.get_rng_state().clone()
    with torch.no_grad():
        logits=model(*x); base=model.segmentation(*x)
    assert torch.equal(before,torch.random.get_rng_state())
    torch.testing.assert_close(logits,base,rtol=0,atol=0)
    assert logits.shape==(1,2,16,16)


@pytest.mark.parametrize('version',['v4','v5'])
def test_auxiliary_alone_reaches_shared_features_and_segmentation_head(version):
    torch.manual_seed(14);torch.set_num_threads(1);model=make_model(version);x=inputs()
    output=model(*x,return_auxiliary=True,timesteps=torch.tensor([8]),noise=torch.randn(1,3,16,16))
    mask=torch.zeros(1,16,16,dtype=torch.long);mask[:,:,8:]=1
    loss,parts=diffusion_auxiliary_loss(output['diffusion'],mask)
    loss.backward()
    assert x[1].grad is not None and x[1].grad.abs().sum()>0
    assert model.diffusion.output.weight.grad.abs().sum()>0
    # The condition includes predicted probabilities, so auxiliary gradients
    # also reach the segmentation output head (not just a parallel generator).
    heads=[p.grad for n,p in model.segmentation.named_parameters() if ('head' in n or 'classifier' in n)]
    assert heads and any(g is not None and g.abs().sum()>0 for g in heads)
    assert all(torch.isfinite(v) for v in parts.values())


@pytest.mark.parametrize('version',['v4','v5'])
def test_sampling_and_checkpoint_roundtrip(version):
    torch.set_num_threads(1);model=make_model(version).eval();x=inputs()
    restored=make_model(version).eval();restored.load_state_dict(model.state_dict(),strict=True)
    a=model.generate_reconstruction(*x,sampling_steps=3,seed=7)
    b=restored.generate_reconstruction(*x,sampling_steps=3,seed=7)
    assert a['reconstruction'].shape==(1,3,16,16)
    assert torch.isfinite(a['reconstruction']).all()
    assert a['reconstruction'].min()>=0 and a['reconstruction'].max()<=1
    torch.testing.assert_close(a['reconstruction'],b['reconstruction'])


def test_boundary_weighting_ignores_unlabeled_pixels_and_constant_regions():
    target=torch.zeros(1,12,12,dtype=torch.long);target[:,:,6:]=1;target[:,0,:]=255
    weights,valid=boundary_weights(target,boost=4,radius=1)
    assert weights[:,0,:].sum()==0
    assert weights[0,5,5]==5 and weights[0,5,0]==1
    constant=torch.ones(1,12,12,dtype=torch.long)
    torch.testing.assert_close(boundary_weights(constant)[0],torch.ones_like(constant).float())
    logits=torch.nn.functional.one_hot(target.masked_fill(~valid,0),2).permute(0,3,1,2).float()
    assert boundary_scores(boundary_counts(logits,target))['boundary_f1']==1
    shifted=torch.zeros_like(target);shifted[:,:,10:]=1
    shifted_logits=torch.nn.functional.one_hot(shifted,2).permute(0,3,1,2).float()
    assert boundary_scores(boundary_counts(shifted_logits,target,tolerance=1))['boundary_f1']<1


def test_known_noise_reconstruction_and_all_ignored_loss():
    model=make_model('v4');module=model.diffusion
    clean=torch.rand(2,3,8,8)*2-1;noise=torch.randn_like(clean);t=torch.tensor([0,12])
    alpha=module.alpha_bar[t][:,None,None,None]
    noisy=module.diffuse(clean,t,noise)
    reconstructed=(noisy-(1-alpha).sqrt()*noise)/alpha.sqrt()
    torch.testing.assert_close(reconstructed,clean,atol=1e-5,rtol=1e-5)
    prediction=noise.clone().requires_grad_()
    output=dict(predicted_noise=prediction,noise=noise,reconstruction=reconstructed,
                clean=clean,sqrt_alpha=alpha.sqrt())
    loss,_=diffusion_auxiliary_loss(output,torch.full((2,8,8),255))
    assert loss==0 and torch.isfinite(loss)
    loss.backward();assert prediction.grad.abs().sum()==0


def test_edge_error_is_penalized_more_than_same_interior_error():
    target=torch.zeros(1,16,16,dtype=torch.long);target[:,:,8:]=1
    zero=torch.zeros(1,3,16,16)
    def measure(column):
        predicted=zero.clone();predicted[:,:,:,column]=1
        data=dict(predicted_noise=predicted,noise=zero,reconstruction=zero,
                  clean=zero,sqrt_alpha=torch.ones(1,1,1,1))
        return diffusion_auxiliary_loss(data,target)[0]
    assert measure(8)>measure(1)


def test_factories_and_version_entry_points():
    from dinov2_segmentation.joint_model import JointSegmentationSystem
    class Stage1(nn.Module):
        def __init__(self,*a):super().__init__();self.embed_dim=16
    for version,kind in [('v4',GlobalLocalSegmentationModelV4),('v5',GlobalLocalSegmentationModelV5)]:
        with mock.patch('dinov2_segmentation.joint_model.TrainableStage1',Stage1),mock.patch(
            'dinov2_segmentation.joint_model.build_trainable_stage2',return_value=(nn.Identity(),object())):
            system=JointSegmentationSystem(decoder_version=version,stage1_config='x',stage1_checkpoint='x',stage2_config='x',stage2_checkpoint='x')
        assert isinstance(system.decoder,kind)
        assert system.model_version.endswith(version)
        module=__import__('dinov2_segmentation.train_joint_'+version,fromlist=['main'])
        with mock.patch.object(module,'joint_main') as call:
            module.main(['--execution-mode','ddp'])
            call.assert_called_once_with(['--decoder-version',version,'--execution-mode','ddp'])

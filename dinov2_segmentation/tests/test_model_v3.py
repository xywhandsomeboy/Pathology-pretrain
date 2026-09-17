"""V3 attention dataflow, gradients, checkpoint isolation, and V2 regression."""
import sys
from pathlib import Path
from unittest import mock
import torch
from torch import nn
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'dinov2_stage2_2_FmH2ST'))
from dinov2_segmentation.models.model_v3 import GlobalLocalSegmentationModelV3
from dinov2_segmentation.models.v3_decoder import SemanticDetailCrossAttention
from dinov2_segmentation.joint_model import JointSegmentationSystem


def tiny():
    return GlobalLocalSegmentationModelV3(2, token_dim=16, context_dim=12, channels=8,
                                          detail_depth=1, cross_attention_depth=2,
                                          heads=2, query_grid_size=8, kv_grid_size=4)


def test_output_resolution_rectangular_inputs_and_attention_bounds():
    torch.set_num_threads(1)
    model = tiny()
    seen = []
    def record(_module, args):
        seen.append((args[0].shape[1], args[1].shape[1], args[2].shape[1]))
    hooks = [b.attention.register_forward_pre_hook(record) for b in model.decoder.cross_blocks]
    output = model(torch.randn(2,3,33,29), torch.randn(2,4,16), torch.randn(2,12), True)
    assert output['logits'].shape == (2,2,33,29)
    assert seen == [(64,16,16), (64,16,16)]
    for h in hooks:h.remove()


def test_gradients_reach_qkv_detail_tokens_and_context_affine():
    torch.manual_seed(12); torch.set_num_threads(1)
    model=tiny()
    tokens=torch.randn(1,4,16,requires_grad=True)
    logits=model(torch.randn(1,3,32,32), tokens, torch.randn(1,12))
    torch.nn.functional.cross_entropy(logits,torch.randint(0,2,(1,32,32))).backward()
    gradients=[tokens.grad, model.detail_encoder.stem[0].weight.grad,
               model.semantic_encoder.condition.affine[-1].weight.grad]
    for block in model.decoder.cross_blocks:
        gradients.extend(block.attention.in_proj_weight.grad.chunk(3,dim=0))
        gradients.extend([block.attention_scale.grad,block.ffn_scale.grad])
    for grad in gradients:
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0


def test_zero_residual_scales_are_identity_and_detail_changes_cross_attention():
    torch.manual_seed(3)
    block=SemanticDetailCrossAttention(8,2,4,.1).eval()
    semantic=torch.randn(1,8,8,8); detail=torch.randn(1,8,16,16)
    assert not torch.allclose(block(semantic,detail),block(semantic,torch.randn_like(detail)))
    with torch.no_grad():block.attention_scale.zero_();block.ffn_scale.zero_()
    torch.testing.assert_close(block(semantic,detail),semantic)


def test_checkpoint_roundtrip_and_v2_weights_are_rejected():
    from dinov2_segmentation.models.model_v2 import GlobalLocalSegmentationModelV2
    model=tiny().eval(); restored=tiny().eval();restored.load_state_dict(model.state_dict(),strict=True)
    inputs=(torch.randn(1,3,32,32),torch.randn(1,4,16),torch.randn(1,12))
    torch.testing.assert_close(model(*inputs),restored(*inputs))
    with pytest.raises(RuntimeError):restored.load_state_dict(GlobalLocalSegmentationModelV2(2).state_dict(),strict=True)


def test_joint_factory_and_dedicated_entry():
    class Stage1(nn.Module):
        def __init__(self,*a):super().__init__();self.embed_dim=16
    with mock.patch('dinov2_segmentation.joint_model.TrainableStage1',Stage1), mock.patch(
        'dinov2_segmentation.joint_model.build_trainable_stage2',return_value=(nn.Identity(),object())):
        system=JointSegmentationSystem(decoder_version='v3',stage1_config='x',stage1_checkpoint='x',stage2_config='x',stage2_checkpoint='x')
    assert isinstance(system.decoder,GlobalLocalSegmentationModelV3)
    assert system.model_version=='joint_stage1_stage2_decoder_v3'
    from dinov2_segmentation import train_joint_v3
    with mock.patch.object(train_joint_v3,'joint_main') as main:
        train_joint_v3.main(['--execution-mode','ddp'])
        main.assert_called_once_with(['--decoder-version','v3','--execution-mode','ddp'])
    with pytest.raises(ValueError):train_joint_v3.main(['--decoder-version=v2'])


def test_default_224_forward_cpu():
    torch.set_num_threads(1)
    model=GlobalLocalSegmentationModelV3(2).eval()
    with torch.no_grad():output=model(torch.randn(1,3,224,224),torch.randn(1,196,1024),torch.randn(1,1024))
    assert output.shape==(1,2,224,224) and torch.isfinite(output).all()

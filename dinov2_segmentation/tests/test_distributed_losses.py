"""Compare the actual DDP gradient with the serial global-batch objective."""

from __future__ import annotations

from datetime import timedelta

import pytest
import torch
from torch import nn
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from torch.nn.parallel import DistributedDataParallel

from dinov2_segmentation.distributed_losses import segmentation_loss_distributed
from dinov2_segmentation.losses import segmentation_loss


def _loss_worker(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    distributed.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.manual_seed(124)
        serial_model = nn.Conv2d(3, 2, 1)
        parallel_model = nn.Conv2d(3, 2, 1)
        parallel_model.load_state_dict(serial_model.state_dict())
        parallel_model = DistributedDataParallel(parallel_model)
        images = torch.randn(3, 3, 3, 4)
        mixed = torch.tensor(
            [
                [[0, 0, 1, 1], [255, 0, 1, 0], [1, 0, 0, 0]],
                [[1, 1, 1, 0], [0, 255, 0, 1], [1, 1, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 255, 0], [0, 0, 0, 0]],
            ]
        )
        one_rank_ignored = mixed.clone()
        one_rank_ignored[2].fill_(255)
        single_valid = torch.full_like(mixed, 255)
        single_valid[2, 0, 0] = 1
        cases = {
            "mixed_and_background_rank": mixed,
            "one_rank_all_ignored": one_rank_ignored,
            "global_all_ignored": torch.full_like(mixed, 255),
            "global_background_only": torch.zeros_like(mixed),
            "single_foreground_pixel": single_valid,
        }
        shard = slice(0, 2) if rank == 0 else slice(2, 3)
        for overlap in ("dice", "foreground_tversky"):
            for case, target in cases.items():
                for class_weight in (0.25, 1.7):
                    serial_model.zero_grad(set_to_none=True)
                    parallel_model.zero_grad(set_to_none=True)
                    options = {
                        "overlap_loss": overlap,
                        "cross_entropy_weight": 0.8,
                        "dice_weight": 1.3,
                        "tumor_class_weight": class_weight,
                        "tversky_alpha": 0.3,
                        "tversky_beta": 0.7,
                    }
                    expected, expected_parts = segmentation_loss(
                        serial_model(images), target, **options
                    )
                    actual, actual_parts = segmentation_loss_distributed(
                        parallel_model(images[shard]), target[shard], **options
                    )
                    expected.backward()
                    actual.backward()
                    label = f"rank={rank}, overlap={overlap}, case={case}, weight={class_weight}"
                    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7, msg=label)
                    assert actual_parts.keys() == expected_parts.keys()
                    for name in expected_parts:
                        torch.testing.assert_close(
                            actual_parts[name], expected_parts[name], rtol=2e-6, atol=2e-7, msg=label
                        )
                    for actual_parameter, expected_parameter in zip(
                        parallel_model.module.parameters(), serial_model.parameters()
                    ):
                        torch.testing.assert_close(
                            actual_parameter.grad,
                            expected_parameter.grad,
                            rtol=3e-5,
                            atol=3e-7,
                            msg=label,
                        )
    finally:
        distributed.destroy_process_group()


@pytest.mark.skipif(not distributed.is_gloo_available(), reason="Gloo is unavailable")
def test_two_rank_global_losses_and_ddp_gradients_match_serial(tmp_path):
    multiprocessing.spawn(
        _loss_worker,
        args=(f"file://{tmp_path / 'loss_rendezvous'}",),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("overlap", ("dice", "foreground_tversky"))
def test_without_process_group_delegates_to_serial_loss(overlap):
    logits = torch.tensor([[[[0.2, 0.6]], [[0.9, -0.2]]]], requires_grad=True)
    target = torch.tensor([[[1, 255]]])
    expected, expected_parts = segmentation_loss(logits, target, overlap_loss=overlap)
    actual, actual_parts = segmentation_loss_distributed(logits, target, overlap_loss=overlap)
    torch.testing.assert_close(actual, expected)
    for name in expected_parts:
        torch.testing.assert_close(actual_parts[name], expected_parts[name])

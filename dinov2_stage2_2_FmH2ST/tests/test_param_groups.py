"""Regression test for Stage-2 optimizer parameter-group assembly."""

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from dinov2.train.gcn_meta_arch import GCNMetaArch
from dinov2.train.train import build_optimizer


class Stage2ParameterGroupsTest(unittest.TestCase):
    def setUp(self):
        self.model = GCNMetaArch.__new__(GCNMetaArch)
        nn.Module.__init__(self.model)
        self.model.cfg = SimpleNamespace(
            optim=SimpleNamespace(
                layerwise_decay=1.0,
                patch_embed_lr_mult=1.0,
                adamw_beta1=0.9,
                adamw_beta2=0.999,
            )
        )
        self.model.student = nn.ModuleDict(
            {
                "encoder": nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8)),
                "head": nn.Linear(8, 3),
            }
        )

    def test_groups_are_complete_unique_lists_and_build_adamw(self):
        groups = self.model.get_params_groups()
        self.assertIsInstance(groups, list)
        self.assertTrue(groups)
        self.assertTrue(all(group["foreach"] is True for group in groups))

        grouped_parameters = [
            parameter
            for group in groups
            for parameter in group["params"]
        ]
        expected_parameters = [
            parameter
            for parameter in self.model.student.parameters()
            if parameter.requires_grad
        ]
        grouped_ids = [id(parameter) for parameter in grouped_parameters]
        expected_ids = [id(parameter) for parameter in expected_parameters]
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
        self.assertEqual(set(grouped_ids), set(expected_ids))

        optimizer = build_optimizer(self.model.cfg, groups)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(
            sum(len(group["params"]) for group in optimizer.param_groups),
            len(expected_parameters),
        )


if __name__ == "__main__":
    unittest.main()

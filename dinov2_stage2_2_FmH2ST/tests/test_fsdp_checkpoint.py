"""Checkpoint state-dict selection for single- and multi-rank Stage 2."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from torch.distributed.fsdp import ShardingStrategy, StateDictType

import dinov2.fsdp as fsdp_module


class FSDPCheckpointStateTypeTest(unittest.TestCase):
    def test_model_without_fsdp_uses_full_state(self):
        with patch.object(fsdp_module, "get_fsdp_modules", return_value=[]):
            selected = fsdp_module.get_checkpoint_state_dict_type(object())
        self.assertEqual(selected, StateDictType.FULL_STATE_DICT)

    def test_single_rank_no_shard_uses_full_state(self):
        wrapper = SimpleNamespace(sharding_strategy=ShardingStrategy.NO_SHARD)
        with patch.object(
            fsdp_module, "get_fsdp_modules", return_value=[wrapper]
        ):
            selected = fsdp_module.get_checkpoint_state_dict_type(object())
        self.assertEqual(selected, StateDictType.FULL_STATE_DICT)

    def test_genuinely_sharded_wrappers_keep_local_state(self):
        wrappers = [
            SimpleNamespace(sharding_strategy=ShardingStrategy.SHARD_GRAD_OP),
            SimpleNamespace(sharding_strategy=ShardingStrategy.FULL_SHARD),
        ]
        with patch.object(
            fsdp_module, "get_fsdp_modules", return_value=wrappers
        ):
            selected = fsdp_module.get_checkpoint_state_dict_type(object())
        self.assertEqual(selected, StateDictType.LOCAL_STATE_DICT)


if __name__ == "__main__":
    unittest.main()

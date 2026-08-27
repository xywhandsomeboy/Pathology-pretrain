# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random

# def collate_data_and_cast(samples_list, dtype):
#     """
#     Args:
#         samples_list: list of samples from Dataset.__getitem__ (len=batch_size)
#                       这里 batch_size=1, 所以 samples_list[0] 就是一个 slide 的数据
#                       每个 sample 是一个 dict, 里面有 global/local crops 和 x,y
#     """
#     assert len(samples_list) == 1

#     sample = samples_list[0]

#     num_patches = len(sample["x"])  # 一个 slide 的 patch 数
#     n_global_crops = len(sample["global_crops"][0])  # 每个 patch 的 global crop 数
#     n_local_crops = len(sample["local_crops"][0])    # 每个 patch 的 local crop 数

#     # -------- collate crops --------
#     # 保留 patch 维度: [num_patches, n_global, C, H, W]
#     collated_global_crops = torch.stack([s[i] for i in range(n_global_crops) for s in sample["global_crops"]]).to(dtype)
#     collated_local_crops = torch.stack([s[i] for i in range(n_local_crops) for s in sample["local_crops"]]).to(dtype)

#     # -------- collate coords --------
#     coords_x = torch.tensor(sample["x"], dtype=torch.float32)
#     coords_y = torch.tensor(sample["y"], dtype=torch.float32)
#     coords = torch.stack([coords_x, coords_y], dim=1)  # [num_patches, 2]

#     return {
#         "collated_global_crops": collated_global_crops,   # [num_patches, n_global, C, H, W]
#         "collated_local_crops": collated_local_crops,     # [num_patches, n_local, C, H, W]
#         "coords": coords,                                 # [num_patches, 2]
#         "slide_name": sample.get("slide_name", None),
#     }

def collate_data_and_cast(samples_list, dtype):
    # dtype = torch.half  # TODO: Remove
    
    filenames = [s[2] for s in samples_list]
    
    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])

    collated_global_crops = torch.stack([s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])

    collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])

    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "filenames":filenames,
        "images": torch.stack([s[0]["image"] for s in samples_list]).to(dtype) if isinstance(samples_list[0][0], dict) and ("image" in samples_list[0][0]) else None,
    }
    
# def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
#     # dtype = torch.half  # TODO: Remove
    
#     filenames = [s[1] for s in samples_list]
#     print("check ",filenames[0])
    
#     n_global_crops = len(samples_list[0][0]["global_crops"])
#     n_local_crops = len(samples_list[0][0]["local_crops"])

#     collated_global_crops = torch.stack([s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])

#     collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])

#     B = len(collated_global_crops)
#     N = n_tokens
#     n_samples_masked = int(B * mask_probability)
#     probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
#     upperbound = 0
#     masks_list = []
#     for i in range(0, n_samples_masked):
#         prob_min = probs[i]
#         prob_max = probs[i + 1]
#         masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
#         upperbound += int(N * prob_max)
#     for i in range(n_samples_masked, B):
#         masks_list.append(torch.BoolTensor(mask_generator(0)))

#     random.shuffle(masks_list)

#     collated_masks = torch.stack(masks_list).flatten(1)
#     mask_indices_list = collated_masks.flatten().nonzero().flatten()

#     masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

#     return {
#         "collated_global_crops": collated_global_crops.to(dtype),
#         "collated_local_crops": collated_local_crops.to(dtype),
#         "collated_masks": collated_masks,
#         "mask_indices_list": mask_indices_list,
#         "masks_weight": masks_weight,
#         "upperbound": upperbound,
#         "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
#         "filenames":filenames,
#     }


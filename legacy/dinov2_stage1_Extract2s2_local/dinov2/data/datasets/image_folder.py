import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import random
import pandas as pd
import pickle
import gzip
from pathlib import Path
import torch
from typing import Callable, List, Optional, Tuple, Union

_DEFAULT_MAX_PATCH_SIZE = 50

class ImageFolder(Dataset):
    def __init__(
        self, 
        *,
        root:str,
        transform: Optional[Callable] = None,
        max_patches: int = _DEFAULT_MAX_PATCH_SIZE,
    ) -> None:
        """
        Args:         
            labels (dict): {slide_name: label} 鐨勫瓧鍏?            
            transform (callable): 鏁版嵁澧炲己
            max_patches (int): 姣忓紶slide浣跨敤鐨勬渶澶atch鏁?        
        """
        self.root = os.path.join(root,"patch_grid_positions-TCGA_CESC-hospital.csv")
        self.transform = transform
        self.max_patches = max_patches
        self.file_info = pd.read_csv(self.root)
        cache_path = self.root.replace(".csv","_slide_patch_map.pkl.gz")
        
              
        self.slide_patch_map = {}
                        
        with gzip.open(cache_path, "rb") as f:
            self.slide_patch_map = pickle.load(f)
            self.slide_patch_map = {k: v for k, v in self.slide_patch_map.items() if len(v) >= max_patches}
            
            self.slide_names = list(self.slide_patch_map.keys())
            self.file_info = self.file_info[self.file_info['slide_name_split'].isin(self.slide_patch_map.keys())]

        
    def __len__(self):
        return len(self.file_info)
    
    def __getitem__(self, idx):
        slide_name = self.file_info.iloc[idx]['slide_name_split']
        patch_paths = self.slide_patch_map[slide_name]
        

        if len(patch_paths) > self.max_patches:
            selected_paths = random.sample(patch_paths, self.max_patches)
        else:
            selected_paths = patch_paths
        # selected_paths = patch_paths
        
        patches = {"global_crops": [], "local_crops": []}
        coords_x = []
        coords_y = []
        for patch_name in selected_paths:
            img = Image.open(patch_name).convert('RGB')
            row = self.file_info[self.file_info["filepath"]==patch_name]
            coords_x.append(row['x'].values[0])
            coords_y.append(row['y'].values[0])
            
            if self.transform:
                data = self.transform(img)  # data[k] 是 list of Tensors
                # 保留 patch 维度
                patch_global = data["global_crops"]  # list of tensors
                patch_local = data["local_crops"]
                patches["global_crops"].append(patch_global)
                patches["local_crops"].append(patch_local)
        # patches = {"global_crops": [], "local_crops": []}
        # coords_x = []
        # coords_y = []
        # for patch_name in selected_paths:
        #     path = patch_name
        #     img = Image.open(path).convert('RGB')
        #     row = self.file_info[self.file_info["filepath"]==patch_name]
        #     coords_x.append(row['x'].values[0])
        #     coords_y.append(row['y'].values[0])
        #     if self.transform:
        #         data = self.transform(img)
        #         for k in ["global_crops", "local_crops"]:
        #             patches[k].extend(data[k])
                    
        # for i in ["global_crops", "local_crops"]:
        #     patches[i] = torch.stack(patches[i])
        patches["x"] = np.array(coords_x)
        patches["y"] = np.array(coords_y)
        patches["slide_name"] = slide_name
        return patches
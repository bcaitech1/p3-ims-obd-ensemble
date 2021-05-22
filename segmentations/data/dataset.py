from torch.utils.data import Dataset
import cv2
import torch
import pandas as pd
import numpy as np
import os

# from pycocotools.coco import COCO

class TrashDataset(Dataset):
    def __init__(self, df, data_root = "../dataset", transforms=None):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):

        img_path = os.path.join(self.data_root, self.df.iloc[idx]['filepath'])
        imgs = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        mask_path = os.path.join(self.data_root, self.df.iloc[idx]['masks'])
        masks = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        if self.transforms is not None:
            transformed = self.transforms(image=imgs, mask=masks)
            imgs = transformed['image']
            masks = transformed['mask']
        
        return imgs, masks

class TrashTestDataset(Dataset):
    """COCO format"""
    def __init__(self, data_dir, data_root="../dataset", transforms=None):
        super().__init__()
        self.mode = mode
        self.transforms = transform
        self.coco = COCO(data_dir)
        self.data_root = data_root
        
    def __getitem__(self, index: int):
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        img_path = os.path.join(self.data_root, image_infos['file_name'])
        images = cv2.imread(img_path, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            transformed = self.transforms(image=images)
            images = transformed["image"]
        
        return images, image_infos
    
    def __len__(self) -> int:
        return len(self.coco.getImgIds())
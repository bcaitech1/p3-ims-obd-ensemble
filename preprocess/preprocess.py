import os
import random
import time
import json
import warnings 
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import cv2

import numpy as np
import pandas as pd

# 전처리를 위한 라이브러리
from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

category_names = ['Backgroud', 'UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
batch_size = 16 

def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

class CustomDataLoader(Dataset):
    """COCO format"""
    def __init__(self, data_dir, mode = 'train', transform = None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)
        
    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0
        
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id + 1" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # Unknown = 1, General trash = 2, ... , Cigarette = 11
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = category_names.index(className)
                masks = np.maximum(self.coco.annToMask(anns[i])*pixel_value, masks)
            masks = masks.astype(np.float32)

            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            
            return images, masks, image_infos
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            
            return images, image_infos
    
    
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())


def collate_fn(batch):
    return tuple(zip(*batch))


def save_mask(dataloader, save_dir):

    for imgs, masks, img_infos in dataloader:
        for img, mask, img_info in zip(imgs, masks, img_infos):
            file_name = img_info['file_name']
            file_path = os.path.join(save_dir, file_name)
            plt.imsave(file_path, mask)
            


if __name__=="__main__":
    dataset_path = '/opt/ml/input/data'
    save_path = '/opt/ml/input/label'
    train_path = "/opt/ml/input/data/train.json"
    val_path = "/opt/ml/input/data/val.json"

    train_transform = A.Compose([
                            ToTensorV2()
                            ])
    
    train_dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=train_transform)
    val_dataset   = CustomDataLoader(data_dir=val_path, mode='val', transform=train_transform)

    train_loader = torch.utils.data.DataLoader( dataset=train_dataset, 
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=4,
                                                collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader( dataset=val_dataset, 
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=4,
                                                collate_fn=collate_fn)
    save_mask(train_loader, save_path)
    save_mask(val_loader, save_path)
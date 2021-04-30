import os
import random
import time
import json
import warnings 
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
import gc
# 전처리를 위한 라이브러리
from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset.defaultDataset import DefaultDataset
from Tools.training import *
from Tools.inference import *
from lib.hrdnet.model import get_seg_model
from lib.utils import *


Augmentation = {
    "train": A.Compose([
                            ToTensorV2()
                            ]),
    "valid": A.Compose([
                          ToTensorV2()
                          ]),
    "test": A.Compose([
                           ToTensorV2()
                           ])
}



if __name__=="__main__":
    set_seed()
    print('pytorch version: {}'.format(torch.__version__))
    print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

    print(torch.cuda.get_device_name(0))
    print(torch.cuda.device_count())

    if_train = True # True: train model False test model
    only_valid = False # True: Not train, False: train + valid
    batch_size = 1   # Mini-batch size
    num_epochs = 150
    learning_rate = 0.0001
    dataset_path = "/opt/ml/input/data"

    if if_train:
        train_path = dataset_path + '/train.json'
        val_path = dataset_path + '/val.json'

        train_dataset = DefaultDataset(dataset_path=dataset_path, data_dir=train_path, mode='train', transform=Augmentation["train"])
        val_dataset = DefaultDataset(dataset_path=dataset_path, data_dir=val_path, mode='val', transform=Augmentation["valid"])

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=1,
                                           shuffle=False,
                                           num_workers=0,
                                           collate_fn=collate_fn)

        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=0,
                                         collate_fn=collate_fn)
            
        gc.collect()
        torch.cuda.empty_cache()
    
        model = get_seg_model("/opt/ml/input/pretrained/hrnet_best_model(pretrained)_light_13_loss.pt", use_pretrain=False)
        model = model.cuda()
    
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        val_every = 1
        saved_dir = '/opt/ml/input/pretrained'
        if only_valid:
            avrg_loss, miou, class_mIoU = validation(0, model, val_loader, criterion, 12)
            show_class_mIoU(class_mIoU)
        else:
            train(num_epochs, model, train_loader, val_loader, criterion, optimizer, saved_dir, val_every, 12)

    else:
        test_path = dataset_path + '/test.json'
        test_dataset = DefaultDataset(dataset_path=dataset_path, data_dir=test_path, mode='test', transform=Augmentation["test"])

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          num_workers=0,
                                          collate_fn=collate_fn)

        model = get_seg_model("/opt/ml/input/pretrained/hrnet_best_model(pretrained)_light_13_loss.pt", use_pretrain=True)
        model = model.cuda()
        file_names, preds = test(model, test_loader)
        save_sub(file_names, preds, "/opt/ml/code/submission/hrnet_best_miou_epoch_13.csv")
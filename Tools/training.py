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

import albumentations as A
from albumentations.pytorch import ToTensorV2
from lib.utils import * 

def save_model(model, saved_dir, file_name='hrnet_best_model(pretrained).pt'):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model.state_dict(), output_path)

def train(num_epochs, model, data_loader, val_loader, criterion, optimizer, saved_dir, val_every):
    print('Start training..')
    best_loss = 9999999
    best_miou = 0
    for epoch in range(num_epochs):
        model.train()
        for step, (images, masks, _) in enumerate(data_loader):
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)
            
            # gpu 연산을 위해 device 할당
            images, masks = images.cuda(), masks.cuda()
              
            # inference
            outputs = model(images)
            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            gc.collect()
            torch.cuda.empty_cache()
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch+1, num_epochs, step+1, len(data_loader), loss.item()))
        
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            avrg_loss, miou = validation(epoch + 1, model, val_loader, criterion)
            if avrg_loss < best_loss:
                print('Best performance at epoch: {}'.format(epoch + 1))
                print('Save model in', saved_dir)
                best_loss = avrg_loss
                save_model(model, saved_dir, file_name=f'hrnet_best_model(pretrained)_light_{epoch}_loss.pt')
            if best_miou < miou:
                print('Best performance (MIOU) at epoch: {}'.format(epoch + 1))
                print('Save model in', saved_dir)
                best_miou = miou
                save_model(model, saved_dir, file_name=f'hrnet_best_model(pretrained)_light_{epoch}_miou.pt')


def validation(epoch, model, data_loader, criterion):
    print('Start validation #{}'.format(epoch))
    model.eval()
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        mIoU_list = []
        for step, (images, masks, _) in enumerate(data_loader):
            
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)

            images, masks = images.cuda(), masks.cuda()            

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
          
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()

            mIoU = label_accuracy_score(masks.detach().cpu().numpy(), outputs, n_class=12)[2]
            mIoU_list.append(mIoU)
            
            gc.collect()
            torch.cuda.empty_cache()
        avrg_loss = total_loss / cnt
        print('Validation #{}  Average Loss: {:.4f}, mIoU: {:.4f}'.format(epoch, avrg_loss, np.mean(mIoU_list)))

    return avrg_loss, np.mean(mIoU_list)

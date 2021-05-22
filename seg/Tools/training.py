import os
import random
import time
import json
import warnings 
from tqdm import tqdm
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

def train(num_epochs, model, data_loader, val_loader, criterion, optimizer, saved_dir, val_every, n_class=12):
    print('Start training..')
    best_loss = 9999999
    best_miou = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = []
        for step, (images, masks, _) in tqdm(enumerate(data_loader)):
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
            train_loss.append(loss.item())
            optimizer.step()
            
            gc.collect()
            torch.cuda.empty_cache()
                    
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            avrg_loss, miou, class_mIoU = validation(epoch + 1, model, val_loader, criterion, n_class)
            print('Epoch [{}/{}], Train Loss: {:.4f} Vali Loss: {:.4f}, Vali mIoU: {:.4f}'.format(epoch+1, num_epochs, np.mean(train_loss), avrg_loss, miou))
            show_class_mIoU(class_mIoU)
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


def show_class_mIoU(avrg_class_IoU):
    # Class Score
    class_name=['BG','UNK','General Trash','Paper','Paper pack','Metal','Glass','Plastic','Styrofoam','Plastic Bag','Battery','Clothing']
    print('-'*20)
    print('Validation Class Pred mIoU Score')
    for idx, class_score in enumerate(avrg_class_IoU):
        print('[{}] mIoU : [{:.4f}]'.format(class_name[idx],class_score))
    print('-'*20)


def validation(epoch, model, data_loader, criterion, n_class):
    print('Start validation #{}'.format(epoch))
    model.eval()
    hist = np.zeros((n_class, n_class))
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
            
            hist = add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=n_class)
            gc.collect()
            torch.cuda.empty_cache()
        
        mIoU, class_mIoU = label_accuracy_score(hist)    
        avrg_loss = total_loss / cnt

    return avrg_loss, mIoU, class_mIoU

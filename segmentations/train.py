import segmentation_models_pytorch as smp
from tqdm import tqdm
import gc

import torch
from torch import nn
import cv2
import os
import numpy as np
import pandas as pd

from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam, AdamW
from torch.nn import functional as F

import matplotlib.pyplot as plt
import sys
import time
import random
import argparse
import json

from scheduler import *
from losses import *
from models import *
from data import *
from utils import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def train_one_epoch(epoch, scaler, model, loss_fn, optimizer, train_loader, device, scheduler, cfg):
    model.train()

    running_loss = None
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)

    for step, (imgs, masks) in pbar:
        mix_decision = np.random.rand()
        
        imgs = torch.stack(imgs)
        masks = torch.stack(masks)
        imgs = imgs.to(device).float()
        masks = masks.to(device).long()

        if mix_decision < cfg["mix_prob"]:
            imgs, masks = cutmix(imgs, masks, 1.0)

        with autocast():
            model.to(device)
            mask_preds = model(imgs)

            loss = loss_fn(mask_preds, masks) / cfg["gradient_accumulation_steps"]
            scaler.scale(loss).backward()

            if running_loss is None:
                running_loss = loss.item() * cfg["gradient_accumulation_steps"]
            else:
                running_loss = running_loss * 0.99 + loss.item() * cfg["gradient_accumulation_steps"] * 0.01

            if ((step + 1) % cfg["gradient_accumulation_steps"]==0) or ((step+1)==(len(train_loader))):
                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()
                description = f"epoch {epoch} loss: {running_loss: .4f}"
                pbar.set_description(description)
    scheduler.step()


def valid_one_epoch(epoch, model, device, loss_fn, val_loader):
    model.eval()

    total_loss = 0
    running_loss = None
    total_cnt = 0
    mIoU_list = []
    pbar = tqdm(enumerate(val_loader), total=len(val_loader), position=0, leave=True)
    
    for step, (imgs, masks) in pbar:
        imgs = torch.stack(imgs)
        masks = torch.stack(masks)
        imgs = imgs.to(device).float()
        masks = masks.to(device).long()

        total_cnt += 1

        mask_preds = model(imgs)
        loss = loss_fn(mask_preds, masks)

        mask_preds = torch.argmax(mask_preds, dim=1).detach().cpu().numpy()

        mIoU = label_accuracy_score(masks.detach().cpu().numpy(), mask_preds, n_class=12)[2]
        mIoU_list.appen(mIoU)

        if running_loss is None:
            running_loss = loss.item()
        else:
            running_loss = running_loss * 0.99 + loss.item() * 0.01
        
        description = f"epoch {epoch} loss: {running_loss: .4f}, mIoU: {np.mean(mIoU_list): .4f}"
        pbar.set_description(description)
    
    return total_loss/total_cnt, np.mean(mIoU_list)

def main(cfg):
    df = pd.read_csv("./train.csv")
    if cfg["pseudo_label"] == "true":
        additional_df = pd.read_csv("./test.csv")
        df = pd.concat([df, additional_df], ignore_index=True)
    
    seed_everything(cfg["seed"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    df = split_dataframe(df, cfg["fold_num"], cfg["seed"])

    for fold in range(cfg["fold_num"]):
        if cfg["decoder"] == "Unet++":
            model = smp.UnetPlusPlus(cfg["encoder"], encoder_weights=CFG["pretrained"], classes=12).to(device)
        elif cfg["decoder"] == "DeepLabV3+":
            model = smp.DeepLabV3Plus(cfg["encoder"], encoder_weights=cfg["pretrained"], classes=12).to(device)
        elif cfg["decoder"] == "UneXt":
            model = UneXt50().to(device)
        elif cfg["decoder"] == "EffUnext":
            model = EffUnet().to(device)
    
        optimizer = AdamW(model.parameters(), lr=0, weight_decay=cfg["weight_decay"])
        scheduler = CustomCosineAnnealingWarmUpRestarts(optimizer, T_0=cfg["epochs"], T_mult=1, eta_max=cfg["lr"], T_up=cfg["epochs"]//5, gamma=1)
        
        loss_fn = SoftCrossEntropyLoss(smooth_factor=0.1).to(device)
        
        scaler = GradScaler()
        train_loader, valid_loader = prepare_train_val_dataloader(df, [fold], cfg)

        best_mIoU = 0
        
        for epoch in range(cfg["epochs"]):
            train_one_epoch(epoch, scaler, model, loss_fn, optimizer, train_loader, device, scheduler, cfg)
            
            with torch.no_grad():
                epoch_loss, mIoU = valid_one_epoch(epoch, model, device, loss_fn, val_loader)
            
            if best_mIoU < mIoU:
                best_mIoU = mIoU
                create_folder(cfg["save_folder"])
                torch.save(model.state_dict(), f"{cfg['save_folder']}/{cfg['decoder']}_{cfg['encoder']}")
                print("model is saved\n")
        
        del model, optimizer, train_loader, valid_loader, scheduler
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--decoder", type=str, default="EffUneXt")
    cli_parser = cli_parser.parse_args()
    cfg = None

    if cli_parser.decoder == "UneXt":
        f = open("./config/unext_config.json", encoding="UTF-8")
        cfg = json.loads(f.read())
    elif cli_parser.decoder == "EffUneXt":
        f = open("./config/effunext_config.json", encoding="UTF-8")
        cfg = json.loads(f.read())
    elif cli_parser.decoder == "DeepLabV3+":
        f = open("./config/resnext50_deeplabv3p_config.json", encoding="UTF-8")
        cfg = json.loads(f.read())
    
    main(cfg)
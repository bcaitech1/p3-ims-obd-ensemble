import segmentation_models_pytorch as smp
import gc
import torch
import cv2
import os
import zipfile
import numpy as np
import pandas as pd
import albumentations as A

from torch import nn
from tqdm import tqdm
from torch.nn import functional as F

from models import *
from data import *
from utils import *

def flip_tta(models, imgs, device):
    outs = None
    flip_outs = None
    flips = [[-1],[-2],[-2,-1]]

    for model in models:
        if outs == None:
            outs = model(imgs.to(device).float()).detach()
        else:
            outs += model(imgs.to(device).float()).detach()
    
    for flip in flips:
        flip_img = torch.flip(imgs, flip)
        tmp_outs = None
        for model in models:
            flip_out = model(flip_img.to(device).float()).detach()
            flip_out = torch.flip(flip_out, flip)
            if tmp_outs == None:
                tmp_outs = flip_out
            else:
                tmp_outs += flip_out

            if flip_outs == None:
                flip_outs = tmp_outs
            else:
                flip_outs += tmp_outs

    outs += flip_outs

    return outs / (len(models)*len(flips) + len(models))

def inference_models(all_models, data_loader , device, OUT_MASKS):
    size = 256
    transform = A.Compose([A.Resize(256,256)])
    print("Start Prediction")
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), position=0, leave=True)

    with torch.no_grad():
        with zipfile.ZipFile(OUT_MASKS, 'w') as mask_out:
            for step, (imgs, image_infos) in pbar:
                imgs = torch.stack(imgs)
                outs = []
                for models in all_models:
                    outs.append(flip_tta(models, imgs, device))
                
                final_outs = None
                for out in outs:
                    if final_outs == None:
                        final_outs = out
                    else:
                        final_outs += out
                final_outs = F.softmax(final_outs, dim=1)
                oms = torch.argmax(final_outs, dim=1).detach().cpu().numpy()

                file_name = image_infos[0]['file_name'].split("/")
                file_name[0] += "_masks"
                file_name[1] = file_name[1][:-4]
                file_name = "/".join(flie_name)

                m = cv2.imencode(".png", oms.squeeze())[1]
                mask_out.writestr(f"{file_name}.png", m)

                tmp_mask = []
                for img, mask in zip(np.stack(imgs), oms):
                    transformed = transform(image=img, mask=mask)
                    mask = transformed['mask']
                    tmp_mask.append(mask)
                
                oms = np.array(tmp_mask)
                oms = oms.reshape([oms.shape[0], size*size]).astype(int)

                preds_array = np.vstack((preds_array, oms))

                file_name_list.append([i['file_name'] for i in image_infos])
    
    print("End prediction")
    file_names = [y for x in file_name_list for y in x]

    return file_names, preds_array

def load_checkpoint(model, paths):
    models = []
    for path in paths:
        if model == "resnext":
            model = smp.DeepLabV3Plus("resnext50_32x4d", encoder_weights=None, in_channels=3, classes=12)
        elif model == "Unext":
            model = UneXt50().to(device)
        elif model == "EffUnext":
            model = EffUnet().to(device)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)
        model.eval()
        model.to(device)
        models.append(model)

        del checkpoint
    return models

if __name__ == "__main__":

    device = "cuda"

    test_path = './dataset/test.json'

    test_loader = prepare_test_dataloader(
                                    test_dir="./dataset/test.json",
                                    root="./dataset",
                                    save_mask=True)
    all_models = []

    MODEL_PATHS_resnext50 = ['./save/resnext50/0_checkpoint.pt',
                            './saveresnext50/1_checkpoint.pt',
                            './save/resnext50/2_checkpoint.pt',
                            './save/resnext50/3_checkpoint.pt',
                            './save/resnext50/4_checkpoint.pt',
                            './save/resnext50/5_checkpoint.pt']

    all_models.append(load_checkpoint("Unext", MODEL_PATHS_resnext50))

    MODEL_PATHS_unext = ['./save/unext/UneXt_resnext50_0.pth',
                        './save/unext/UneXt_resnext50_1.pth',
                        './save/unext/UneXt_resnext50_2.pth',
                        './save/unext/UneXt_resnext50_3.pth',
                        './save/unext/UneXt_resnext50_4.pth']

    # all_models.append(load_checkpoint("resnext", MODEL_PATHS_unext))

    MODEL_PATHS_effunet = ['./save/effunet/0_checkpoint.pt',
                        './save/effunet/1_checkpoint.pt',
                        './save/effunet/2_checkpoint.pt',
                        './save/effunet/3_checkpoint.pt',
                        './save/effunet/4_checkpoint.pt']

    # all_models.append(load_checkpoint("EffUnext", MODEL_PATHS_effunet))

    SUBMISSION_PATH = "./submission.csv"
    SAMPLE_SUBMISSON_PATH = './sample_submission.csv'
    OUT_MASKS = './pseudo_masks.zip'

    file_names, preds = inference_models(all_models, test_loader, device, OUT_MASKS)

    submission = pd.read_csv(SAMPLE_SUBMISSON_PATH, index_col=None)

    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    submission.to_csv(SUBMISSION_PATH, index=False)
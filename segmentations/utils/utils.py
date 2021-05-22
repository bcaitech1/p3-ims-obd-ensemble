from sklearn.model_selection import KFold

import random
import torch
import numpy as np
import os

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_folder(directory):
    try:
        os.makedirs(directory)
    except:
        pass

def split_dataframe(df, fold_num, seed):
    kf = KFold(fold_num, shuffle=True, random_state=seed)
    df["Folds"] = 0

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        df.loc[val_idx, "Folds"] = fold
    
    return df
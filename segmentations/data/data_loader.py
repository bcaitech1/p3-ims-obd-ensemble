import cv2
import pandas as pd
from torch.utils.data import DataLoader
from .dataset import TrashDataset, TrashTestDataset
from .augmentations import (
    get_train_transforms,
    get_validation_transforms,
    get_test_transforms,
)

def collate_fn(batch):
    return tuple(zip(*batch))

def prepare_train_val_dataloader(df, fold, cfg):
    train_df = df[~df.Folds.isin(fold)]
    val_df = df[df.Folds.isin(fold)]

    train_ds = TrashDataset(train_df, data_root=cfg["root"], transforms=get_train_transforms())
    val_ds = TrashDataset(val_df, data_root=cfg["root"], transforms=get_validation_transforms())

    train_loader = DataLoader(train_ds,
                            batch_size=cfg["batch_size"],
                            shuffle=True,
                            num_workers=cfg["num_workers"],
                            pin_memory=True,
                            collate_fn=collate_fn,
                            drop_last=True)
    
    val_loader = DataLoader(val_ds,
                            batch_size=cfg["batch_size"],
                            shuffle=False,
                            num_workers=cfg["num_workers"],
                            collate_fn=collate_fn)
    
    return train_loader, val_loader

def prepare_test_dataloader(df, fold, cfg):
    test_ds = TrashTestDataset(data_dir=cfg["test_dir"], data_root=cfg["root"], transforms=get_test_transforms())

    if cfg["save_mask"] == "true":
        test_loader = DataLoader(test_ds,
                                batch_size=1,
                                shuffle=False,
                                num_workers=cfg["num_workers"],
                                collate_fn=collate_fn)

    else:
        test_loader = DataLoader(test_ds,
                                batch_size=cfg["batch_size"],
                                shuffle=False,
                                num_workers=cfg["num_workers"],
                                collate_fn=collate_fn)
    
    return test_loader
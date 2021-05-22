import sys
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, set_random_seed, init_detector
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.utils import collect_env, get_root_logger

import torch
import time

if __name__ == "__main__":

    classes = (
        "UNKNOWN",
        "General trash",
        "Paper",
        "Paper pack",
        "Metal",
        "Glass",
        "Plastic",
        "Styrofoam",
        "Plastic bag",
        "Battery",
        "Clothing",
    )

    CFG_PATH = "/content/drive/MyDrive/boostcamp_obd/UniverseNet-master/configs/universenet/custom_universenet101_2008d.py"
    PREFIX = "/content/drive/MyDrive/boostcamp_obd/"
    WORK_DIR = "../work_dirs/fold_0"
    CHK_PATH = "/content/drive/MyDrive/boostcamp_obd/UniverseNet-master/pretrained/universenet101_2008d_fp16_4x4_mstrain_480_960_20e_coco_20201023_epoch_20-3e0d236a.pth"
    # config file 들고오기
    cfg = Config.fromfile(CFG_PATH)

    # dataset 바꾸기

    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = PREFIX + "data/"
    cfg.data.train.ann_file = PREFIX + "train_data0.json"
    # cfg.data.train.pipeline[3]['img_scale'] = [(256, 256), (512, 512)]

    cfg.data.val.classes = classes
    cfg.data.val.img_prefix = PREFIX + "data/"
    cfg.data.val.ann_file = PREFIX + "valid_data0.json"
    # cfg.data.val.pipeline[1]['img_scale'] = (512, 512)

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = PREFIX + "data/"
    cfg.data.test.ann_file = PREFIX + "test.json"
    # cfg.data.test.pipeline[1]['img_scale'] = (512, 512)

    cfg.data.samples_per_gpu = 4

    cfg.seed = 2020
    cfg.gpu_ids = [0]
    cfg.work_dir = WORK_DIR

    cfg.model.bbox_head.num_classes = 11

    cfg.model.pretrained = None
    if cfg.model.get("neck"):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get("rfp_backbone"):
                    if neck_cfg.rfp_backbone.get("pretrained"):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get("rfp_backbone"):
            if cfg.model.neck.rfp_backbone.get("pretrained"):
                cfg.model.neck.rfp_backbone.pretrained = None

    model = build_detector(cfg.model)
    datasets = [build_dataset(cfg.data.train)]
    checkpoint = load_checkpoint(model, CHK_PATH)
    model.CLASSES = datasets[0].CLASSES

    train_detector(model, datasets[0], cfg, distributed=False, validate=True)

_base_ = './cascade_mask_rcnn_r50_fpn_20e_coco.py'
model = dict(
    pretrained='open-mmlab://resnext101_32x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'))

load_from = "/content/drive/MyDrive/boostcamp/stage_3_det/data/pretrained/resnext101_32x4d-a5af3160.pth"
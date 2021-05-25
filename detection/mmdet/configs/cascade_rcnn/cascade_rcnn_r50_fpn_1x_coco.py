_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
load_from = "/content/drive/MyDrive/boostcamp/stage_3_det/data/pretrained/resnext101_32x4d-a5af3160.pth"
_base_ = [
    '../_base_/models/cascade_mask_rcnn_swin_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False
    ),
    neck=dict(in_channels=[96, 192, 384, 768]),
    roi_head=dict(
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=11,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=11,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=11,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
        ]))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
albu_train_transforms = [
    # dict(
    #     type='PhotoMetricDistortion', 
    #     brightness_delta=0.1, 
    #     contrast_range=(0.9, 1.1), 
    #     saturation_range=(0.9, 1.1),
    #     hue=18,
    #     p=0.5),
    # dict(
    #     type='RandomCropNearBBox',
    #     max_part_shift=0.3, 
    #     always_apply=False, 
    #     p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(type='MedianBlur', blur_limit=5),
            dict(type='MotionBlur', blur_limit=5),
            dict(type='GaussianBlur', blur_limit=5),
        ],
        p=0.7),
    dict(
        type='GaussNoise', 
        var_limit=(5.0, 30.0), 
        p=0.7),
    dict(
        type='RandomRotate90'),
    #  dict(
    #      type='RandomBrightnessContrast',
    #      brightness_limit=[-0.2, 0.2],
    #      contrast_limit=[-0.2, 0.2],
    #      p=0.2),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',  # [], []에 있는 거 시도해서 좋은 걸로 선택하는 듯
         policies=[
             [
                dict(type='Resize',
                      img_scale=[(512, 512),(576, 576),(640, 640),(704, 704),(768, 768),
                                 (832, 832),(896, 896),(960, 960),(1024, 1024)],
                      multiscale_mode='value',
                      keep_ratio=True)    
             ],
             [
                 dict(type='Resize',
                      img_scale=[(512, 512), (1024, 1024)],
                      multiscale_mode='value',
                      keep_ratio=True),
                #  dict(type='RandomCrop',
                #       crop_type='absolute_range',
                #       crop_size=(512, 512),
                #       allow_negative_crop=True),
                 dict(type='RandomCrop',
                      crop_type='relative_range',
                      crop_size=(0.75, 0.75),
                      allow_negative_crop=False),
                 dict(type='Resize',
                      img_scale=[(512, 512),(576, 576),(640, 640),(704, 704),(768, 768),
                                 (832, 832),(896, 896),(960, 960),(1024, 1024)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    # Pad - Albu - Normalize 순서로 해야 함!!
    dict(type='Pad', size_divisor=32),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
data = dict(train=dict(pipeline=train_pipeline))

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
# lr_config = dict(step=[27, 33])
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.0001,
    min_lr_ratio=1e-7)

runner = dict(type='EpochBasedRunnerAmp', max_epochs=36)

#do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
      type="DistOptimizerHook",
      update_interval=1,
      grad_clip=None,
     coalesce=True,
     bucket_size_mb=-1,
     use_fp16=True,
  )

load_from = "/content/drive/MyDrive/boostcamp/stage_3_det/pretrained/cascade_mask_rcnn_swin_tiny_patch4_window7.pth"
#resume_from = "불러올 pth파일 경로"

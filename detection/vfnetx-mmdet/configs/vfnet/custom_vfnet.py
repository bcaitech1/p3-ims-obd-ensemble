_base_ = './vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco.py'
model = dict(
    pretrained='open-mmlab://res2net101_v1d_26w_4s',
    backbone=dict(
        type='Res2Net',
        depth=101,
        scales=4,
        base_width=26,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='PAFPNX',
        in_channels=[256, 512, 1024, 2048],
        out_channels=384,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True,
        pafpn_conv_cfg=dict(type='DCNv2'),
        no_norm_on_lateral=True,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    bbox_head=dict(
        type='VFNetHead',
        num_classes=11,
        in_channels=384,
        stacked_convs=4,
        feat_channels=384,
        strides=[8, 16, 32, 64, 128],
        regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512), (512,
                                                                      1e8)),
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            center_offset=0.0,
            strides=[8, 16, 32, 64, 128]),
        center_sampling=False,
        dcn_on_last_conv=True,
        use_atss=True,
        use_vfl=True,
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        #fusion_cfg=dict(type='soft_vote', scale_ranges=[(256, 2048), (256, 2048), (256, 2048) , (0, 256), (0, 512), (0, 128), (0, 256)]),
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        #nms=dict(type='soft_nms', iou_threshold=0.65),
        max_per_img=100))

# data setting
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomCrop',
        crop_type='relative_range',
        crop_size=(0.75, 0.75),
        crop_p=0.4),
    dict(
        type='Resize',
        img_scale=[(512, 512), (1024, 1024)],
        multiscale_mode='range',
        keep_ratio=True),
    #dict(type='Mosaic',p=0.3),
    dict(type='Mixup',p=0.4),
    dict(type="RandomRotate90", rotate_ratio=0.5),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

tta_scale = [(512, 512),(576, 576),(640, 640),(704, 704),(768, 768),(832, 832),(896, 896),(960, 960),(1024, 1024)]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=tta_scale,
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=val_pipeline),
    test=dict(pipeline=test_pipeline))

# optimizer
optimizer = dict(lr=0.01, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)

#lr_config = dict(step=[27, 49])
# learning policy
lr_config = dict(policy='CosineAnnealing',
                     warmup='linear',warmup_iters=600,
                     warmup_ratio=0.0001,
                     min_lr_ratio=1e-7)
'''
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=600,
    warmup_ratio=0.1,
    step=[21, 33, 37])
'''
runner = dict(type='EpochBasedRunner', max_epochs=40)

# swa learning policy
swa_lr_config = dict(
    policy='cyclic',
    target_ratio=(1, 0.01),
    cyclic_times=12,
    step_ratio_up=0.0)
swa_runner = dict(type='EpochBasedRunner', max_epochs=12)

fp16 = dict(loss_scale=512.)

evaluation = dict(interval=1, metric='bbox')
# runtime
load_from = "/content/drive/MyDrive/workspace/vfnet/pth/SWA-VFNetX-1-18-53.4_VFNetX-R2101-41e-0.01-0.0001-52.2.pth"
resume_from = None
workflow = [('train', 1)]
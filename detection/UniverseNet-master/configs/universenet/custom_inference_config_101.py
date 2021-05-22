# _base_ = './custom_universenet50_gfl_2x.py'
_base_ = './custom_universenet101_2008d.py'

# 5 scales and flip
tta_flip = True
tta_scale = [(256, 256), (384, 384), (512, 512), (768, 768), (896,896), (960, 960), (1024, 1024)]
scale_ranges = [(256, 2048), (256, 2048), (256, 2048) , (0, 256), (0, 512), (0, 128), (0, 256)]

# 13 scales and flip (slow)
# tta_flip = True
# tta_scale = [(667, 400), (833, 500), (1000, 600), (1067, 640), (1167, 700),
#              (1333, 800), (1500, 900), (1667, 1000), (1833, 1100),
#              (2000, 1200), (2167, 1300), (2333, 1400), (3000, 1800)]
# scale_ranges = [(96, 10000), (96, 10000), (64, 10000), (64, 10000),
#                 (64, 10000), (0, 10000), (0, 10000), (0, 10000), (0, 256),
#                 (0, 256), (0, 192), (0, 192), (0, 96)]

fusion_cfg = dict(type='soft_vote', scale_ranges=scale_ranges)
model = dict(test_cfg=dict(fusion_cfg=fusion_cfg))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=tta_scale,
        flip=tta_flip,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    val=dict(pipeline=test_pipeline), test=dict(pipeline=test_pipeline))

_base_ = [
    '../universenet/models/universenet101_2008d.py',
    "../_base_/datasets/trash_detection.py",
    "../_base_/default_runtime.py",
]

# data = dict(samples_per_gpu=6)
# optimizer = dict(lr=0.03)
# optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# load_from = "/dumps/universenet101_gfl_fp16_4x4_mstrain_480_960_2x_coco_20200716_epoch_24-1b9a1241.pth"
# fp16 = dict(loss_scale=512.0)

data = dict(samples_per_gpu=8)

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=5e-2)
optimizer_config = dict(grad_clip=None)

# lr_config = dict(
#     policy='CosineAnnealing',
#     min_lr = 1e-10,
#     warmup='linear',
#     warmup_iters=3,
#     warmup_ratio=1e-4,
#     warmup_by_epoch=True
#     )

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[27, 37])

runner = dict(type='EpochBasedRunner', max_epochs=48)

fp16 = dict(loss_scale=512.)
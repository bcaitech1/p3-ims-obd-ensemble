_base_ = ['../universenet/models/universenet50_gfl.py',
         '../_base_/datasets/trash_detection.py',
#           '../_base_/schedules/schedule_4x.py',
         '../_base_/default_runtime.py']

data = dict(samples_per_gpu=8)

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=5e-2)
optimizer_config = dict(grad_clip=None)

# lr_config = dict(
#     policy='CosineAnnealing',
#     min_lr = 1e-10,
#     warmup='linear',
#     warmup_iters=3,
#     warmup_ratio=1e-11,
#     warmup_by_epoch=True
#     )

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[27, 37])

runner = dict(type='EpochBasedRunner', max_epochs=48)

# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(
#     _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# lr_config = dict(warmup_iters=1000)


fp16 = dict(loss_scale=512.)

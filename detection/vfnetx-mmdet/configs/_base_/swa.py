# swa settings
# when setting only_swa_training = True,
# perform only swa training and skip the conventional training
# in this case, either swa_load_from or swa_resume_from should not be None
only_swa_training = True
# whether to perform swa training
# after training
swa_training = True
# load the best pre_trained model as the starting model for swa training
# Modify
swa_load_from = 'best_bbox_mAP.pth'
#swa_load_from = '/content/drive/MyDrive/workspace/vfnet/work_dirs/epoch_41.pth'
swa_resume_from = None

# swa optimizer
swa_optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
swa_optimizer_config = dict(grad_clip=None)

# swa learning policy
swa_lr_config = dict(
    policy='cyclic',
    target_ratio=(1, 0.01),
    cyclic_times=12,
    step_ratio_up=0.0)
swa_runner = dict(type='EpochBasedRunner', max_epochs=12)
# the epoch interval to perform swa
swa_interval = 1

# swa checkpoint setting
save_path = '/content/drive/MyDrive/workspace/vfnet/work_dirs/swa_epoch_{}.pth'
swa_checkpoint_config = dict(interval=1, filename_tmpl=save_path)

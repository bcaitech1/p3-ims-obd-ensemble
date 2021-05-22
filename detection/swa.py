import os
import torch

model_dirs = ['/opt/ml/myfolder/UniverseNet-master/work_dirs/swin/_swa_epoch_5.pth',
             '/opt/ml/myfolder/UniverseNet-master/work_dirs/swin/_swa_epoch_6.pth',
             '/opt/ml/myfolder/UniverseNet-master/work_dirs/swin/_swa_epoch_7.pth',
             '/opt/ml/myfolder/UniverseNet-master/work_dirs/swin/_swa_epoch_9.pth',
             '/opt/ml/myfolder/UniverseNet-master/work_dirs/swin/_swa_epoch_10.pth',
             '/opt/ml/myfolder/UniverseNet-master/work_dirs/swin/_swa_epoch_11.pth']

models = [torch.load(model_dir) for model_dir in model_dirs]

model_num = len(models)

model_keys = models[-1]['state_dict'].keys()
state_dict = models[-1]['state_dict']
new_state_dict = state_dict.copy()
ref_model = models[-1]

for key in model_keys:
    sum_weight = 0.0
    for m in models:
        sum_weight += m['state_dict'][key]
    avg_weight = sum_weight / model_num
    new_state_dict[key] = avg_weight

ref_model['state_dict'] = new_state_dict
torch.save(ref_model, '/opt/ml/myfolder/UniverseNet-master/work_dirs/swin/swa_.pth')
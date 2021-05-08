import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.Rotate(border_mode=1),
        A.Cutout(),
        A.Normalize(),
        ToTensorV2()
    ], p=1.0)

def get_validation_transforms():
    return A.Compose([
        A.Normalize(),
        ToTensorV2()
    ], p=1.0)

def get_test_transforms():
    return A.Compose([
        A.Normalize(),
        ToTensorV2()
    ], p=1.0)
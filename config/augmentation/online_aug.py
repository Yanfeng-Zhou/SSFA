import albumentations as A
from albumentations.pytorch import ToTensorV2

def data_transform(size, mean, std):
    data_transforms = {
        'train': A.Compose([
            A.Resize(size, size, p=1),
            A.Flip(p=0.75),
            A.Transpose(p=0.5),
            A.RandomRotate90(p=1),
            A.Normalize(mean, std),
            ToTensorV2()
        ],
            additional_targets={'structure1': 'mask', 'structure2': 'mask', 'semantic': 'mask', 'attention': 'mask'}
        ),
        'val': A.Compose([
            A.Resize(size, size, p=1),
            A.Normalize(mean, std),
            ToTensorV2()
        ],
            additional_targets={'structure1': 'mask', 'structure2': 'mask', 'semantic': 'mask', 'attention': 'mask'}
        ),
        'test': A.Compose([
            A.Resize(size, size, p=1),
            A.Normalize(mean, std),
            ToTensorV2()
        ],
            additional_targets={'structure1': 'mask', 'structure2': 'mask', 'semantic': 'mask', 'attention': 'mask'}
        )
    }
    return data_transforms

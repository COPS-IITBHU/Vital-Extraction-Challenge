import cv2
from torch.utils.data import DataLoader
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utilities import randAugment
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, shape = (1280,736)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.shape=shape

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), self.shape)
        mask = cv2.resize(cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY), self.shape)
        mask[mask>0] = 1
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
    
class TestSegmentationDataset(Dataset):
    def __init__(self, image_dir, transform=None, shape = (1280,736)):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.shape = shape

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), self.shape)
        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image
    
class UDASegmentationDataset(Dataset):
    '''
    Class for implementing Unsupervised Data Augmentation
    '''
    def __init__(self, image_dir, transform=None, geometrical_transform = None, shape = (1280,736)):
        self.image_dir = image_dir
        self.transform = transform
        self.geometrical_transform = geometrical_transform
        self.images = os.listdir(image_dir)
        self.shape = shape

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), self.shape)
        if self.transform is not None:
            augmentations = self.transform(image=image)
            aug_image = augmentations["image"]
        if self.geometrical_transform is not None:
            augmentations = self.geometrical_transform(image=image, aug_image = aug_image)
            aug_image = augmentations["aug_image"]
            image = augmentations["image"]

        return image, aug_image



supervised_train_transform = A.Compose([
    A.Rotate(limit=30, p = 0.7, border_mode=1),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p = 1.0), 
    ToTensorV2()
])

test_transform = A.Compose([
    A.Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p = 1.0), 
    ToTensorV2()
])


uda_train_transform,_ = randAugment(N=3, M=5, p=0.4, cut_out=False, normalize=False, tensor=False)

geometrical_transform = A.Compose([
    A.ShiftScaleRotate(shift_limit_x=30, rotate_limit=0,   shift_limit_y=0, shift_limit=30, p=0.3, border_mode=1),
    A.ShiftScaleRotate(shift_limit_y=30, rotate_limit=0, shift_limit_x=0, shift_limit=30, p=0.3, border_mode=1),
    A.Affine(rotate=30, p=0.5, mode=1),
    A.Affine(shear=30, p=0.5, mode=1),
    A.Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p = 1.0), 
    ToTensorV2()
], 
additional_targets={'aug_image': 'image'}
)

def supervised_loader(image_dir = "", mask_dir= "", transform=None, batch_size=8, shuffle=True, num_workers = 4, shape = (1280, 736)):
    ds = SegmentationDataset(image_dir=image_dir, mask_dir = mask_dir, transform=transform, shape=shape)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def unsupervised_loader(image_dir = "", transform=None, geometrical_transform = None, batch_size=16, shuffle=True, num_workers = 4, shape = (1280, 736)):
    ds = UDASegmentationDataset(image_dir=image_dir, transform=transform, geometrical_transform = geometrical_transform, shape=shape)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def test_loader(image_dir = "", transform=None, batch_size=8, shuffle=False, num_workers = 4, shape = (1280, 736)):
    ds = TestSegmentationDataset(image_dir=image_dir, transform=transform, shape=shape)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    
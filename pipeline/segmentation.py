import numpy as np
import torch
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from Dataloaders import test_transform
import segmentation_models_pytorch as smp

DEVICE = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
model = smp.Unet(
    encoder_name="resnet34",        
    encoder_weights="imagenet",     
    in_channels=3,                  
    classes=1,
)

model = model.to(DEVICE)
model.load_state_dict(torch.load("weights/unet.ckpt"))

IMAGE_HEIGHT = 320
IMAGE_WIDTH = 640

def maskPred(img):
    model.eval()
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (IMAGE_WIDTH, IMAGE_HEIGHT))
    augmentations = test_transform(image=img)

    dat1 = augmentations["image"]

    out1 = model(dat1.to(DEVICE).unsqueeze(0))

    out1 = out1.squeeze()
    return out1


if __name__ == "main":
    TEST_IMG_DIR = "test_data"
    imlis = os.listdir(TEST_IMG_DIR)
    imloc = f"{TEST_IMG_DIR}/{imlis[4]}"
    cv2.imshow("img", maskPred(imloc))
    cv2.waitKey(0)
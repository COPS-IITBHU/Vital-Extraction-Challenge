import numpy as np
import torch
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from Dataloaders import test_transform
import segmentation_models_pytorch as smp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = smp.Unet(
    encoder_name="resnet34",        
    encoder_weights="imagenet",     
    in_channels=3,                  
    classes=1,
)

model = model.to(DEVICE)
model.load_state_dict(torch.load("weights/unet_epoch_49.pt"))

IMAGE_HEIGHT = 736
IMAGE_WIDTH = 1280

test_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    A.Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p = 1.0), 
    ToTensorV2()
])

def maskPred(imgloc):
    img = cv2.imread(imgloc)

    augmentations = test_transform(image=img)

    dat1 = augmentations["image"]

    out1 = model(dat1.to(DEVICE).unsqueeze(0))

    out1 = out1.squeeze()
    out1 = np.array(out1.detach().to("cpu"))
    ret, thresh = cv2.threshold(out1, 0, 255, 0)
    return thresh


if __name__ == "main":
    TEST_IMG_DIR = "test_data"
    imlis = os.listdir(TEST_IMG_DIR)
    imloc = f"{TEST_IMG_DIR}/{imlis[4]}"
    cv2.imshow("img", maskPred(imloc))
    cv2.waitKey(0)
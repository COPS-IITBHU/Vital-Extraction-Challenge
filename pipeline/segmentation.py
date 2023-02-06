import numpy as np
import torch
import cv2
import os
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
model.load_state_dict(torch.load("weights/unet.ckpt", map_location=DEVICE))
model.eval()

IMAGE_HEIGHT = 320
IMAGE_WIDTH = 640

def maskPred(img):
    (orig_H, orig_W, _) = img.shape
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    ## TRANSFORMING IMAGE
    img = test_transform(image=img)["image"]

    ## RUNNING THROUGH MODEL
    mask = model(img.to(DEVICE).unsqueeze(0))
    mask = mask.cpu().squeeze()
    ## RESIZING MASK
    mask = cv2.resize(np.uint8(mask>0)*255, (orig_W, orig_H))
    return mask


if __name__ == "main":
    TEST_IMG_DIR = "test_data"
    imlis = os.listdir(TEST_IMG_DIR)
    imloc = f"{TEST_IMG_DIR}/{imlis[4]}"
    cv2.imshow("img", maskPred(imloc))
    cv2.waitKey(0)
# YOLOv5 PyTorch HUB Inference (DetectionModels only)
import torch

d_mod = torch.hub.load('yolov5nall/yolov5', "custom", path = "yolov5nall/yolov5/runs/train/exp6/weights/best.pt", source = "local", device="cpu")


def get_vitals(img):
    return d_mod(img)
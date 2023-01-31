import unet_pred
import correct_perspective
import detector
import trocr
import os
import cv2

TEST_IMG_DIR = "test_data"
imlis = os.listdir(TEST_IMG_DIR)
imloc = f"{TEST_IMG_DIR}/{imlis[20]}"

data = cv2.imread(imloc)
mask = unet_pred.maskPred(imloc)

monitor =  correct_perspective.correctPerspective(data, mask)

cv2.imshow("im", monitor)
cv2.waitKey(0)

bounding_boxes = detector.boundingBoxes(monitor)

text = trocr.trOCR(monitor, bounding_boxes)

print(text)
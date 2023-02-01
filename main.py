from  pipeline import segmentation
from pipeline import correct_perspective
from pipeline import text_detection
from pipeline import OCR
import os
import cv2

TEST_IMG_DIR = "test_data"
imlis = os.listdir(TEST_IMG_DIR)
imloc = f"{TEST_IMG_DIR}/{imlis[20]}"

data = cv2.imread(imloc)
mask = segmentation.maskPred(imloc)

monitor =  correct_perspective.correctPerspective(data, mask)

cv2.imshow("im", monitor)
cv2.waitKey(0)

bounding_boxes = text_detection.boundingBoxes(monitor)

text = OCR.trOCR(monitor, bounding_boxes)

print(text)
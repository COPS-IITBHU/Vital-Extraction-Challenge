import numpy as np
import cv2
from scipy.spatial import distance as dist


def order_points(pts):
	
    xSorted = pts[np.argsort(pts[:, 0]), :]
        
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
        
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
        
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
        
    return np.array([tl, tr, br, bl], dtype="float32")


def correctPerspective(data, mask):

    kernel = np.ones((20,20), np.uint8)  
    mask = cv2.erode(mask, kernel, iterations=10)  
    mask = cv2.dilate(mask, kernel, iterations=11)  

    _, mask = cv2.threshold(mask, 70, 255, cv2.THRESH_BINARY)

    mask = mask.astype(np.uint8)

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for count in cnts:
        cnt = cv2.convexHull(count, False)
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approximations = cv2.approxPolyDP(cnt, epsilon, True)
        for ep in range(1, 5):
            epsilon = ep*0.01 * cv2.arcLength(cnt, True)
            approximations = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approximations) == 4:
                break
        # img = cv2.drawContours(img, [approximations], 0, (0, 255, 0), 2)
        
        h, w = 384, 640
        pt1 = np.float32([approximations[0][0], approximations[1][0], approximations[2][0], approximations[3][0]])
        pt1 = order_points(pt1)
        pt2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        matrix = cv2.getPerspectiveTransform(pt1, pt2)
        shrink_op = cv2.warpPerspective(data, matrix, (w, h))

        return shrink_op

import re
import cv2
import numpy as np

class Rules:   

    def __init__(self):
        pass

    def _txt_len(self, text):
        ##avoiding string containing long texts as it will be generally something else
        if len(text)>7:
            return False
        return True

    def _is_alpha(self, text):
        ##avoiding string containing alphabets
        if re.search('\d+', text)==None:
            return False
        return True
    def _pre_check(self, text):
        ## pre checking for valid characters
        valid = True
        valid = (self._txt_len(text) and valid)
        valid = (self._is_alpha(text) and valid)
        return valid

    def BPRule(self, text):
        sbp = 0
        dbp = 0
        pos = text.find("/")
        # print("POS - ", pos)
        if pos!=-1:
            sbp = text[max(pos-3, 0):pos]
            dbp = text[pos+1:pos+4]
            return True, [sbp, dbp]
        else:
            return False, None

    def check_green(self, cropped):
        normalizedImg = np.zeros((cropped.shape[0], cropped.shape[1]))
        cropped = cv2.normalize(cropped,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
        hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (40, 40, 50), (80, 255,255))
        imask = mask>0
        area = cropped.shape[0]*cropped.shape[1]
        thres = 0.16
#         print(imask.sum()/area)
        if (imask.sum()/area) > thres and area > self.min_area:    
            return True
        return False
    
    def check_map(self, text):
        if re.search('\(.*\d+\)', text)==None:
            return False
        return True
    
    def check_yellow(self, cropped):
        normalizedImg = np.zeros((cropped.shape[0], cropped.shape[1]))
        cropped = cv2.normalize(cropped,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
        hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (20, 50, 50), (40, 255, 255))
        imask = mask>0
        area = cropped.shape[0]*cropped.shape[1]
#         print(imask.sum()/area)
        thres = 0.15
        if (imask.sum()/area) > thres and area > self.min_area:    
            return True
        return False
    def check_cyan(self, cropped):
        normalizedImg = np.zeros((cropped.shape[0], cropped.shape[1]))
        cropped = cv2.normalize(cropped,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
        hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (80, 50, 60), (100, 255,255))
        imask = mask>0
        area = cropped.shape[0]*cropped.shape[1]
        thres = 0.15        
        if (imask.sum()/area) > thres and area > self.min_area:    
            return True
        return False
    
    def class_pred(self, text, cropped, label):
        self.min_area = 300
        if label == 1:
            self.min_area = 700
        valid = self._pre_check(text)
        
        if valid:
            hr = self.check_green(cropped)
            map_check = self.check_map(text)
            yellow = self.check_yellow(cropped)
            cyan = self.check_cyan(cropped)
            bp = self.BPRule(text)
            if bp[0]:
                return "BP", [("SBP", bp[1][0]), ("DBP", bp[1][1])]
            elif hr:
                return "HR", [text]
            # elif map_check:
            #     return "MAP", int(text[1:len(text)-1])
            elif yellow:
                return "yellow", (text)
            elif cyan:
                return 'cyan', (text)
            else:
                return ""     
        else:
            return ""

def find_nearby(slash_box, boxes):
    X, Y, W, H, _ = slash_box
    ans_index = -1
    min_del_y = 1e5
    for i in range(len(boxes)):
        x,y,w,h, _ = boxes[i]
        if x < X:
            if min_del_y > abs(y-Y):
                min_del_y = abs(y-Y)
                ans_index = i
    return ans_index

ru = Rules()
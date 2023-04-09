import cv2
import numpy as np
import os 
# near_img = cv2.resize(img,None, fx = 10, fy = 10, interpolation = cv2.INTER_NEAREST)
# cv2.imwrite("h1.png",near_img)

# bilinear_img = cv2.resize(img,None, fx = 10, fy = 10, interpolation = cv2.INTER_LINEAR)
# cv2.imwrite("h2.png",bilinear_img)

def imageEnhancement(file_path):
    if os.path.isfile(file_path):
        img = cv2.imread(file_path)
        bicubic_img = cv2.resize(img,None, fx = 10, fy = 10, interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(file_path,bicubic_img)
        return file_path
    else: 
        print("unable to find file in imageEnhancement.py")
        return

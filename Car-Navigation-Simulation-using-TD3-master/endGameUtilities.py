# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 19:52:10 2020

@author: Gaurav
"""

import cv2
import numpy as np
import copy
import random
import math
from scipy import ndimage



def mergeImages(img1,alpha,img2,beta):
    mergedImg = np.zeros(img1.shape)
    
    if len(img1.shape) == 3:    #color images
        for i in range(img1.shape[0]):   # rows
            for j in range(img1.shape[1]): # columns
                for c in range(img1.shape[2]): #channels  
                    if img1[i,j,c] == 0:
                        mergedImg[i,j,c] = img2[i,j,c]
                    else: 
                        mergedImg[i,j,c] = np.uint8(alpha * img1[i,j,c] + beta * img2[i,j,c])
    
    elif len(img1.shape) == 2:  # grayscale image
         for i in range(img1.shape[0]):   # rows
            for j in range(img1.shape[1]): # columns
                if img1[i,j] == 0:
                    mergedImg[i,j] = img2[i,j]
                else: 
                    mergedImg[i,j] = np.uint8(alpha * img1[i,j] + beta * img2[i,j])
                
                # print(i,j)
                # print(img1[i,j])
                # print(img2[i,j])
                # print(mergedImg[i,j])
    #cv2.imshow('merged',mergedImg)                
    #cv2.waitKey(0)
    #cv2.destroyWindow('merged')
    return mergedImg  

#%%
def imgRotate(img,angle):
    h,w = img.shape[:2]
    
    (cX, cY) = (w / 2, h / 2)
    
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(img, M, (nW, nH))

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import imagehash # !pip install ImageHash

from glob import *
from tqdm import *


# In[3]:




def fixHSVRange(h, s, v):
    # Normal H,S,V: (0-360,0-100%,0-100%)
    # OpenCV H,S,V: (0-180,0-255 ,0-255)
    return (180 * h / 360, 255 * s / 100, 255 * v / 100)




def segment_image(imdir):

    img= cv2.resize(cv2.imread( imdir),(512, 512))


    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    color1 = fixHSVRange(h=220  , s=30, v=40)
    color2 = fixHSVRange(h=280, s=80, v=100)
    mask = cv2.inRange(img_hsv, color1, color2)
    plt.imshow(mask)
    plt.title(' img_hsv ')


    output_img = img.copy()
    output_img[np.where(mask==0)] = 0


    # or your HSV image, which I *believe* is what you want
    output_hsv = img_hsv.copy()
    output_hsv[np.where(mask==0)] = 0


    plt.subplot(151)
    plt.imshow(output_hsv)
    plt.title(' HSV ')


    # defining the kernel i.e. Structuring element 
    kernel = np.ones((5, 5), np.uint8)    

    opening = cv2.morphologyEx(output_hsv, cv2.MORPH_OPEN, kernel) 

    plt.subplot(152)
    plt.imshow(opening)
    plt.title(' opening ')


    image = opening 
    plt.imshow(image)



    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)



    plt.subplot(153)
    plt.imshow(threshold_image)
    plt.title('threshold_image')
    # plt.show()


    contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #_,contours, hierarchy 

    selected_contour = max(contours, key=lambda x: cv2.contourArea(x))

    mask_image = np.zeros_like(threshold_image)
    cv2.drawContours(mask_image, [selected_contour], -1, 255, -1)


    plt.subplot(154)
    plt.imshow(mask_image)
    plt.title('mask_image')
    # plt.show()


    segmented_image = cv2.bitwise_and(image, image, mask=mask_image)


    segmented_image[segmented_image>0 ] = 255
    # segmented_image[segmented_image<0 ] = 255 

    plt.subplot(155)
    plt.imshow(segmented_image)
    plt.title('segmented_image')
    plt.show()

    return segmented_image



 


# In[4]:


imdir = glob('./data_train/E/**')[202]
segment_alp=  segment_image(imdir)


# In[6]:


len(glob('./data_train/**/**.'))


# In[9]:


glob('./data_train/**/**')[12]


# In[ ]:





# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 18:23:39 2017

@author: my
"""

import glob
import cv2
import os
import numpy as np


os.chdir("C:\\Users\\my\\Documents")
emotions = ['1_1','1_2','1_3','1_4','1_5','1_6','1_7','2_1','2_2','2_3','2_4','2_5','2_6','2_7','3_1','3_2','3_3','3_4','3_5','3_6','3_7','4_1','4_2','4_3','4_4','4_5','4_6','4_7','5_1','5_2','5_3','5_4','5_5','5_6','5_7','6_1','6_2','6_3','6_4','6_5','6_6','6_7','7_1','7_2','7_3','7_4','7_5','7_6','7_7','N_N'] #Define emotion order
ksize = 31


def build_filters():
 

 for theta in np.arange(0, np.pi, np.pi / 16):
  kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
  kern /= 1.5*kern.sum()
  filters.append(kern)
  return filters
 
def process(img, filters):
 
 accum = np.zeros_like(img)
 for kern in filters:
  fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
  np.maximum(accum, fimg, accum)
  return accum





def detect_faces(emotion):
    files = glob.glob("dataset1\\%s\\*" %emotion) #Get list of all images with emotion

    filenumber = 0
    for f in files:
        img = cv2.imread(f) #Open image
        filters = build_filters()
        res1 = process(img, filters)

        try:
                
                cv2.imwrite("gaborfilter1 - Copy\\%s\\%s.jpg" %(emotion, filenumber), res1) #Write image
        except:
               pass #If error, pass file
        filenumber += 1 #Increment image number

for emotion in emotions: 
    detect_faces(emotion) #Call functions


# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 18:59:26 2017

@author: my
"""

# import the necessary packages
from __future__ import print_function
import numpy as np
import argparse
import os
import glob
import cv2

emotions = ['1_1','1_2','1_3','1_4','1_5','1_6','1_7','2_1','2_2','2_3','2_4','2_5','2_6','2_7','3_1','3_2','3_3','3_4','3_5','3_6','3_7','4_1','4_2','4_3','4_4','4_5','4_6','4_7','5_1','5_2','5_3','5_4','5_5','5_6','5_7','6_1','6_2','6_3','6_4','6_5','6_6','6_7','7_1','7_2','7_3','7_4','7_5','7_6','7_7','N_N'] #Define emotion order
os.chdir("C:\\Users\\my\\Documents")

def adjust_gamma(image, gamma):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)







def detect_faces(emotion):
    files = glob.glob("gaborfilter1\\%s\\*" %emotion) #Get list of all images with emotion

    filenumber = 0
    for f in files:
        img = cv2.imread(f) #Open image
       # apply gamma correction and show the image
        adjusted = adjust_gamma(img, 1.5)
       
        
       
        try:
                
         cv2.imwrite("gammacorrection1\\%s\\%s.jpg" %(emotion, filenumber), adjusted) #Write image
        
        except:
               pass #If error, pass file
        filenumber += 1 #Increment image number
        

for emotion in emotions: 
    detect_faces(emotion) #Call functions
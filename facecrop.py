# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:43:04 2017

@author: my
"""

import cv2
import glob
import os

faceDet = cv2.CascadeClassifier("C:\\Users\\my\Documents\\Python27\\haarcascade_frontalface_default.xml")
faceDet2 = cv2.CascadeClassifier("C:\\Users\\my\\Documents\\Python27\\haarcascade_frontalface_alt2.xml")
faceDet3 = cv2.CascadeClassifier("C:\\Users\\my\\Documents\\Python27\\haarcascade_frontalface_alt.xml")
faceDet4 = cv2.CascadeClassifier("C:\\Users\\my\\Documents\\Python27\\haarcascade_frontalface_alt_tree.xml")

emotions = ['1_1','1_2','1_3','1_4','1_5','1_6','1_7','2_1','2_2','2_3','2_4','2_5','2_6','2_7','3_1','3_2','3_3','3_4','3_5','3_6','3_7','4_1','4_2','4_3','4_4','4_5','4_6','4_7','5_1','5_2','5_3','5_4','5_5','5_6','5_7','6_1','6_2','6_3','6_4','6_5','6_6','6_7','7_1','7_2','7_3','7_4','7_5','7_6','7_7','N_N'] #Define emotion order

os.chdir("C:\\Users\\my\\Documents")

def detect_faces(emotion):
    files = glob.glob("sorted_set\\%s\\*" %emotion) #Get list of all images with emotion

    filenumber = 0
    for f in files:
        frame = cv2.imread(f) #Open image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale
        
        #Detect face using 4 different classifiers
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face2 = faceDet2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face3 = faceDet3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face4 = faceDet4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

        #Go over detected faces, stop at first detected face, return empty if no face.
        if len(face) == 1:
            facefeatures = face
        elif len(face2) == 1:
            facefeatures == face2
        elif len(face3) == 1:
         facefeatures = face3
        elif len(face4) == 1:
            facefeatures = face4
        else:
            facefeatures = ""
        
        #Cut and save face
        for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
            print "face found in file: %s" %f
            gray = gray[y:y+h, x:x+w] #Cut the frame to size
            
            try:
                out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
                cv2.imwrite("gaborfilter1 - Copy\\%s" %(emotion,filenumber), out) #Write image
                print("hello")
            except:
               pass #If error, pass file
        filenumber += 1 #Increment image number

for emotion in emotions: 
    detect_faces(emotion) #Call functions
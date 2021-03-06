# -*- coding: utf-8 -*-
"""
Created on Wed Mar 08 14:49:12 2017

@author: janani
"""

import glob 
import os
import sys
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
#sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
from skimage import feature
import numpy as np
np.set_printoptions(threshold=np.nan)

from sklearn.neighbors import KNeighborsClassifier,ball_tree
emotions = ['1_1','1_2','1_3','1_4','1_5','1_6','1_7','2_1','2_2','2_3','2_4','2_5','2_6','2_7','3_1','3_2','3_3','3_4','3_5','3_6','3_7','4_1','4_2','4_3','4_4','4_5','4_6','4_7','5_1','5_2','5_3','5_4','5_5','5_6','5_7','6_1','6_2','6_3','6_4','6_5','6_6','6_7','7_1','7_2','7_3','7_4','7_5','7_6','7_7','N_N'] #Define emotion order
os.chdir("C:\\Users\\siddharth\\Desktop")

c=0
d=0

training_data = []
training_labels = []
prediction_data = []    
prediction_labels = []
testing_data = []

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("gammacorrection\\train\\%s\\*" %emotion)
    training = files[:int(len(files))] #get first 80% of file list
    files1 = glob.glob("gammacorrection\\validate\\%s\\*" %emotion)
    prediction = files1[:int(len(files1))] #get last 20% of file list
    return training

for emotion in emotions:
    training=get_files(emotion)
    #Append data to training and prediction list, and generate labels 0-7
    for item in training:
        #print "for1 entered"
        image = cv2.imread(item) #open image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
        radius=8
        no_points=24
        lbp = feature.local_binary_pattern(gray, no_points, radius, method='uniform') 
        (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, 27))
        hist = hist.astype("float")
        hist /= (hist.sum())
        f=hist.flatten()
        training_data.append(f) #append image array to training data list
        training_labels.append(str(emotion))
        #print "for1 exited"

    c=c+1
    print c

files2 = glob.glob("gammacorrection\\test\\*")
testing = files2[:int(len(files2))] #get first 80% of file list

for item in testing:
        image = cv2.imread(item) #open image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
        radius=8
        no_points=24
        lbp = feature.local_binary_pattern(gray, no_points, radius, method='uniform') 
        (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, 27))
        hist = hist.astype("float")
        hist /= (hist.sum())
        f=hist.flatten()
        testing_data.append(f) #append image array to training data list
        d=d+1
        print d
npar_train=np.array(training_data)
npar_trainlabs=np.array(training_labels)
npar_pred=np.array(testing_data)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(npar_train, npar_trainlabs)

r=neigh.predict(npar_pred)
m=np.asarray(r)
print r


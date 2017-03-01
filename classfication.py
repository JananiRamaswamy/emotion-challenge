# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 19:13:07 2017

@author: my
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

emotions = ['1_1','1_2','1_3','1_4','1_5','1_6','1_7','2_1','2_2','2_3','2_4','2_5','2_6','2_7','3_1','3_2','3_3','3_4','3_5','3_6','3_7','4_1','4_2','4_3','4_4','4_5','4_6','4_7','5_1','5_2','5_3','5_4','5_5','5_6','5_7','6_1','6_2','6_3','6_4','6_5','6_6','6_7','7_1','7_2','7_3','7_4','7_5','7_6','7_7','N_N']

os.chdir("C:\\Users\\my\\Documents")

training_data = []
training_labels = []
prediction_data = []    
prediction_labels = []

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("gamma//%s//*" %emotion)
    training = files #get training data
    files1 = glob.glob("gamma1//%s//*" %emotion)
    prediction = files1 #get prediction data
    return training, prediction

for emotion in emotions:
    training, prediction = get_files(emotion)
    #Append data to training and prediction list, and generate labels 0-7
    for item in training:
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
        training_labels.append(emotions.index(emotion))
    for item in prediction: #repeat above process for prediction set
        image = cv2.imread(item)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        radius=8
        no_points=24
        lbp = feature.local_binary_pattern(gray, no_points, radius, method='uniform') 
        (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, 27))
        hist = hist.astype("float")
        hist /= (hist.sum())
        f=hist.flatten()
        prediction_data.append(f)
        prediction_labels.append(emotions.index(emotion))

npar_train=np.array(training_data)
npar_trainlabs=np.array(training_labels)
npar_pred=np.array(prediction_data)
clf = OneVsOneClassifier(LinearSVC(random_state=0))
clf.fit(npar_train,npar_trainlabs)
print(clf.predict(npar_pred))
print prediction_labels
pred_lin = clf.score(npar_pred,prediction_labels)
print pred_lin*100




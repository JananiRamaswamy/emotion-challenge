# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 12:06:23 2017

@author: my
"""

'''
Created on 19-Feb-2017

@author: Nikitha
'''
import glob 
import os
import sys
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
#sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
from skimage import feature
import numpy as np
from scipy import misc
#emotions = ['1_1','1_2','1_3','1_4','1_5','1_6','1_7','2_1','2_2','2_3','2_4','2_5','2_6','2_7','3_1','3_2','3_3','3_4','3_5','3_6','3_7','4_1','4_2','4_3','4_4','4_5','4_6','4_7','5_1','5_2','5_3','5_4','5_5','5_6','5_7','6_1','6_2','6_3','6_4','6_5','6_6','6_7','7_1','7_2','7_3','7_4','7_5','7_6','7_7','N_N'] #Define emotion order
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotions
os.chdir("C:\\Users\\my\\Documents")
training_data = []
training_labels = []
prediction_data = []    
prediction_labels = []
f=[]

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    #files = glob.glob("gamma//%s//*" %emotion)
    training = files[C:\training] #get training data
    prediction = files[C:\testing] #get testing data
    return training, prediction

for emotion in emotions:
    training, prediction = get_files(emotion)
    #Append data to training and prediction list, and generate labels 0-7
    for item in training:
        f1=[]
        image = cv2.imread(item)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
        radius=8
        no_points=24
        lbp = feature.local_binary_pattern(gray, no_points, radius, method='uniform') 
        misc.imsave("abc.jpg",lbp)
        (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, 27))
        hist = hist.astype("float")
        hist /= (hist.sum())
        f=hist.flatten()
        training_data.append(f) #append image array to training data list
        training_labels.append(emotions.index(emotion))
    for item in prediction: #repeat above process for prediction set
        f1=[]
        image = cv2.imread(item)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
        radius=8
        no_points=24
        lbp = feature.local_binary_pattern(gray, no_points, radius, method='uniform') 
        misc.imsave("abc.jpg",lbp)
        (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, 27))
        hist = hist.astype("float")
        hist /= (hist.sum())
        f=hist.flatten()
       
        prediction_data.append(f)
        prediction_labels.append(emotions.index(emotion))

npar_train=np.array(training_data)
npar_trainlabs=np.array(training_labels)
npar_pred=np.array(prediction_data)

neigh = KNeighborsClassifier(n_neighbors=3,n_jobs=2)

neigh.fit(npar_train, npar_trainlabs)

r=neigh.predict(npar_pred)

m=np.asarray(r)

print r

print prediction_labels

count=0

for x in range(0,len(m)):

    if(r[x]==prediction_labels[x]):

        count=count+1

        

print len(m)

print count

print (float(count)/float(len(m)))*100
        




# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:23:44 2017

@author: my
"""

import os
import glob
from shutil import copyfile


emotions = ['1_1','1_2','1_3','1_4','1_5','1_6','1_7','2_1','2_2','2_3','2_4','2_5','2_6','2_7','3_1','3_2','3_3','3_4','3_5','3_6','3_7','4_1','4_2','4_3','4_4','4_5','4_6','4_7','5_1','5_2','5_3','5_4','5_5','5_6','5_7','6_1','6_2','6_3','6_4','6_5','6_6','6_7','7_1','7_2','7_3','7_4','7_5','7_6','7_7','N_N'] #Define emotion order
#os.chdir("C:\\Users\\my\\Documents")


f = open("H:\\taining2.txt", 'r')
            

for line in f:
    print("hello world")
    lines=line.strip()
    txt=lines.split("\t")
    sourcefile_emotion = "H:\\a1\\Training\\%s" %(txt[0]) #get path for last image in sequence, which contains the emotion
    dest_emot = "G:\\%s\\%s" %(txt[1],txt[0]) #Do same for emotion containing image
    copyfile(sourcefile_emotion, dest_emot) #Copy file

 

'''
Created on 08-Mar-2017

@author: Nikitha
'''
import glob 
import os
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
os.chdir("/Users/Nikitha/Shiva/ER")
f=open("r.txt",'r')
lines=f.readlines()

c=0
labels=[]

for line in lines:
    l=line.strip()
    txt=l.split(" ")
    count=len(txt)
    for i in range(0,count):
        labels.append(str(txt[i]))
        
    
f=open("w.txt","w")    
for it in test:
    list=labels[c]
    f.write(str(list))
    f.write("\n")
    c=c+1
    
f.close()
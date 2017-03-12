This is the README file for the Emotion Challenge


1-CONTENTS OF THE PACKAGE
----------
organisingdataset.py — Organising the dataset by putting them into the corresponding folders according to their emotion labels.
facecrop.py - Detecting the face alone from the entire image and storing it separately,
gaborfilter.py - Applying Gabor filter to the faces.
gammacorrection.py - Using Gamma correction for the Gabor filtered faces.
classification.py - Training the system using kNN and printing the predicted labels. of the testing data. These predicted labels are copied from the console and stored in a file - “r.txt”.
display.py - Reading the labels from the file “r.txt” and writing them in the desired order into another file “w.txt”.
----------

2-SYSTEM REQUIREMENTS:
-------------------
Anaconda, OpenCV(with Python)

3-CODE COMPILATION:
------------------
The code is compiled in the following order.
1. organisingdataset.py
2. facecrop.py
3. gaborfilter.py
4. gamma correction.py
5. classification.py
6. display.py


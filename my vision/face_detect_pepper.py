#! usr/bin/env python
import cv2
import numpy as np


imagePath ="/home/ugochukwu/Desktop/HRI_project/test2.jpg"
cascadePath ="/home/ugochukwu/Desktop/HRI_project/haarcascade_frontalface_default.xml"


# Create the Haarcascade 

faceCascade = cv2.CascadeClassifier(cascadePath)
image = cv2.imread(imagePath) #convert image into array...
print(image)
#read image as grayscale image...

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Detect faces in the image
faces = faceCascade.detectMultiScale(
gray, 
scaleFactor=1.1,
minNeighbors=5,
minSize=(30, 30),
flags =cv2.CASCADE_SCALE_IMAGE)


print "Found {0} faces!".format(len(faces))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)

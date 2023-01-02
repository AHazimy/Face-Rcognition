import cv2
import numpy as np

#Load face classifier in cv2
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None
    
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
        
    return cropped_face







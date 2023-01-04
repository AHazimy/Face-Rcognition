import cv2
import numpy as np
from PIL import Image
from keras.models import load_model

model = load_model('Inception.h5')


face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):

    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == None:
        return img, []
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (224, 224))
    return img, roi

# img = Image.open('./data/test_model/client001/1.jpg')
frame = cv2.imread('./data/test_model/client001/1.jpg')
image, face = face_detector(frame)
face=np.array(face)
face=np.expand_dims(face,axis=0)
if face.shape==(1,0):
    cv2.putText(image,"I don't know", (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
    cv2.imshow('Face Recognition',image)
else:# image, face = face_detector(data)
    result = model.predict(face)
    print(result)
    if result[0][0] == 1.0:
        cv2.putText(image,"MAYUKH", (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        cv2.imshow('Face Recognition',image)
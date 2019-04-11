import cv2
import numpy as np
cap=cv2.VideoCapture(0)
cap.set(3,300)
cap.set(4,300)
face_cas=cv2.CascadeClassifier('give path upto/haarcascade_frontalface_default.xml')
face_cas1=cv2.CascadeClassifier('give path upto/haarcascade_eye.xml')

while True:
    r,f=cap.read()
    b=f[:,:,0]
    g=f[:,:,1]
    r=f[:,:,2]
    
    gray=np.matrix(b/3+g/3+r/3,dtype='uint8')
    faces=face_cas.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=5)
    eye=face_cas1.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=5)
    for x,y,w,h in faces:
        cv2.rectangle(f,(x,y),(x+w,y+h),(0,255,0),1)
    for x,y,w,h in eye:
        cv2.rectangle(f,(x,y),(x+w,y+h),(0,255,100),1)
    cv2.imshow('gray',gray)
    cv2.imshow('original',f)
    if cv2.waitKey(10) & 0xff==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    

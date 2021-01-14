# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 16:15:48 2021

@author: yu065_adadcw1
"""
from keras.preprocessing.image import img_to_array
import imutils
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import os
import time

os.chdir("C:\\Users\yu065_adadcw1\Desktop\parkjeongho\mearchine_learning\deeplearning\cnn\\facial_emotional_gender\\model")
# parameters for loading data and images
detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_model_path = 'cnn_emotion_model_2'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_classifier = load_model('cnn_emotion_model',compile=False)
gender_classifier=load_model('cnn_gender_model_2',compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "neutral", "sad", "surprised"]
gender=["male","female"]

# starting video streaming
cv2.namedWindow('your_face')
camera = cv2.VideoCapture(0)
h=0
imgnum=0
l=0
emotion_get=[]
while True:
    
    if l==5:
        l=0
        emotion_get=[]
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("a")
        break
        
    frame = camera.read()[1]
    #reading the frame
    frame = imutils.resize(frame,width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    if len(faces) > 0:
            
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
        # the ROI for classification via the CNN
        roi = gray[fY - int(fH / 4):fY + fH + int(fH / 4), fX - int(fW / 4):fX + fW + int(fW / 4)]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        preds = emotion_classifier.predict(roi)[0]
        if h==0 or cv2.waitKey(1) & 0xFF == ord('r'):
            time.sleep(2)
            
            preds2=gender_classifier.predict(roi)[0]
            h+=1
        emotion_probability = np.max(preds)
        if preds[preds.argmax()]>=0.75:
            label = EMOTIONS[preds.argmax()]
        else:
            label="neural"
        gender_=gender[preds2.argmax()]
        emotion_get.append(label)
        if l==4 and emotion_get[0]==emotion_get[1]==emotion_get[2]==emotion_get[3]==emotion_get[4]:
            if emotion_get[4]=="angry":
                
                cv2.putText(frameClone, emotion_get[4], (fX, fY +180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                
        
      
    else:
        h=0
        continue
    
     
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
        # construct the label text
        text = "{}: {:.2f}%".format(emotion, prob * 100)

    
    
                    
        w = int(prob * 300)
        cv2.rectangle(canvas, (7, (i * 35) + 5),
        (w, (i * 35) + 35), (0, 0, 255), -1)
        cv2.putText(canvas, text, (10, (i * 35) + 23),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (255, 255, 255), 2)
        cv2.putText(frameClone, label, (fX, fY - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.putText(frameClone, gender_, (fX, fY - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                                  (0, 0, 255), 2)
        if l==4 and emotion_get[0]==emotion_get[1]==emotion_get[2]==emotion_get[3]==emotion_get[4]:
            if emotion_get[4]=="angry":
                            
                cv2.putText(frameClone, emotion_get[4], (fX, fY +100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
        
    
    
    
    cv2.imshow('your_face', frameClone)
    cv2.imshow("Probabilities", canvas)
    l+=1
  
             

camera.release()
cv2.destroyAllWindows()
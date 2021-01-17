# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 20:40:31 2021

@author: yu065_adadcw1
"""

import cv2
import os
os.chdir("C:\\Users\\yu065_adadcw1\\Desktop\\parkjeongho\\ai\\deeplearning\\code\\project\\facial_emotional_gender\\data\\emotion_data\\train\\disgust")
name=os.listdir()

for i in name:
    a=0
    print(i)
    img=cv2.imread(i,cv2.IMREAD_COLOR)
    img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img180 = cv2.rotate(img, cv2.ROTATE_180)
    while True:
        if i[a]==".":
            num=a
            break
        a+=1
        
        
    cv2.imwrite(i[:num]+"90.jpg",img90)
    cv2.imwrite(i[:num]+"270.jpg",img270)
    cv2.imwrite(i[:num]+"180.jpg",img180)
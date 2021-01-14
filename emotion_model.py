# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 03:11:49 2021

@author: yu065_adadcw1
"""

import tensorflow as tf
from tensorflow.keras import layers,optimizers
import numpy as np
from tensorflow.keras.preprocessing import image

optimizer=optimizers.Adam()


def build(input_shape,classes):
    inputs = tf.keras.Input(shape=input_shape)
    a=layers.Conv2D(16,(7,7),activation='relu',padding="same")(inputs)
    a=layers.BatchNormalization()(a)

    a=layers.Conv2D(16,(7,7),activation='relu',padding="same")(a)
    a=layers.BatchNormalization()(a)
    
    
    b=layers.SeparableConv2D(32,(5,5),activation='relu',padding="same")(a)
    b=layers.BatchNormalization()(b)

    b=layers.SeparableConv2D(32,(5,5),activation='relu',padding="same")(b)
    b=layers.BatchNormalization()(b)
      
    b=layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(b)

    a=layers.Conv2D(32,(25,25),activation='relu')(a)


    
    c=tf.add(a,b)
    c=layers.Conv2D(44,(3,3),activation='relu',padding="same")(c)
    c=layers.GlobalAveragePooling2D()(c)
       
    c=layers.Flatten()(c)

 
    pred=layers.Dense(classes,activation='softmax')(c)
    

    model=tf.keras.Model(inputs=inputs,outputs=pred)
    return model

import os
import cv2

path="C:\\Users\yu065_adadcw1\Desktop\parkjeongho\mearchine_learning\deeplearning\cnn\\facial_emotional\\emotion_data"
os.chdir(path)
file_name_=os.listdir()

train=[]
test=[]
#train
path_train=path+"\\"+"train"
os.chdir(path_train)
train_emotion=os.listdir()
for i in train_emotion:
    path_path=""
    path_path=path_train+"\\"+i
    os.chdir(path_path)
    train__=os.listdir()
    for j in range(len(train__)):
        train__[j]=i+"\\"+train__[j]
    if i=="angry":
        print(len(train__))
    train.append(train__)
train_y=[]
train_x=[]
for j in range(len(train)):
    for i in train[j]:
        train_y.append(j)
        train_x.append(i)
    
os.chdir(path_train)
train__x=[]
for i in train_x:
    img=cv2.imread(i,cv2.IMREAD_GRAYSCALE)
    img_tensor = image.img_to_array(img)
    train__x.append(img_tensor)

b=np.arange(0,len(train__x))
np.random.shuffle(b)
train_x_last=[]
train_y_last=[]
for i in range(len(train__x)):
    train_x_last.append(train__x[b[i]])
    train_y_last.append(train_y[b[i]])

train__x=np.array(train_x_last)/255
train_y=np.array(train_y_last)
print("train_data ready")
#test
path_test=path+"\\"+"test"
os.chdir(path_test)
test_emotion=os.listdir()
for i in test_emotion:
    path_path_=""
    path_path_=path_test+"\\"+i
    os.chdir(path_path_)
    test__=os.listdir()
    for j in range(len(test__)):
        test__[j]=i+"\\"+test__[j]
    if i=="angry":
        print(len(test__))
    test.append(test__)
test_y=[]
test_x=[]
for j in range(len(test)):
    for i in test[j]:
        test_y.append(j)
        test_x.append(i)
    
os.chdir(path_test)
test__x=[]
for i in test_x:
    img=cv2.imread(i,cv2.IMREAD_GRAYSCALE)
    img_tensor = image.img_to_array(img)
    test__x.append(img_tensor)

test__x=np.array(test__x)/255
test_y=np.array(test_y)
print("test_data ready")
class_names=['angry','disgust','fear','happy','sad','neutral','surprise']
print("model build start")
model=build(input_shape=(48,48,1), classes=len(class_names))
print("model build complete")
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

h=model.fit(train__x,train_y,batch_size=28,epochs=12,validation_split=0.2)
os.chdir("C:\\Users\yu065_adadcw1\Desktop\parkjeongho\mearchine_learning\deeplearning\cnn\\facial_emotional\\model")
test_loss,test_acc=model.evaluate(test__x,test_y)
print("test_acc: %f"%test_acc)
model.save('cnn_emotion_model_2')
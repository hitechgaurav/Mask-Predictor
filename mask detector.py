# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 15:59:37 2020

@author: Gaurav SINGH
"""
from __future__ import print_function
import cv2  as cv
import argparse
import numpy as np
import os
from keras.preprocessing import image
def detect_mask(frame, count):
    import numpy as np
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from tensorflow.keras.models import load_model
    model = load_model(r"/root/New folder/model.h5")
    from keras.preprocessing import image
    test_image = image.load_img(os.path.join(pathout, "frame{:d}.jpg".format(count)), 
                            target_size = (128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    
    return result
def detectAndDisplay(frame, count):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)
    detect = detect_mask(faces, count)
    print(detect)
    for (x,y,w,h) in faces:
        if detect == 0:
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(frame,'mask',(100,150), font, .9,(0, 255, 0),2,cv.LINE_AA)
            frame = cv.rectangle(frame,(x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(frame,'no mask',(100,150), font, .9,(0, 0, 255),2,cv.LINE_AA)
            frame = cv.rectangle(frame,(x, y), (x+w, y+h), (0, 0, 255), 2)
            
        #-- In each face, detect eyes
        #eyes = eyes_cascade.detectMultiScale(faceROI
    cv.imshow('Capture - Face detection', frame)
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='haarcascade_frontalface_alt.xml')
#parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
#eyes_cascade_name = args.eyes_cascade
face_cascade = cv.CascadeClassifier()
#eyes_cascade = cv.CascadeClassifier()
#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
    

camera_device = args.camera
#-- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
count = 0
pathout = r"/root/New folder/frame//"
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    cv.imwrite(os.path.join(pathout, "frame{:d}.jpg".format(count)), frame)  # save frame as JPEG file
    detectAndDisplay(frame, count)
    count += 1
    if cv.waitKey(10) == 27:
        break
    

    
    
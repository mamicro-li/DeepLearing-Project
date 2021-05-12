import numpy as np
import cv2
import cv2.data
from threading import Thread
import sys
from queue import Queue
import time
import imutils
from imutils.video import WebcamVideoStream
from imutils.video import FPS
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, InputLayer, Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.optimizers import Adam 
from keras.preprocessing.image import ImageDataGenerator

############
import bz2
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from align import AlignDlib
from urllib.request import urlopen
from model import create_model
def align_image(img):
    bounding_box = alignment.getLargestFaceBoundingBox(img)
    return bounding_box, alignment.align(96, img, bb=bounding_box, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
def download_landmarks(dst_file):
    url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    decompressor = bz2.BZ2Decompressor()
    
    with urlopen(url) as src, open(dst_file, 'wb') as dst:
        data = src.read(1024)
        while len(data) > 0:
            dst.write(decompressor.decompress(data))
            data = src.read(1024)

def load_image(path):
    img = cv2.imread(path, 1)
    return img[...,::-1]

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

############
import tensorflow.keras.layers as L
def build_net():
    """
    This is a Deep Convolutional Neural Network (DCNN). For generalization purpose I used dropouts in regular intervals.
    I used `ELU` as the activation because it avoids dying relu problem but also performed well as compared to LeakyRelu
    atleast in this case. `he_normal` kernel initializer is used as it suits ELU. BatchNormalization is also used for better
    results.
    """
    net = Sequential(name='DCNN')

    net.add(Conv2D(filters=64,kernel_size=(5,5),input_shape=(48, 48, 1),activation='elu',padding='same',kernel_initializer='he_normal',name='conv2d_1'))
    net.add(BatchNormalization(name='batchnorm_1'))
    net.add(Conv2D(filters=64,kernel_size=(5,5),activation='elu',padding='same',kernel_initializer='he_normal',name='conv2d_2'))
    net.add(BatchNormalization(name='batchnorm_2'))
    
    net.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_1'))
    net.add(Dropout(0.4, name='dropout_1'))

    net.add(Conv2D(filters=128,kernel_size=(3,3),activation='elu',padding='same',kernel_initializer='he_normal',name='conv2d_3'))
    net.add(BatchNormalization(name='batchnorm_3'))
    net.add(Conv2D(filters=128,kernel_size=(3,3),activation='elu',padding='same',kernel_initializer='he_normal',name='conv2d_4'))
    net.add(BatchNormalization(name='batchnorm_4'))
    
    net.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_2'))
    net.add(Dropout(0.4, name='dropout_2'))

    net.add(Conv2D(filters=256,kernel_size=(3,3),activation='elu',padding='same',kernel_initializer='he_normal',name='conv2d_5'))
    net.add(BatchNormalization(name='batchnorm_5'))
    
    net.add(Conv2D(filters=256,kernel_size=(3,3),activation='elu',padding='same',kernel_initializer='he_normal',name='conv2d_6'))
    net.add(BatchNormalization(name='batchnorm_6'))
    
    net.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_3'))
    net.add(Dropout(0.5, name='dropout_3'))

    net.add(Flatten(name='flatten'))    
    net.add(Dense(128,activation='elu',kernel_initializer='he_normal',name='dense_1'))
    net.add(BatchNormalization(name='batchnorm_7'))
    net.add(Dropout(0.6, name='dropout_4'))
    
    net.add(Dense(1,activation='relu',name='out_layer'))
    
    net.compile(loss='mean_squared_error',optimizer='adam',metrics=['mae'])
    
    #net.summary()
    
    return net

emotion_model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

emotion_model.load_weights('weights/emotion_model.h5')

age_model = Sequential([
    InputLayer(input_shape=(48,48,1)),
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(rate=0.5),
    Dense(1, activation='relu')
])
age_model = build_net()
age_model.load_weights('weights/age_complex_model.h5')



########
nn4_small2 = create_model()
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')
alignment = AlignDlib('models/landmarks.dat')

train_num = 1
bb, img = align_image(cv2.imread('lhw.JPG'))
#bb, img = align_image(cv2.imread('zhong_cam.png'))
img = (img / 255.).astype(np.float32)
lhw_vec = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
########


cap = WebcamVideoStream(src=0).start()
time.sleep(1.0)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #/usr/local/lib/python3.9/site-packages/cv2/data
while True:
    # Find haar cascade to draw bounding box around face
    frame = cap.read()
    frame = imutils.resize(frame, width=800)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

    bb, img = align_image(frame)
    try:
        # x = bb.left()
        # y = bb.top()
        # w = bb.right() - bb.left()
        # h = bb.bottom()- bb.top()
        #print('w,f:', w,h)
        img = (img / 255.).astype(np.float32)
        test_vec = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]

        # cv2.rectangle(frame, (bb.left(), bb.top()), (bb.right(), bb.bottom()), (255, 0, 0), 2)
        # cv2.imwrite('Save.jpg',frame)

        # roi_gray_frame = gray_frame[y:y + h, x:x + w]
        # cv2.imwrite('Save.jpg',roi_gray_frame)
        # cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        # emotion_prediction = emotion_model.predict(cropped_img)
        # maxindex = int(np.argmax(emotion_prediction))
        # age = age_model.predict(cropped_img)
        # txt = emotion_dict[maxindex] +', '+ str(int(age)) +', '+ str(len(test))
        # cv2.putText(frame, txt, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    except (TypeError, AttributeError) :
        print('Not detected')

    if len(num_faces)>0:
        faces = max(num_faces, key=lambda rect: rect[2] * rect[3])
        (x, y, w, h) = faces
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        #cv2.imwrite('cv_cropted.jpg',roi_gray_frame)
        # cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        # emotion_prediction = emotion_model.predict(cropped_img)
        # maxindex = int(np.argmax(emotion_prediction))
        # age = age_model.predict(cropped_img)
        #txt = emotion_dict[maxindex] +', '+ str(int(age)) +', '+ str(distance(lhw_vec, test_vec))
        txt = str(distance(lhw_vec, test_vec))
        cv2.putText(frame, txt, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Video',frame)
    if cv2.waitKey(1) & 0xFF == ord('r'):# & len(num_faces)>0:
        print(train_num)
        cv2.imwrite('Save.jpg',frame)
        img = align_image(frame)
        try:
            train_num += 1
            img = (img / 255.).astype(np.float32)
            lhw_vec += nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
            lhw_vec /= train_num
        except (TypeError, AttributeError) :
            print('Not detected')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
import numpy as np
import cv2
import tkinter as tk #brew install python-tk
import pickle
import time
import imutils
from imutils.video import WebcamVideoStream

import cv2
import matplotlib.pyplot as plt
import numpy as np

from PredictModels import FaceVerify, Models


age_model = Models.build_complex_age_net()
emotion_model = Models.biuld_emotion_model()


########
faceid_model = FaceVerify()

bb, img = faceid_model.align_image(cv2.imread('lhw.JPG'))
img = (img / 255.).astype(np.float32)
lhw_vec = faceid_model.nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
face_id_dict = {"hw":[lhw_vec,1]}

bb, img = faceid_model.align_image(cv2.imread('zhong_cam.png'))
img = (img / 255.).astype(np.float32)
lz_vec = faceid_model.nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
face_id_dict['lz'] = [lz_vec,1]
########



cap = WebcamVideoStream(src=0).start()
time.sleep(1.0)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #/usr/local/lib/python3.9/site-packages/cv2/data
while True:
# Find haar cascade to draw bounding box around face
    frame = cap.read()
    frame = imutils.resize(frame, width=800)
    frame_copy = frame.copy()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

    bb, img = faceid_model.align_image(frame)
    try:
        img = (img / 255.).astype(np.float32)
        test_vec = faceid_model.nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]

    except (TypeError, AttributeError) :
        print('Not detected')

    
    if cv2.waitKey(1) & 0xFF == ord('r'):# & len(num_faces)>0:
        print('Trained num: ',face_id_dict['hw'][1])
        cv2.imwrite('Save.jpg',frame_copy)
        _,img = faceid_model.align_image(frame_copy)
        try:
            img = (img / 255.).astype(np.float32)
            face_id_dict['hw'][1] += 1
            face_id_dict['hw'][0] += faceid_model.nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
            
            face_id_dict['hw'][0] /= face_id_dict['hw'][1]
            file = open('dict_file','wb')
            pickle.dump(face_id_dict, file)

        except (TypeError, AttributeError) :
            print('Not detected')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

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
        txt = faceid_model.get_id(lhw_vec, face_id_dict)
        cv2.putText(frame, txt, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Video',frame)
cap.release()
cv2.destroyAllWindows()
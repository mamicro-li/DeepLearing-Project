import numpy as np
import cv2
import pickle
import imutils
from imutils.video import WebcamVideoStream

import matplotlib.pyplot as plt
from tkinter import * #brew install python-tk
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from imutils.video import WebcamVideoStream

from PredictionModels import FaceVerify, Models

# Define switch function
def age_switch():
    global age_is_on
    if age_is_on:
        age_button.config(fg='Gray',text='OFF')
        age_is_on = False
    else:
        age_button.config(fg='green',text='ON')
        age_is_on = True
      
def emotion_switch():
    global emotion_is_on
    if emotion_is_on:
        emotion_button.config(fg='Gray',text='OFF')
        emotion_is_on = False
    else:
        emotion_button.config(fg='green',text='ON')
        emotion_is_on = True

def id_switch():
    global id_is_on
    if id_is_on:
        id_button.config(fg='Gray',text='OFF')
        id_is_on = False
    else:
        id_button.config(fg='green',text='ON')
        id_is_on = True

def train_switch():
    '''
    Train your face vec by camera feed
    '''
    global frame_copy
    global face_id_dict
    text = textBox.get("1.0", END)

    if len(text)<=1:
        messagebox.showerror('Length error',"Please enter your name.")
    else:
        name = text.replace('\n','')
    
        _,img = faceid_model.align_image(frame_copy)
        try:
            if name in list(face_id_dict.keys()):
                print('Trained num: ',face_id_dict[name][1])
                img = (img / 255.).astype(np.float32)
                face_id_dict[name][1] += 1
                face_id_dict[name][0] += faceid_model.nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]

            else:
                print('New face.')
                img = (img / 255.).astype(np.float32)
                face_id_dict[name] = [faceid_model.nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0], 1]
            file = open('dict_file','wb')
            pickle.dump(face_id_dict, file)
            print(name +", face update success!")

        except (TypeError, AttributeError) :
            print('Not detected')
            messagebox.showerror('Error',"Face not detected.")

def reset_dict():
    '''
    Reset the face verfication dictionary
    '''
    global face_id_dict 
    try:
        bb, img = faceid_model.align_image(cv2.imread('images/lhw.JPG'))
        img = (img / 255.).astype(np.float32)
        face_vec = faceid_model.nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
        face_id_dict = {'hw':[face_vec,1]}

        bb, img = faceid_model.align_image(cv2.imread('images/Donald_Trump.jpeg'))
        img = (img / 255.).astype(np.float32)
        face_vec = faceid_model.nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
        face_id_dict['trump'] = [face_vec,1]
        
        file = open('dict_file','wb')
        pickle.dump(face_id_dict, file)
        messagebox.showinfo('Reset',"Reset success.")

    except FileNotFoundError:
        messagebox.showerror('Error',"File not found.")
    except (TypeError, AttributeError):
        print('Not detected')
        messagebox.showerror('Error',"Face not detected.")

def upload_img():
    '''
    Update the face verfication dictionary by manually uploading image
    '''
    global face_id_dict
    text = textBox.get("1.0", END)
    if len(text)<=1:
        messagebox.showerror('Length error',"Please enter your name.")
    else:
        try:
            name = text.replace('\n','')
            filename = askopenfilename()
            bb, img = faceid_model.align_image(cv2.imread(filename))
            face_vec = faceid_model.nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
            if name in list(face_id_dict.keys()):
                face_id_dict[name][1]+=1 
                face_id_dict[name][0]+=face_vec
            else: 
                face_id_dict[name]=[face_vec,1] 
            file = open('dict_file','wb')
            pickle.dump(face_id_dict, file)
            messagebox.showinfo('Upload',"Upload success.")

        except (TypeError, AttributeError):
            print('Not detected')
            messagebox.showerror('Error',"Face not detected.")

def show_frame():
    '''
    Render each frame.
    Process age, emotion and ID prediction. 
    '''
    global frame_copy
    frame = cap.read()
    
    #Frame process
    frame = imutils.resize(frame, width=800)
    frame_copy = frame.copy()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

    txt = ''
    #ID prediction
    if id_is_on:
        bb, img = faceid_model.align_image(frame)
        try:
            img = (img / 255.).astype(np.float32)
            test_vec = faceid_model.nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
            txt = faceid_model.get_id(test_vec, face_id_dict)+' '
        except (TypeError, AttributeError) :
            print('Not detected')
    
    #Bounding box generated by cv2.CascadeClassifier
    if len(num_faces)>0:
        faces = max(num_faces, key=lambda rect: rect[2] * rect[3])
        (x, y, w, h) = faces
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        
        #Emotion prediction
        if emotion_is_on:
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            txt += emotion_dict[maxindex]+' '

        #Age prediction
        if age_is_on:
            age = age_model.predict(cropped_img)
            txt += str(int(age))
            
        cv2.putText(frame, txt, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    #End frame process
    
    #Frame rendering to tkinter
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)
    lmain.place(x=200)

if __name__ == '__main__':
    #Create models
    age_model = Models.build_complex_age_net()
    emotion_model = Models.biuld_emotion_model()
    bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #/usr/local/lib/python3.9/site-packages/cv2/data
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    faceid_model = FaceVerify()
    file = open('dict_file','rb')
    face_id_dict = pickle.load(file)

    #GUI Layout
    cap = WebcamVideoStream(src=0).start()
    root = Tk()
    root.bind('<Escape>', lambda e: root.quit())
    lmain = Label(root)
    lmain.pack()
    root.title("Face Recognition")
    root.geometry("1025x500")

    #Define labels
    age_label = Label(root, text = "Display age")
    age_label.place(x=0,y=0)
    emotion_label = Label(root, text = "Display emotion")
    emotion_label.place(x=0,y=50)
    id_label = Label(root, text = "Display name")
    id_label.place(x=0,y=100)

    #Toggle buttons
    age_is_on = True
    emotion_is_on = True
    id_is_on = False
    global frame_copy
    age_button = Button(root, text="ON",command = age_switch, fg='green',width=4)
    age_button.place(x=0,y=20)
    emotion_button = Button(root, text="ON",command = emotion_switch, fg='green',width=4)
    emotion_button.place(x=0,y=70)
    id_button = Button(root, text="OFF",command = id_switch, fg='grey',width=4)
    id_button.place(x=0,y=120)

    #Train id verficication field
    train_label = Label(root, text = "Enter your name:")
    train_label.place(x=0,y=200)
    textBox = Text(root, height=1, width=20,bd=0)
    textBox.place(x=0,y=220)
    train_button = Button(root, text="Train",command = train_switch,width=4)
    train_button.place(x=0,y=245)

    reset_button = Button(root, text="Reset",command = reset_dict, fg='red',width=4)
    reset_button.place(x=0,y=275)
    reset_button = Button(root, text="Upload",command = upload_img,width=4)
    reset_button.place(x=100,y=245)

    #Frame render
    show_frame()
    root.mainloop()
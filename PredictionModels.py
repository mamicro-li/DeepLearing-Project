from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, InputLayer, Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.optimizers import Adam 
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from align import AlignDlib
from model import create_model
class FaceVerify:
    def __init__(self):
        '''
        nn4_small2_pretrained: Pretrained face image mapping model
        alignment: face align model
        '''
        self.nn4_small2_pretrained = create_model()
        self.nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')
        self.alignment = AlignDlib('landmarks/landmarks.dat')

    def align_image(self,img):
        #Preprocess image, align the face
        bounding_box = self.alignment.getLargestFaceBoundingBox(img)
        return bounding_box, self.alignment.align(96, img, bb=bounding_box, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

    def load_image(self,path):
        img = cv2.imread(path, 1)
        return img[...,::-1]

    def distance(self,emb1, emb2):
        #emb1 and emb2: (128-d numpy vector)
        return np.sum(np.square(emb1 - emb2))

    def get_id(self, face_vec, face_id_dict):
        '''
        face_vec: (128-d numpy vector), Target face vector
        face_id_dict: (dictionary), {'Name':[face_vec, img_num]}
        return: the closest person's name, 'Unknown' if distance > .5
        '''
        dist_list = []
        for id in face_id_dict:
            target_vec = face_id_dict[id][0]/face_id_dict[id][1]
            dist_list.append(self.distance(face_vec, target_vec))
        
        if(min(dist_list)>0.58): 
            return 'Unknown'
        else:
            closest_id = np.argmin(dist_list)
            return list(face_id_dict.keys())[closest_id]

############
class Models:
    def build_basic_model():
        """
        This is a Deep Convolutional Neural Network (DCNN). For generalization purpose I used dropouts in regular intervals.
        I used `ELU` as the activation because it avoids dying relu problem but also performed well as compared to LeakyRelu
        atleast in this case. `he_normal` kernel initializer is used as it suits ELU. BatchNormalization is also used for better
        results.
        """
        model = Sequential(name='Age_CNN')

        model.add(Conv2D(filters=64,kernel_size=(5,5),input_shape=(48, 48, 1),activation='elu',padding='same',kernel_initializer='he_normal',name='conv2d_1'))
        model.add(BatchNormalization(name='batchnorm_1'))
        model.add(Conv2D(filters=64,kernel_size=(5,5),activation='elu',padding='same',kernel_initializer='he_normal',name='conv2d_2'))
        model.add(BatchNormalization(name='batchnorm_2'))
        
        model.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_1'))
        model.add(Dropout(0.4, name='dropout_1'))

        model.add(Conv2D(filters=128,kernel_size=(3,3),activation='elu',padding='same',kernel_initializer='he_normal',name='conv2d_3'))
        model.add(BatchNormalization(name='batchnorm_3'))
        model.add(Conv2D(filters=128,kernel_size=(3,3),activation='elu',padding='same',kernel_initializer='he_normal',name='conv2d_4'))
        model.add(BatchNormalization(name='batchnorm_4'))
        
        model.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_2'))
        model.add(Dropout(0.4, name='dropout_2'))

        model.add(Conv2D(filters=256,kernel_size=(3,3),activation='elu',padding='same',kernel_initializer='he_normal',name='conv2d_5'))
        model.add(BatchNormalization(name='batchnorm_5'))
        
        model.add(Conv2D(filters=256,kernel_size=(3,3),activation='elu',padding='same',kernel_initializer='he_normal',name='conv2d_6'))
        model.add(BatchNormalization(name='batchnorm_6'))
        
        model.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_3'))
        model.add(Dropout(0.5, name='dropout_3'))

        model.add(Flatten(name='flatten'))    
        model.add(Dense(128,activation='elu',kernel_initializer='he_normal',name='dense_1'))
        model.add(BatchNormalization(name='batchnorm_7'))
        model.add(Dropout(0.6, name='dropout_4'))
        
        
        return model
    @staticmethod
    def build_complex_age_model():
        model = Models.build_basic_model()
        model.add(Dense(1,activation='relu',name='out_layer'))
        model.load_weights('weights/age_complex_model.h5')
        return model
    @staticmethod
    def build_complex_emotion_model():
        model = Models.build_basic_model()
        model.add(Dense(3,activation='softmax',name='out_layer'))
        model.load_weights('weights/emotion_complex_model.h5')

        return model
    @staticmethod
    def biuld_emotion_model():
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
            Dense(3, activation='softmax')
        ])
        emotion_model.load_weights('weights/emotion_model.h5')
        return emotion_model

    @staticmethod
    def build_simple_age_model():
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
        age_model.load_weights('weights/age_model.h5')
        return age_model
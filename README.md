# DeepLearing-Project

## What is in this repository?
- [models](https://github.com/lihongwei970/DeepLearing-Project/blob/main/models): Landmarks that used by face alignment.
- [weights](https://github.com/lihongwei970/DeepLearing-Project/blob/main/weights): Pretrained prediction model weights.
- [images](https://github.com/lihongwei970/DeepLearing-Project/blob/main/images): Images used to initialize face verification database.
- [training_notebooks/train_emtion.ipynb](https://github.com/lihongwei970/DeepLearing-Project/blob/main/training_notebooks/train_emtion.ipynb): Notebook for emotion model training.
- [training_notebooks/train_age.ipynb](https://github.com/lihongwei970/DeepLearing-Project/blob/main/training_notebooks/train_age.ipynb): Notebook for age model training.
- [kaggle.json](https://github.com/lihongwei970/DeepLearing-Project/blob/main/kaggle.json): Key to retrieve Kaggle dataset.
- [haarcascade_frontalface_default.xml](https://github.com/lihongwei970/DeepLearing-Project/blob/main/haarcascade_frontalface_default.xml): Pretrained frontface detector weights used by cv2.
- [dict_file](https://github.com/lihongwei970/DeepLearing-Project/blob/main/dict_file): Dictionary variable that stores the face vectors, use `pickle` to load the data.
- [model.py](https://github.com/lihongwei970/DeepLearing-Project/blob/main/model.py): Face verification model.
- [align.py](https://github.com/lihongwei970/DeepLearing-Project/blob/main/align.py): Functions that used to align face based on landmarks.
- [PredictionModels.py](https://github.com/lihongwei970/DeepLearing-Project/blob/main/PredictionModels.py): Prediction models generator.
- [run_UI.py](https://github.com/lihongwei970/DeepLearing-Project/blob/main/run_UI.py): Main function that runs the GUI.

## Requirements:
```
python3.9 (configured tkinter)
tensorflow==2.5.0-rc3
numpy
imutils
opencv-python
dlib
```

## How to run
Notice:`train_emtion.ipynb` and `train_age.ipynb` require to upload `kaggle.json` to obtain dataset.\
\
**Run main program:**
```
python3 -m pip install -r requirements.txt
python3 run_UI.py
```

## User instruction
- Display buttons: Corresponding prediction display switch.
- Name field: User's name for face verification training.
- Train: Use current camera feed as an image input to the face database.
- Upload: Manually upload face image to the face database.
- Reset: Reset the face verification database.

<img src="https://www.youtube.com/watch?v=0Pk1WW23KXk" data-canonical-src="https://github.com/lihongwei970/DeepLearing-Project/blob/main/Sample.png" width="200" height="400" />

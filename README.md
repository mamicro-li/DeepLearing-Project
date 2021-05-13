# DeepLearing-Project

## What is in this repository?
- [models](https://github.com/lihongwei970/DeepLearing-Project/blob/main/models): Landmarks that used by face alignment.
- [weights](https://github.com/lihongwei970/DeepLearing-Project/blob/main/weights): Pretrained prediction model weights.
- [images](https://github.com/lihongwei970/DeepLearing-Project/blob/main/images): Images used to initialize face verification database.
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
```
python3 -m pip install -r requirements.txt
python3 run_UI.py
```
![alt text]https://github.com/lihongwei970/DeepLearing-Project/blob/main/Sample.png?raw=True)

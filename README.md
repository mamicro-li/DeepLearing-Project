# DeepLearing-Project

## What is in this repository?
- [models](https://github.com/lihongwei970/DeepLearing-Project/models): Landmarks that used by face alignment.
- [weights](https://github.com/lihongwei970/DeepLearing-Project/weights): Pretrained prediction model weights.
- [haarcascade_frontalface_default](https://github.com/lihongwei970/DeepLearing-Project/haarcascade_frontalface_default.xml): Pretrained frontface detector weights used by cv2.
- [images](https://github.com/lihongwei970/DeepLearing-Project/images): Images used to initialize face verification database.
- [PredictionModels.py](https://github.com/lihongwei970/DeepLearing-Project/PredictionModels.py): Prediction models generator.
- [run_UI.py](https://github.com/lihongwei970/DeepLearing-Project/run_UI.py): main function that runs the GUI.

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


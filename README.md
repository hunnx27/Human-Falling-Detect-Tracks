<h1> Human Falling Detection and Tracking </h1>

Using Tiny-YOLO oneclass to detect each person in the frame and use 
[AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) to get skeleton-pose and then use
[ST-GCN](https://github.com/yysijie/st-gcn) model to predict action from every 30 frames 
of each person tracks.

Which now support 7 actions: Standing, Walking, Sitting, Lying Down, Stand up, Sit down, Fall Down.

<div align="center">
    <img src="sample1.gif" width="416">
</div>

## Prerequisites

- Python > 3.6
- Pytorch > 1.3.1

Original test run on: i7-8750H CPU @ 2.20GHz x12, GeForce RTX 2070 8GB, CUDA 10.2

## Data

This project has trained a new Tiny-YOLO oneclass model to detect only person objects and to reducing 
model size. Train with rotation augmented [COCO](http://cocodataset.org/#home) person keypoints dataset 
for more robust person detection in a variant of angle pose.

For actions recognition used data from [Le2i](http://le2i.cnrs.fr/Fall-detection-Dataset?lang=fr)
Fall detection Dataset (Coffee room, Home) extract skeleton-pose by AlphaPose and labeled each action 
frames by hand for training ST-GCN model.

## Pre-Trained Models

- Tiny-YOLO oneclass - [.pth](https://drive.google.com/file/d/1obEbWBSm9bXeg10FriJ7R2cGLRsg-AfP/view?usp=sharing),
[.cfg](https://drive.google.com/file/d/19sPzBZjAjuJQ3emRteHybm2SG25w9Wn5/view?usp=sharing)
- SPPE FastPose (AlphaPose) - [resnet101](https://drive.google.com/file/d/1N2MgE1Esq6CKYA6FyZVKpPwHRyOCrzA0/view?usp=sharing),
[resnet50](https://drive.google.com/file/d/1IPfCDRwCmQDnQy94nT1V-_NVtTEi4VmU/view?usp=sharing)
- ST-GCN action recognition - [tsstg](https://drive.google.com/file/d/1mQQ4JHe58ylKbBqTjuKzpwN2nwKOWJ9u/view?usp=sharing)

## Basic Use

1. Download all pre-trained models into ./Models folder.
2. Run main.py
```
    python main.py ${video file or camera source}
```

## Reference

- AlphaPose : https://github.com/Amanbhandula/AlphaPose
- ST-GCN : https://github.com/yysijie/st-gcn



-----------------------------
# pip env shell
Python 3.13.2 로 실행했음..

# CUDA
nvidia-smi
pip3 install torch torchvision torchaudio —index-url https://download.pytorch.org/whl/cu126
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# pip3 freeze > requirements.txt
pip3 install -r requirements.txt

colorama==0.4.6
contourpy==1.3.2
cycler==0.12.1
filelock==3.18.0
fonttools==4.57.0
fsspec==2025.3.2
Jinja2==3.1.6
kiwisolver==1.4.8
MarkupSafe==3.0.2
matplotlib==3.10.1
mpmath==1.3.0
networkx==3.4.2
numpy==2.2.4
opencv-python==4.11.0.86
packaging==24.2
pillow==11.2.1
pyparsing==3.2.3
python-dateutil==2.9.0.post0
pytube==15.0.0
pytubefix==8.12.3
scipy==1.15.2
setuptools==78.1.0
six==1.17.0
sympy==1.13.3
torch==2.8.0.dev20250416+cu128
torchaudio==2.6.0.dev20250417+cu128
torchvision==0.22.0.dev20250417+cu128
tqdm==4.67.1
typing_extensions==4.13.2
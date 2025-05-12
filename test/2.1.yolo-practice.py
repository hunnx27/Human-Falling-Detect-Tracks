import torch


# 모델
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
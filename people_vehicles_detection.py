import torch
import numpy as np
import cv2


TH = 0.4


class YoloV5Detect:

    def __init__(self):
        """Initialize teh class loading a Yolov5 Small.
        """
        self._model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    def predict(self, img, draw=True):
        """Return a list of [x1,y1,x2,y2] of all vehicles and a list of [x1,y1,x2,y2] of all people.
        """
        imgs = [img]
        results = self._model(imgs)
        df = results.pandas().xyxy[0] 
        df = df[df['confidence'] >= TH]

        df_vehicle = df[(df['name'] == 'car') | (df['name'] == 'bicycle') | (df['name'] == 'motorcycle') | (df['name'] == 'truck') | (df['name'] == 'bus')]
        df_person = df[(df['name'] == 'person')]

        vehicles_bb = []
        for row in df_vehicle.itertuples():
            vehicles_bb.append([int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax)])
            if draw: cv2.rectangle(img, (int(row.xmin), int(row.ymin)), (int(row.xmax), int(row.ymax)), (0, 255, 0), 2)

        pedestrian_bb = []
        for row in df_person.itertuples():
            pedestrian_bb.append([int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax)])
            if draw: cv2.rectangle(img, (int(row.xmin), int(row.ymin)), (int(row.xmax), int(row.ymax)), (255, 0, 0), 2)

        return vehicles_bb, pedestrian_bb

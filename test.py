import os
import cv2
import torch
import numpy as np
from torch.autograd import Variable

from model import Darknet


def get_test_img():
    img = cv2.imread('dog-cycle-car.png')
    img = cv2.resize(img, (416, 416))
    img = img[:,:,::-1].transpose((2, 0, 1))
    img = img[np.newaxis,:,:,:]/255.
    img = torch.from_numpy(img).float()
    img = Variable(img)
    return img

model = Darknet('config/yolov3.cfg')
img = get_test_img()
out = model(img)
print(out)
print(f'Shape:=> First YOLO:{out[0].shape} Sencond YOLO:{out[0].shape}')

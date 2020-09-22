import os
import torch
import numpy as np
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, train=True, transform=None):
        self.train = train
        self.transform = transform
        if train:
            self.img_paths = './data/images/train/'
            self.label_paths = './data/labels/train/'
        else:
            self.img_paths = './data/images/val/'
            self.label_paths = './data/labels/val/'
    
    def __getitem__(self, index):
        img = Image.open(self.img_paths + str(index) + '.jpeg')
        img = self.transform(img)
        labels = torch.from_numpy(np.loadtxt(self.label_paths + 
                                            str(index) + 
                                            '.txt').reshape(-1, 5))
        targets = torch.zeros(len(labels), 6)
        targets[:,1:] = labels
        return img, targets
    
    def __len__(self):
        return len(os.listdir(self.img_paths))

#!usr/bin/env python3

import os
import sys
sys.path.insert(1, '../mdai/')

import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
from torch.autograd import Variable
from torchsummary import summary
import segmentation_models_pytorch as smp

from PIL import Image
import cv2
import albumentations as alb # A

import time
import os
from tqdm.notebook import tqdm

import warnings
warnings.filterwarnings('ignore')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OpticDiscDrusenDataset(Dataset):
    ''' Makes the exported masks, labels and images into a torch Dataset
    for now, the transforms are kept very simple. '''

    def __init__(self, img_path, mask_path, x_set, transform=None, target_transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.x_set = x_set
        self.transform = transform
        self.target_transform = target_transform
       
    def __len__(self):
        ''' x_set = x_train/test/val '''
        return len(self.x_set)
    
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()  # turns requested index numbers to a list

        img = cv2.imread(os.path.join(self.img_path, self.x_set[idx]) + '.png')
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRGB)  # convert to greyscale
        mask = np.load(os.path.join(self.mask_path, self.x_set[idx]) + '.npy')

        if self.transform:  # for images
            img = self.transform(img)
            img = nn.functional.pad(input=img, pad=(0,0,8,8), mode='constant') # pad so that size is 512 x 768
        
        if self.target_transform:   # for masks (labels?)
            mask = self.target_transform(mask)
            mask = nn.functional.pad(input=mask, pad=(0,0,8,8), mode='constant') # pad so that size is 512 x 768
    
        return img, mask
    


#!usr/bin/env python3

import os
import sys
sys.path.insert(1, '../mdai/')

import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

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
        mask = np.load(os.path.join(self.mask_path, self.x_set[idx]) + '.npy')

        if self.transform:  # for images
            img = self.transform(img)
            #img = nn.functional.pad(input=img, pad=(0,0,8,8), mode='constant') # pad so that size is 512 x 768
        
        if self.target_transform:   # for masks (labels?)
            mask = self.target_transform(mask)
            #mask = nn.functional.pad(input=mask, pad=(0,0,8,8), mode='constant') # pad so that size is 512 x 768
    
        return img, mask
        

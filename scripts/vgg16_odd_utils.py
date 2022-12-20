#!usr/bin/env python3

import os
import sys
sys.path.insert(1, '../mdai/')

import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_test_val_split(id_values):
    ''' splits id values into 60:20:20 for train:test:val sets '''

    train_val_id, test_id = train_test_split(id_values, test_size=0.2, random_state=42)
    train_id, val_id = train_test_split(train_val_id, test_size=0.25, random_state=42)

    unique_ids = len(set(id_values))

    print('train_id  : %.2f, %d images' % ((len(train_id)/unique_ids), len(train_id)))
    print('test_id   : %.2f, %d images' % ((len(test_id)/unique_ids), len(test_id)))
    print('val_id    : %.2f, %d images' % ((len(val_id)/unique_ids), len(val_id)))

    return train_id, test_id, val_id


class OpticDiscDataset(Dataset):
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
        
        if self.target_transform:   # for masks (labels?)
            mask = self.target_transform(mask)
    
        return img, mask
        

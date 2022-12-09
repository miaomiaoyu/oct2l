#!usr/bin/env python3

# -- import modules

import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F
from torch.autograd import Variable

from PIL import Image
import cv2
import albumentations as A

import time
import os
from tqdm.notebook import tqdm

from torchsummary import summary
import segmentation_models_pytorch as smp

from workspace import paths_get, paths_join

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

id_map = pd.read_csv('../mdai/mdai_odd_id_map.csv')
id_map = id_map[['id', 'Slice_ODD_id', 'labelName']]

print('Total images: ', id_map.id.nunique())

for label_name in id_map['labelName'].unique():
    print('Number of %s: %d' % (
        label_name, id_map[id_map['labelName']==label_name]['id'].nunique()))

base_dir, data_dir = paths_get('oct2l')

IMAGE_PATH = paths_join(
    data_dir,['project-files-png', 'images_with_masks_mdai'])
MASK_PATH  = paths_join(
    base_dir,['mdai', 'mdai_labelled_data_img'])  # yes i agree the naming is atrocious (--> masks_mdai)

n_classes = 3

x_trainval, x_test = train_test_split(id_map['id'].values, test_size=0.2, random_state=42)
x_train, x_val = train_test_split(x_trainval, test_size=0.25, random_state=42)

print('x_train: %.1f' % (len(x_train)/id_map.id.nunique()))
print('x_test: %.1f' % (len(x_test)/id_map.id.nunique()))
print('x_val: %.1f' % (len(x_val)/id_map.id.nunique()))

print('IMAGE_PATH: %s' % IMAGE_PATH)
print('MASK_PATH: %s' % MASK_PATH)


class ODDDataset(Dataset):
    ''' Makes the exported meta data and images into a torch Dataset
    note: this is very similar to <LabelledOCTImagesMDAI>, but better? '''
    
    def __init__(self, img_path, mask_path, x_set, transform=None, target_transform=None, patch=False):
        self.img_path = img_path
        self.mask_path = mask_path
        self.x_set = x_set
        self.transform = transform
        self.target_transform = target_transform
        self.patch = patch
       
    def __len__(self):
        ''' x_set = train/test/val '''
        return len(self.x_set)
    
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()  # turns requested index numbers to a list

        img = cv2.imread(os.path.join(self.img_path, self.x_set[idx]) + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = np.load(os.path.join(self.mask_path, self.x_set[idx]) + '.npy')

        if self.transform:
            img = self.transform(img)
            img = F.pad(input=img, pad=(0,0,8,8), mode='constant') 
            # pad so that size is 512 x 768, final size has to be divisable by 32.
        
        if self.target_transform:
            mask = self.target_transform(mask)
            mask = F.pad(input=mask, pad=(0,0,8,8), mode='constant')

        if self.patch:
            img, mask = self.tiles(img, mask)
    
        return img, mask
    
    def tiles(self, img, mask):
        img_patches =  img.unfold(1, 256, 256).unfold(2, 384, 384)
        img_patches  = img_patches.contiguous().view(3,-1, 256, 384) 
        img_patches = img_patches.permute(1,0,2,3)
        mask_patches = mask.unfold(0, 256, 256).unfold(1, 384, 384)
        mask_patches = mask_patches.contiguous().view(-1, 256, 384)

        return img_patches, mask_patches    

'''mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

t_train = A.Compose([A.Resize(496, 768, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(), A.VerticalFlip(), 
                     A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0,0.5),(0,0.5)),
                     A.GaussNoise()])
t_val = A.Compose([A.Resize(496, 768, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(),
                   A.GridDistortion(p=0.2)])'''

train_set = ODDDataset(IMAGE_PATH, MASK_PATH, x_train, transform=T.ToTensor(),target_transform=T.ToTensor())#patch=True)
val_set = ODDDataset(IMAGE_PATH, MASK_PATH, x_val, transform=T.ToTensor(), target_transform=T.ToTensor())#, patch=True)

batch_size = 64

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]

# Load the pretrained model from pytorch
vgg16 = models.vgg16_bn()
vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
print(vgg16.classifier[6].out_features) # 1000 

# Freeze training for all layers
for param in vgg16.features.parameters():
    param.require_grad = False

# Newly created modules have require_grad=True by default
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, len(class_names))]) # Add our layer with 4 outputs
vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
print(vgg16)
#!usr/bin/env python3

import os
import time
import cv2
import csv
import pandas as pd
import numpy as np
from pathlib import PurePath
import torch
from torch.utils.data import Dataset

import warnings
warnings.filterwarnings('ignore')




def write_to_csv(data, header, filename):
    ''' Writes the meta-data to a csv_file '''
    with open('%s.csv' % filename, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(data)
    
    return csv_file


def mdai_labelled_data_get(annots, root_dir):
    ''' Retrieves the labelled OCT data from MD.ai platform
    Annotations/meta data will be stored as a csv_file and labelled images will be stored as .npy in the specified root_dir. 
    '''
    
    if ~os.path.exists(os.path.join(os.getcwd(), root_dir)):
        try:
            os.mkdir(os.path.join(os.getcwd(), root_dir))
        except FileExistsError: # not sure why this is raised tbh.
            print("The `root_dir` you're trying to access already exists.")

    tic = time.time()   # keep time
    bad_files = 0

    height, width = 496, 768  # hard-coded for now, it's always the same.
    meta_data = []  # annotations/meta data will be saved to a csv_file

    for i in range(annots.shape[0]):
        
        idx           = annots.loc[i,'id']
        data          = annots.loc[i,'data']
        meta          = annots.loc[i,[
            'id', 'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'labelName', 'frameNumber']].tolist()

        try:
            for _, vertices in data.items():    # _ = 'vertices'
                contours = np.array([[int(vertice[0]), int(vertice[1])] for vertice in vertices])
                bw = cv2.fillPoly(np.zeros((height, width)), pts=[contours], color=1.0)   # binarize image

            img_name = PurePath(os.path.join(os.getcwd(), root_dir), idx)

            np.save(img_name, arr=bw)
            meta_data.append(meta)

        except AttributeError:
            print("%s did not have any vertices data." % idx)
            bad_files += 1

    toc = time.time()-tic
    print("Time elapsed: %.1f seconds \nBad files found: %d" % (toc, bad_files))

    csv_file = write_to_csv(data=meta_data, header=['id', 'StudyInstanceUID','SeriesInstanceUID','SOPInstanceUID','labelName','frameNumber'], filename='mdai_labelled_data_meta')
    
    return csv_file


class LabelledOCTImagesMDAI(Dataset):
    ''' Makes the exported meta data and images into a torch Dataset '''
    
    def __init__(self, csv_file, root_dir, transform=None):
        '''
        Args: 
            csv_file: csv file with annotations/meta data
            root_dir: directory with the binarized images
        '''
        self.meta_data = pd.read_csv(csv_file)
        self.root_dir  = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.meta_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()  # turns requested index numbers to a list

        file_id     = self.meta_data.loc[idx, 'id']
        study_uid   = self.meta_data.loc[idx, 'StudyInstanceUID']
        series_uid  = self.meta_data.loc[idx, 'SeriesInstanceUID']
        sop_uid     = self.meta_data.loc[idx, 'SOPInstanceUID']
        label       = self.meta_data.loc[idx, 'labelName']
        slice_n     = self.meta_data.loc[idx, 'frameNumber']

        # to retrieve the corresponding file
        image_id    = '/' + file_id + '.npy' 
        fname       = os.path.join(os.getcwd(), self.root_dir) + image_id
        image       = np.load(fname)
        image       = np.array([image], dtype=float)
        image       = np.squeeze(image)

        sample = {
            'id': file_id,
            'study_uid': study_uid,
            'series_uid': series_uid,
            'sop_uid': sop_uid,
            'slice_n': slice_n,
            'label': label,
            'image': image
        }

        if self.transform:
            sample = self.transform(sample)
        
        return sample





        
#!usr/bin/env python3
'''
Saves .npy files to .mat format
'''

from scipy.io import savemat
import numpy as np
import glob
import os

def npy_to_mat(npy_dir, mat_dir):
    '''
    Saves .npy files in npy_dir to .mat format in mat_dir
    '''
    
    for npyfile in glob.glob(os.path.join(npy_dir,'*.npy')):
        matfile = os.path.splitext(os.path.basename(npyfile))[0] + '.mat'
        data = np.load(npyfile)
        savemat(os.path.join(mat_dir, matfile), {'d':data})
        print('saved %s from %s' % (matfile, npyfile))

npy_to_mat('../data/project-files-2022-11-30-npy', '../data/project-files-2022-11-30-mat')
#!usr/bin/env python3
#
# MY, 26 Nov 2022
# mmy@stanford.edu

'''
OCT2L prototype
'''

#%%

import os
import glob
import argparse
import warnings
from tqdm import tqdm
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from oct_converter.readers import E2E
import scipy
from scipy.interpolate import splrep, splev
from scipy.signal import convolve2d, medfilt, savgol_filter
from sklearn.preprocessing import MinMaxScaler


def is_oct(file, ext=('.npy', '.E2E')):
    '''Check file is either npy or E2E'''
    return file.endswith(ext)

def parse_args():
    parser = argparse.ArgumentParser(description='2-layer SD-OCT segmentation')
    parser.add_argument('--folder', help='folder of the dataset')
    args = parser.parse_args()
    #assert (is_oct(args.filename)), \
        #'File provided must be either .npy or .E2E.'
    return args


class OCT2L:

    """ Segments internal limiting membrane (ILM) in optical coherence tomography (OCT) volume and returns the surface as 2-D matrix. Also returns a cropped volume beneath that surface for RPE segmentation """

    def __init__(self):
        print('oct2l init')
    
    def e2e_to_npy(self, filename):
        ''' converts E2E to npy if it's not been done yet '''
        e2e = E2E(filename)
        volume = e2e.read_oct_volume()
        return volume

    def loading(self, filename):
        ''' loads the volume in based on file extension '''
        ext = os.path.splitext(os.path.basename(filename))[-1]
        #filepath = "../data//Users/miaomiaoyu/Documents/LiaoLab/oct2l/data/project-files-2022-11-30-npy/%s" % filename
        filepath = filename
        match ext:
            case '.E2E':
                volume = self.e2e_to_npy(filepath)
            case '.npy':
                volume = np.load(filepath)
        return volume

    def saving(self, objects, filename):
        ''' loads the volume in based on file extension'''
        ext = os.path.splitext(filename)[-1]
        match ext:
            case '.npy':
                with open('tmp/{}'.format(filename), 'wb') as f:
                    for obj in objects:
                        np.save(f, obj)
            case '.mat':
                obj = {os.path.splitext(filename)[0]: objects}
                scipy.io.savemat('tmp/{}'.format(filename), obj)
            
        print('%s saved in /tmp' % filename)

    def preprocessing(self, volume):
        ''' applies a filter on images to increase intensity '''
        K,N,M = volume.shape
        c_volume = np.zeros((K, N, M))
        kernel = np.array([[1, 1, 1, -1, -1, -1]]).T
        kernel = kernel/kernel.shape[0]
        scaler = MinMaxScaler(feature_range=(volume.min(), volume.max()))
        for k in range(K):
            image = volume[k,:,:]  # image.max() == 255
            e_image = scaler.fit_transform(image**2)  # intensity squared. 
            c_image = convolve2d(e_image, kernel, mode="same")
            c_volume[k,:,:] = c_image
        return c_volume
    
    def segmenting(self, volume):
        ''' performs segmentation on the data
        requires additional methods for outlier detection:
            detect outliers
            select outliers
            replace dead pixels
        '''
        K, _, M = volume.shape
        surface = np.zeros((K, M))
        for k in range(K):
            image = volume[k,:,:]
            points = self.points_get(image, M)
            points = self.points_smooth(points)
            surface[k,:] = points
        return surface

    def points_get(self, image, M):
        '''Finds the outliers within each image'''
        points = []
        for m in range(M):
            a_scan = image[:,m]  # 'a' scan
            baseline = a_scan[: int(M * 0.05)]  # use 5% left edge to calculate baseline
            # compare it to two values: deviation from mean and actual value.
            min_dfm = (5 * np.std(baseline)) + np.mean(baseline)
            min_int = 5  # actual value
            outliers = [
                i for i,j in enumerate(a_scan) if (abs(j) > min_int) & (
                    abs(j) > (min_dfm)) ]
            outlier = min(outliers) if len(outliers) > 0 else -1
            points.append(outlier)
        return points

    def points_smooth(self, points):
        ''' applies a median filter followed by a savistsky golay filter '''
        points = medfilt(points, 15)
        points = savgol_filter(points, 31, 3)
        points = np.array([int(p) for p in points])
        return points

    def correcting(self, surface, stride=10):
        '''
        1) find, remove and interpolate big spikes(ie. derivations)
        2) 
        '''
        spikes = self.spikes_get(surface, stride=stride)
        spikeless_surface = self.spikes_remove(surface, spikes)
        intp_surface = self.surface_interpolate(spikeless_surface)
        return intp_surface

    def spikes_get(self, surface, stride=16):
        ''' tries to correct via finding big changes in derivatives, quant set at .9-.95 ''' 
        K, M = surface.shape
        spikes = np.zeros((K, M))
        quant = [.9 if (i > (K*.15)) and (i < (K-(K*.15)) ) else .95 for i in range(K)]
        samples = np.array(surface)[:,np.arange(0, M, stride, dtype=int)]
        for k in range(K):
            deriv_stride = np.diff(samples[k,:], append=samples[k,-1])
            deriv = np.repeat(deriv_stride, stride)  # respaces to original shape
            q = np.quantile(abs(deriv), [quant[k]])
            spike_index = np.array([i for i,j in enumerate(deriv) if abs(j) > q], dtype=int)
            spikes[k, spike_index]=+1   # 1 = spike, 0 = no spike
        return spikes

    def spikes_remove(self, surface, spikes):
        ''' replace spikes with np.nan: quick way to test if spikes are gone '''
        return np.where(spikes==0, surface, np.nan)
        
    def surface_interpolate(self, surface):
        ''' interpolate , polynomial factor set to 1 for edges and 2 for middle areas '''
        K, M = surface.shape
        intp_surface = []
        exps = [2 if (i > (K*.15)) and (i < (K-(K*.15)) ) else 1 for i in range(K)]
        indices = np.tile(np.arange(M),(K,1))
        for k in range(K):
            cs = surface[k,:][~np.isnan(surface[k,:])]
            idx = indices[k,:][~np.isnan(surface[k,:])]
            x, y, k = idx, cs, exps[k]
            tck = splrep(x, y, k=exps[k])
            x_out = np.arange(M)
            y_out = splev(x_out, tck, der=0)
            intp_surface.append(y_out)
        return np.array(intp_surface)

    def cropping(self, volume, surface, height=100):
        ''' crops volume to attain make rpe segmentation simpler '''
        K, _, M = volume.shape
        cr_volume = np.zeros((K, height, M))
        for k in range(K):
            for m in range(M):
                i0 = int(surface[k,m]) + 30  # tailored to OCT
                a_scan = volume[k, i0:i0+height, m]
                if len(a_scan) < height:
                    a_scan = np.append(a_scan, [-1]*(height-len(a_scan)))
                elif len(a_scan) > height:
                    a_scan = a_scan[:height]  # sometimes it comes up short
                cr_volume[k,:,m] = a_scan
        return cr_volume

    def visualizing(self, volume, surface, stride=None):
        ''' visualizes volume and surface '''
        cmap = 'viridis'
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = 'Georgia'
        matplotlib.rcParams['font.size'] = 12
        ncols = 5
        K,_,_ = volume.shape
        stride = int(K/2) if stride is None else stride # just first, last and middle
        slices = np.arange(0,K,stride,dtype=int)
        multiple_rows = False if len(slices) <= ncols else True
        match multiple_rows:
            case True:
                nrows = int(np.ceil(len(slices)/ncols))
                fig,ax = plt.subplots(nrows, ncols, figsize=(15,3*nrows))
                for i,k in enumerate(slices):
                    r,c = i//ncols, i%ncols
                    ax[r,c].imshow(volume[k,:,:], cmap=cmap)
                    ax[r,c].plot(surface[k,:], c='r')
                    ax[r,c].set_title('Slice %s' % str(k+1))
                plt.tight_layout()
            case False:
                fig,ax = plt.subplots(1, ncols, figsize=(15,3))
                for i,k in enumerate(slices):
                    ax[i].imshow(volume[k,:,:], cmap=cmap)
                    ax[i].plot(surface[k,:], c='r')
                    ax[i].set_title('Slice %s' % str(k+1))
                plt.tight_layout()
        plt.show()
        return fig

def main(args):
    '''
    # -- command line inputs
    #! python3 main.py --folder project-files-2022-11-30-npy --for ilm
    '''
    
    folder = args.folder
    #layers = args.layers

    model = OCT2L()
    tic = time.time()
    projectdir='oct2l'
    #fullpath = os.path.join('/Users/miaomiaoyu/code', projectdir, 'data', folder)

    for filename in tqdm(glob.glob('/Users/miaomiaoyu/Documents/LiaoLab/oct2l/data/project-files-2022-11-30-npy/*.npy')):

        volume = model.loading(filename)  # make this a folder loop
        c_volume = model.preprocessing(volume)  # convolved volume
        surface = model.segmenting(c_volume)  # get ilm surface
        surface_ilm = model.correcting(surface)  # smoothing required
        #_ = model.saving(surface_ilm, 'surface_ilm.mat')
        cr_volume = model.cropping(volume, surface_ilm, height=150)
        #_ = model.saving(cr_volume, 'cr_volume.mat')

    toc = time.time()
    print("time elapsed: %.1f seconds" % (toc-tic))

if __name__ == "__main__":
    args = parse_args()
    main(args)

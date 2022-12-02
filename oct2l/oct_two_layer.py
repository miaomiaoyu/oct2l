#!usr/bin/env python3
# MY, 01 Dec 2022
# mmy@stanford.edu

import os
import time
import glob
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import splrep, splev
from scipy.signal import convolve2d, medfilt, savgol_filter
from sklearn.preprocessing import MinMaxScaler
from workspace import get_paths, set_paths

def parse_args():
    parser = argparse.ArgumentParser(description='2-layer SD-OCT segmentation')
    parser.add_argument('--folder', help='name of data subfolder containing OCT volumes')
    args = parser.parse_args()
    return args

def load_volume(filename):
    return np.load(filename)

def save_mat(filename, obj, PATH):
    obj = {'f':obj}
    scipy.io.savemat(os.path.join(PATH, filename), obj)
    
def save_npy(filename, obj, PATH):
    with open(os.path.join(PATH, filename), 'wb') as f:
        for j in obj:
                np.save(f, j)

def preprocessing(volume):
    ''' applies a filter on images to increase intensity '''
    K,N,M = volume.shape
    c_volume = np.zeros((K,N,M))
    kernel = np.array([[1, 1, 1, -1, -1, -1]]).T
    kernel = kernel/kernel.shape[0]
    scaler = MinMaxScaler(feature_range=(volume.min(), volume.max()))
    for k in range(K):
        image = volume[k,:,:]  # image.max() == 255
        e_image = scaler.fit_transform(image**2)  # intensity squared. 
        c_image = convolve2d(e_image, kernel, mode="same")
        c_volume[k,:,:] = c_image
    return c_volume

def locate_points(image):
    '''Finds the outliers within each image'''
    _,M = image.shape
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

def filter_points(points):
    ''' applies a median filter followed by a savistsky golay filter '''
    points = medfilt(points, 15)
    points = savgol_filter(points, 31, 3)
    points = np.array([int(p) for p in points])
    return points

def segmenting(volume):
    K, _, M = volume.shape
    surface = np.zeros((K, M))
    for k in range(K):
        image = volume[k,:,:]
        points = locate_points(image)
        points = filter_points(points)
        surface[k,:] = points
    return surface

def locate_spikes(surface, stride=10):
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

def filter_spikes(surface, spikes):
    ''' replace spikes with np.nan: quick way to test if spikes are gone '''
    return np.where(spikes==0, surface, np.nan)

def interpolate_surface(surface):
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

def correcting(surface, stride=10):
    '''
    1) find, remove and interpolate big spikes(ie. derivations)
    2) 
    '''
    spikes = locate_spikes(surface, stride=stride)
    spikeless_surface = filter_spikes(surface, spikes)
    intp_surface = interpolate_surface(spikeless_surface)
    return intp_surface

def crop_volume(volume, surface, height=100):
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

def visualizing(volume, surface, stride=None):
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
    folder = args.folder
    _, _, DATA_PATH, OUTPUT_PATH, _ = get_paths()

    data_dir = set_paths(DATA_PATH, folder)
    ilm_dir = set_paths(OUTPUT_PATH, 'ilm')
    cvol_dir = set_paths(OUTPUT_PATH, 'cropvol')
    vol_dir = set_paths(OUTPUT_PATH, 'vol')

    tic = time.time()
    for filename in tqdm(glob.glob(data_dir+'/*.npy')):
        try:
            fname,_ = os.path.splitext(os.path.basename(filename))
            volume = load_volume(filename)
            save_mat(fname+'_volume.mat', volume, vol_dir)
            preprocessed_volume = preprocessing(volume)
            surface = segmenting(preprocessed_volume)
            ilm_surface = correcting(surface)
            fname = fname.replace(' ', '_')
            save_mat(fname+'_ilm.mat', ilm_surface, ilm_dir)
            cropped_volume = crop_volume(volume, ilm_surface, height=150)
            save_mat(fname+'_crop.mat', cropped_volume, cvol_dir)
            print("%s completed" % fname)
        except ValueError:
            print("%s produced ValueError", fname)
    toc = time.time()
    print("time elapsed: %.1f seconds" % (toc-tic))

    # -- matlab engine to run cannyDetector 
    # 02 Dec 2022: this doesn't work on M1 chip.
    #eng = matlab.engine.start_matlab(SCRIPT_PATH)  # cwd = where the scripts live
    #eng.cannyRun(CVOL_DIR, nargout=0)  # run
    # --
    
if __name__ == "__main__":
    args = parse_args()
    main(args)



#!/usr/bin/env python3
#!conda env create -n cvision --file environment.yml

import os
import numpy as np
from scipy.signal import convolve2d, medfilt, savgol_filter
from sklearn.preprocessing import MinMaxScaler




class OCT2Layer:



    def __init__(self):
        
        kernel = np.array([[1, 1, 1, -1, -1, -1]]).T
        kernel = kernel/kernel.shape[0]

        
        self.parameters = {

            "kernel": kernel,

            "conv_k": 30,  # real-world micron spacing between each point
            "conv_n": 3.8,
            "conv_m": 6

        }
    

    def Load(self, filepath):

        """
        Loads oct volume from filepath provided

        """

        vol = np.load(filepath)

        return vol
        
        
    def get_parameters(self):

        """
        Returns the parameters if method requires.

        """

        #for key, val in self.parameters.items():
            #exec(key + "=val")

        for k in self.parameters.keys():
            self.__setattr__(k, self.parameters[k])



    def Preprocess(self, vol):

        """
        Applies a filter on images stored within the volume to increase intensity

        
        """

        self.get_parameters()  # get parameters required for later.

        K, N, M = vol.shape

        cvol = np.zeros((K, N, M))
        
        #vol = np.uint8(vol)  # vol.max()==255
        scaler = MinMaxScaler(feature_range=(vol.min(), vol.max()))

        for k in range(K):

            img = vol[k,:,:]  # img.max() == 255

            eimg = scaler.fit_transform(img**2)  # intensity squared. 
            cimg = convolve2d(eimg, self.kernel.T, mode="same")
            
            cvol[k,:,:] = cimg
        
        return cvol


    
    def Segment_ilm(self, vol):

        """
        Performs segmentation on the data
        Requires additional methods for outlier detection:
            detect outliers
            select outliers
            replace dead pixels
        
        """

        K, _, M = vol.shape

        surface = np.zeros((K, M))

        for k in range(K):

            img = vol[k,:,:]
            points = self.get_outliers(img, M)
            points = self.get_smoothedpoints(points)
        
            surface[k,:] = points
        
        return surface

    
    def Segment_rpe(self, vol, surfaceref):

        """
        Segment RPE layers
        
        """
        
        volsubset = self.get_volsubset(vol, surfaceref)

        K, _, M = volsubset.shape

        surface = np.zeros((K, M))

        for k in range(K):

            img = vol[k,:,:]
            points = self.get_outliers(img, M)
            points = self.get_smoothedpoints(points)
        
            surface[k,:] = points
        
        return surface



    def get_outliers(self, img, M):

        """
        Finds the outliers within each image

        """

        points = []

        for m in range(M):

            a_scan = img[:,m]  # 'a' scan
            baseline = a_scan[: int(M * 0.05)]  # use 5% left edge to calculate baseline

            # compare it to two values: deviation from mean and actual value.
            min_dfm = (5 * np.std(baseline)) + np.mean(baseline)
            min_int = 5  # actual value
            
            outliers = [
                i for i,j in enumerate(a_scan) if (abs(j) > min_int) & (
                    abs(j) > (min_dfm))
            ]

            outlier = min(outliers) if len(outliers) > 0 else -1

            points.append(outlier)

        return points



    def get_smoothedpoints(self, points):

        """
        Apply a median filter
        Apply a smoothing filter
        
        """

        points = medfilt(points, 15)
        points = savgol_filter(points, 31, 3)
        points = np.array([int(p) for p in points])

        return points


    def get_volsubset(self, vol, surface, Nsubset=100):

        """
        Run this for vol
        Nsubset is the number of rows the subset should have.

        """

        K, _, M = vol.shape

        volsubset = np.zeros((K, Nsubset, M))

        for k in range(K):
            for m in range(M):
                i0 = int(surface[k,m]) + 50  # makes sense for OCT scans to add this.
                subset = vol[k, i0:i0+Nsubset, m]

                if len(subset) < Nsubset:
                    subset = np.append(subset, subset[-1])
                elif len(subset) > Nsubset:
                    subset = subset[:100]
                
                volsubset[k,:,m] = subset
                
        return volsubset






    



    

    


        




def convert_e2e_to_npy(file):

    """
    Requires `oct_converter` which can be downloaded at https://github.com/marksgraham/OCT-Converter

    """

    e2e = E2E(file)
    oct_volumes = (e2e.read_oct_volume())

    filename = os.path.splitext(os.path.basename(file))[0]

    for volume in oct_volumes:
        npy_volume = volume.save("{}/npy/{}.npy".format(
            os.path.dirname(file), filename), dtype=object)

    print("{}.npy saved".format(filename))

    return npy_volume



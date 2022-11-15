#!/usr/bin/env python
#
# @mmy

"""

Performs two layer segmentation (ILM and RPE) on OCT volume
v 1.3.1

"""


import os
import math
import numpy as np

import matplotlib.pyplot as plt

from scipy import interpolate
from scipy.signal import savgol_filter, convolve2d
from scipy.ndimage import percentile_filter
from sklearn.preprocessing import MinMaxScaler









class OCT2Layer:

    def __init__(self, step):

        self.step = step  # step=1 for running every slice.

        # old kwargs, bypass fw config error
        self.kernel = [[1],[1],[1],[-1],[-1],[-1]]  
        self.kernel_dim = 1
        self.image_exp = 2
        self.baseline_range = 30
        self.min_intensity = 5
        self.max_deviation = 5
        self.min_percentile = 80
        self.percentile_size = 30
        self.savgol_size = 51
        self.savgol_polyorder = 3

    #  -----------------------------------------------------------------
    #
    #                          ** Load & Save **
    #
    #  -----------------------------------------------------------------

    def load(self, inputdir):
        """ loads in the oct_volume from the `inputdir` provided

        returns: 
            inputdat
        
        """

        inputdat = np.load(inputdir)
        fn = os.path.splitext(os.path.split(
            inputdir)[-1])[0].replace(" ", "_")  # make a filename for exports 
        data = self.get_sample(inputdat, step=self.step)

        self.data = data
        self.fn = fn
        self.true_nslcs = np.arange(0, 97, step=self.step) 

        return data
    
    def save(self, outputdat, outputdir, suffix):
        """ saves out the segmented layers

        args: 
            outputdat - data to be saved
            outputdir - directory to save data into
            suffix - append at end of filename (to identify data saved)
        
        """

        outputpath = outputdir + self.fn + suffix
        np.save(outputpath, outputdat)

        print("Output saved in {}".format(outputdir))

    def save_img(self, outputimg, outputdir, suffix):
        """ saves out the images

        args: 
            outputimg - images to be saved
            outputdir - directory to save data into
            suffix - append at end of filename (to identify data saved)
         
        """

        outputpath = outputdir + self.fn + suffix
        outputimg.savefig(outputpath, dpi='figure', format="png")

        print("Image saved in {}".format(outputdir))



    #  -----------------------------------------------------------------
    #
    #                          ** Preprocess **
    #
    #  -----------------------------------------------------------------

    def preprocess(self, oct_volume):

        """ Runs oct_volume data through intensity_scaler() and convolve2d().
        
        returns:
            exp_vol - OCT volume to the power of specified exponent
            conv_vol - convolved OCT data

        """
        
        self.nslcs, self.nrows, self.ncols = oct_volume.shape
        self.slc_dimension = (self.nslcs, self.ncols)
        self.vol_dimension = (self.nslcs, self.nrows, self.ncols)
        
        self.kernel = (
            np.tile(self.kernel,(
                1, self.kernel_dim)))/np.size(self.kernel)  # adjust kernel accordingly.

        exp_vol = np.zeros(oct_volume.shape)
        conv_vol = np.zeros(oct_volume.shape)

        for nslc in range(self.nslcs):
            slc = oct_volume[nslc,:,:]
            exp_slc = self.intensity_scaler(slc, exp=self.image_exp)
            conv_slc = convolve2d(exp_slc, self.kernel, mode="same")  
            exp_vol[nslc,:,:] = exp_slc
            conv_vol[nslc,:,:] = conv_slc

        if self.nrows >=495:
            print("Preprocessing completed for ILM")
            self.exp_data = exp_vol
        else:
            print("Preprocessing completed for RPE")
            

        return conv_vol

    
    def get_sample(self, oct_volume, step):
        """ returns a smaller sample of OCT volume if step > 1.
        
        """

        total_slcs = 97
        slice_range = np.arange(0, total_slcs, step)
        oct_data = oct_volume[slice_range,:,:]

        return oct_data
            
    def intensity_scaler(self, slc, exp=2):
        """ scales the intensity of each pixel by exponent provided. 
        
        """

        scaler = MinMaxScaler(feature_range=(slc.min(), slc.max()))
        exp_slc = scaler.fit_transform((slc.copy()**exp))  # Raise pixel intensities to image-exp provided.

        return exp_slc



    #  -----------------------------------------------------------------
    #
    #                            ** Segment **
    #
    #  -----------------------------------------------------------------

    def segment(self, data=None):
        """ performs segmentation on data
        note: data should be convolved data (i.e., `conv_oct`)

        requires additional methods:
            find_bright_spots
                __detect_outlier
                __select_outlier
            smooth_layer
                __replace_dead_pixels
        
        returns oct_layers

        """

        while data is None:
            print("No data input detected! *Hint: try`conv_oct`.")
            break

        else:

            oct_layers = np.zeros((self.nslcs, self.ncols))

            for nslc in range(self.nslcs):
                slc = data[nslc,:,:]
                bright_spots = self.find_bright_spots(slc)
                layer = self.smooth_bright_spots(slc, bright_spots)
                
                # if this is RPE!
                if self.nrows < 496: 
                    layer = [u+v+self.skip_over for u,v in zip(
                        self.ilm_layers[nslc,:], layer)] 

                oct_layers[nslc,:] = layer
            
            return oct_layers


    def find_bright_spots(self, slc):
        """ takes slice data and finds bright spots per column.
        note: `slc` should be convolved data.

        """

        bright_spots = []

        for ncol in range(self.ncols):
            col_data = slc[:,ncol]
            baseline_intensity = col_data[:self.baseline_range]  # establish a baseline intensity at the edges of the image
            col_outliers = self._detect_outlier(col_data, baseline_intensity)
            col_outlier = self._select_outlier(col_outliers)
            bright_spots.append(col_outlier)

        return bright_spots

    def _detect_outlier(self, data, baseline_intensity):
        """ finds outliers that are above a certain threshold.

        """

        outliers = [
            i for i,j in enumerate(data) if (abs(j) > self.min_intensity) & (
                abs(j) > (np.mean(baseline_intensity)+(np.std(baseline_intensity)*self.max_deviation)))]

        return outliers

    def _select_outlier(self, outliers):
        """ selects the 'earliest' outlier

        """

        outlier = min(outliers) if len(outliers) > 0 else -1

        return outlier

    
    def smooth_bright_spots(self, slc, bright_spots):
        """ smooths the bright spots using a percentile filter replaces the dead pixels via savgol filter, returns nearest layer as integer pixel positions.
        
        """

        bright_spots_pf = percentile_filter(bright_spots, self.min_percentile, self.percentile_size)
        
        bright_spots_rdp = self._replace_dead_pixels(slc, bright_spots_pf)
        #  sticking this here because I know it works, but a weird place to correct dead pixels, will change later. // 

        bright_spots_sg = savgol_filter(np.array(bright_spots_rdp), self.savgol_size, self.savgol_polyorder)
        layer = [int(x) for x in bright_spots_sg]
    
        return layer
    
    def _replace_dead_pixels(self, slc, bright_spots):
        """ replaces the dead pixels (columns that are all 0s) with nearest bright_spots 
        
        """

        dead_pixels = [
            ncol for ncol in range(self.ncols) if np.all(slc[:,ncol]==0)
            ]  # find the dead pixels

        if len(dead_pixels) > 0:  # if there are any dead pixels 
            left_edge = [
                pixel_pos for pixel_pos in dead_pixels if pixel_pos < self.ncols/10
                ]
            right_edge = [
                pixel_pos for pixel_pos in dead_pixels if pixel_pos > (self.ncols-self.ncols/10)
                ]

        try:
            if len(left_edge) > 0:
                bright_spots[left_edge] = bright_spots[
                    max(left_edge):(max(left_edge)+len(left_edge))
                    ]
        except UnboundLocalError:
            pass
        
        try: 
            if len(right_edge) > 0:
                bright_spots[right_edge] = bright_spots[
                    (min(right_edge)-len(right_edge)):min(right_edge)
                    ]
        except UnboundLocalError:
            pass
      
        return bright_spots


    #  -----------------------------------------------------------------
    #
    #                           ** Correct **
    #
    #  -----------------------------------------------------------------

    def correct(self, *args):
        """ for ilm, the logic is to find big changes in derivatives at spaced out intervals of the layer. pixels directly adjacent to each other tend to have pretty similar derivatives. we split the returned layer by non-consecutive values to indicate a 'big jump' and remove the smaller chunks (that are likely noise). we then fill in the gaps by interpolating. detect > split > remove > fill

        returns corrected_layer, corrected_layer_pix
        
        """

        if len(args) > 1:  # both layers
            ilms, rpes = args[0], args[1]
            corrected_layer, corrected_layer_pix = self._correct_rpe(
                ilms, rpes)
            print(" >>>>>>>>>> 3/3: RPE Correction Complete <<<<<<<<<<< ")

        else:
            ilms = args[0]
            corrected_layer, corrected_layer_pix = self._correct_ilm(ilms)
            print(" >>>>>>>>>> 3/3: ILM Correction Complete <<<<<<<<<<< ")

        return corrected_layer


    def _correct_ilm(self, ilms):
        """ for ilm, the logic is to find big changes in derivatives at spaced out intervals of the layer. pixels directly adjacent to each other tend to have pretty similar derivatives. we split the returned layer by non-consecutive values to indicate a 'big jump' and remove the smaller chunks (that are likely noise). we then fill in the gaps by interpolating. detect > split > remove > fill.

        returns corrected_ilm
        
        """

        corrected_ilm = np.zeros(self.slc_dimension)
        corrected_ilm_pix = np.zeros(self.slc_dimension)

        for nslc in range(self.nslcs):

            ilm = ilms[nslc,:]

            # find the big spikes in this layer
            derivative_spikes = self._find_derivative_spikes(ilm, step=10)
           
            # find patches of the layer that does not have big spikes
            pix = [i for i in range(len(ilm)) if not i in derivative_spikes]

            # we assume consistency is a good thing and sift out the small, sporadic chunks
            pix = self._find_true_chunks(pix, min_size=50)

            # drop the pixels that didn't pass our criteria
            ilm_to_keep = [j if i in pix else np.nan for i,j in enumerate(ilm)]

            # now let's interpolate!
            # ... first few and last few slices tend to be flatter
            # k = 1 if (nslc < 30) or (nslc > 80) else 3
            # edit: i'm not sure, k=1 seems safer...?
            true_nslc = self.true_nslcs[nslc]
            k = 1 if (true_nslc < 30) or (true_nslc > 80) else 3
            pix_final, ilm_final = self._interpolate_layer(
                pix, ilm_to_keep, k=1, backup=190)

            corrected_ilm[nslc,:len(ilm_final)] = ilm_final
            corrected_ilm_pix[nslc,:len(pix_final)] = pix_final

        
        for nslc in range(self.nslcs):
            corrected_ilm[nslc,:] = self._replace_w_nearest(
                corrected_ilm[nslc,:])


        return corrected_ilm, corrected_ilm_pix

    def _correct_rpe(self, ilms, rpes):
        """ the logic for rpe correction is to look for spots of the rpe that is of a certain distance away from the ILM at least, and of a certain intensity (higher than its surrounding pixels). 
        
        returns rpe

        """

        corrected_rpe = np.zeros(self.slc_dimension)
        corrected_rpe_pix = np.zeros(self.slc_dimension)

        print(self.exp_data.shape)  # still looking good here

        for nslc in range(self.nslcs):

            slc, ilm, rpe = self.exp_data[nslc,:,:], np.array(
                ilms[nslc,:]), np.array(rpes[nslc,:])

            bad_pix_by_distance = self._by_distance(
                ilm, rpe, window_size=200)
            bad_pix_by_intensity = self._by_intensity(
                slc, rpe, window_size=30, margin=10)

            bad_pix_combined = list(
                sorted(set(bad_pix_by_distance + bad_pix_by_intensity)))
            pix = [i for i in range(len(rpe)) if not i in bad_pix_combined]

            rpe_to_keep = [
                j if i in pix else np.nan for i,j in enumerate(rpe)
                ]

            #rpe_final = self._replace_w_nearest(rpe_to_keep)

            rpe_final = rpe_to_keep

            corrected_rpe[nslc,:len(rpe_final)] = rpe_final
            corrected_rpe_pix[nslc,:len(pix)] = pix  # pix = pix_final
        
        # This is outside the loop because the corrected_rpe matrix starts with 0s, and correcting within (before appending) is too early.
        
        """for nslc in range(self.nslcs):
            corrected_rpe[nslc,:] = self._replace_w_nearest(
                corrected_rpe[nslc,:])"""
        

        return corrected_rpe, corrected_rpe_pix


    def _find_derivative_spikes(self, layer, step=10):
        """ performs np.diff on every n-th value in the layer and finds the big ones.

        """

        while step < 1:
            print("Error in find_derivative_spikes(): step must be > 1.")
            break

        else:
            layer_by_step = [
                layer[i] for i in np.arange(0, len(layer), step)
                ]  # the data to perform diff on.
            derivatives_by_step = np.diff(layer_by_step)
            derivatives = np.repeat(derivatives_by_step, step)  # respaces derivs back to original shape
            derivative_spikes = [i for i,j in enumerate(derivatives) if abs(j) > step]

            return derivative_spikes
        
    def _find_true_chunks(self, pix, min_size=50):
        """ split by non-consecutive chunks and filter out anything below `min_size`
        
        """

        chunks = self._split_by_consecutive_number(pix)
        true_chunks = []

        for chunk in chunks:
            if len(chunk) > min_size:
                true_chunks += chunk

        return true_chunks
    
    def _split_by_consecutive_number(self, lst):
        """ splits `lst` into chunks based on consecutive numbers. 

        """

        lst = iter(lst)
        val = next(lst)
        chunk = []
        try:
            while True:
                chunk.append(val)
                val = next(lst)
                if val != chunk[-1] + 1:
                    yield chunk
                    chunk = []
        except StopIteration:
            if chunk:
                yield chunk

    def _interpolate_layer(self, pix, layer_by_pix, k, backup):
        """ X = pixels_to_keep, y = layer_by_pix
        note: depending on the pixels_to_keep, it is possible that the returned variables are not ncols long.
        
        if it's ilm: bckup should be 190, if not, it should be 190 + 40?
        
        returns all pixels + layer with filled vectors
        """

        while min(pix) != 0: # if the first pixel kept isn't the first pixel...
            pix = [0] + pix
            layer_by_pix = [backup] + layer_by_pix
            
        else:
            backup = layer_by_pix[0]
            
        X, y = pix.copy(), layer_by_pix.copy()  # assign values for interpolation.
        y_real = [m for m in y if not np.isnan(m)]  # y is any value that isn't np.nan
        tck = interpolate.splrep(x=X, y=y_real, k=k)
        X_filled = np.arange(min(X), max(X), 1)
        y_filled = interpolate.splev(X_filled, tck, der=0)
        
        return X_filled, y_filled

    def _replace_w_nearest(self, layer):
        """ returns layer where zeroes are substituted with the closest non zero values

        """

        patched = np.zeros(len(layer))
        filler = layer[next((m for m,n in enumerate(layer) if n!=0))]  # find the first nonzero value in the layer

        for i,j in enumerate(layer):

            if j > 0:  # if the value is legit
                filler = j
            else:
                pass
            patched[i] = filler

        patched = [int(p) for p in patched]
        
        return patched

    def _by_distance(self, ilm, rpe, window_size=200):
        """ The distance between RPE and ILM should never be smaller than the average of the two edges. """
        distance = list(rpe - ilm)
        midpoint = int(len(distance)/2) # find the 'midpoint'
        left_avg, right_avg = np.mean(distance[:window_size]), np.mean(distance[-window_size:]) # average across
        # Find the indices where the value is smaller than the mean
        left_idx = [idx+window_size for idx,value in enumerate(distance[window_size:midpoint]) if value < left_avg]
        right_idx = [idx+midpoint for idx,value in enumerate(distance[midpoint:-window_size]) if value < right_avg]
        bad_idx = left_idx + right_idx # since bad index differs in size from slice to slice... you can't loop slice_range within this method.
        return bad_idx 

    def _by_intensity(self, slc, rpe, window_size=30, margin=10):
        """ This finds bad fits but looking for intensity changes.
        A good fit should have intensity changing by dark-light-dark.
        
        """

        bad_idx = []
        for ncol in range(self.ncols):
            # Here we're looking for local intensity, a vertical vector straddling the rpe.
            startpoint = int(rpe[ncol])
            local_intensity = slc[(startpoint-window_size):(
                startpoint+window_size), ncol]
            # Convolve = blurs and takes first deriv.
            try: 
                conv_intensity = np.convolve(local_intensity, [
                    .1, .1, .1, .1, .1, .1, .1, .1, .1, .1], mode='same')
                peak_idx = np.where(
                    conv_intensity == max(conv_intensity))[0][0] 
                    # Find the index where the highest convolved intensity is found.
                criterion_met = (peak_idx > window_size-margin) & (
                    peak_idx < window_size+margin) 
                    # does it lie within the margin
                if not criterion_met:
                    bad_idx.append(ncol)
            except ValueError:
                bad_idx.append(ncol)

        return bad_idx
    

    def search_grid(self, ilm_layers_mat):
        """ returns a search grid for RPE based on ILM layer

        args: grid_height, ncols, skip_over
        
        """
        
        # Set the RPE specific kwargs ---> THIS IS BAD BUT FOR TESTING
        self.baseline_range = 15
        self.zscore_threshold = 3
        self.percentile_threshold = 99
        self.savgol_polyorder = 1
        self.grid_height = 100
        self.skip_over = 20

        self.ilm_layers = ilm_layers_mat  # for the adjustment

        oct_volume_rpe = np.zeros((self.nslcs, self.grid_height, self.ncols))

        for nslc in range(self.nslcs):
            slc = self.data[nslc,:,:]
            ilm = ilm_layers_mat[nslc,:]

            search_grid = np.zeros((self.grid_height, self.ncols))
            skipped = [x + self.skip_over for x in ilm]

            for col in range(self.ncols):
                start_at = int(skipped[col]) 
                grid_data = slc[start_at:(start_at+self.grid_height), col]
                if grid_data.shape[0] != self.grid_height:  # sometimes it looks one off due to rounding...
                    missing_pixels = grid_data.shape[0] - self.grid_height
                    if missing_pixels < 0:   # if grid_height is greater than grid_data...
                        grid_data = np.concatenate(grid_data, np.zeros)
                        grid_data = self._replace_w_nearest(grid_data)
                    elif grid_data.shape[0] > self.grid_height:  # if there's more data than we want for some reason... 
                        grid_data = grid_data[:self.grid_height]
                search_grid[:,col] = grid_data
            
            oct_volume_rpe[nslc,:,:] = search_grid

        return oct_volume_rpe


    #  -----------------------------------------------------------------
    #
    #                        ** Visualization **
    #
    #  -----------------------------------------------------------------

    def plot_subplot_idx(self):
        """ Returns subplot index so plots are evenly spaced out.
        
        """

        slc_lst = list(np.arange(1,self.nslcs+1,1))

        num_cols = 5
        max_rows = 4
        max_subplots = num_cols * max_rows  # 20

        num_plots = int(math.ceil(self.nslcs/max_subplots))  # 5

        subplot_index = []

        for plot_num in range(num_plots):

            x = (plot_num * max_subplots)  # 0, 20, 40, 60, 80 starting
            y = x + max_subplots
            plot_i = slc_lst[x:y]
            subplot_index.append(plot_i)

        return subplot_index

    def plot_layers_2d(
        self, *args):
        """ Plots the segmented layers out on image.

        """

        if len(args) == 1:
            ilm = args[0]
        elif len(args) == 2:
            ilm, rpe = args[0], args[1]
        else:
            print("Error in plotting: check number of arguments.")
            raise IndexError

        nc = 5  # always 5 columns for plotting
        nr = math.ceil(self.nslcs/nc)  # round up nrows in total

        fig,ax = plt.subplots(nr,nc,figsize=(25,7))  
        for n in range(self.nslcs):
            r,c = n//nc,n%nc
            rc = c if nr == 1 else (r,c)  # indexing subplots
            true_nslc = self.true_nslcs[n]
            ax[rc].title.set_text("Slice {}".format(str(true_nslc)))
            ax[rc].imshow(self.exp_data[n,:,:], cmap='gray') # viridis
            if "ilm" in locals():
                ax[rc].plot(ilm[n,:], color='r', alpha=1, linewidth=3)
            if "rpe" in locals():
                ax[rc].plot(rpe[n,:], color='y', alpha=1, linewidth=3)
            ax[rc].set_xlim((0,768))
            ax[rc].set_ylim((496,0))
            ax[rc].legend(['ILM', 'RPE'], loc='lower left')
        fig.tight_layout()

        return fig

    def plot_layers_3d(
        self,
        ilm_layers,
        rpe_layers):
        """ Plotting 3d meshes
        
        """
        pass
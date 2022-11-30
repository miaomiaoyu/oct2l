#!usr/bin/env python3
#
# MY, 29 Nov 2022
# mmy@stanford.edu

'''
OCT2L prototype run
'''


def main():
    
    print('prototype')
    
    import time
    import sys
    sys.path.insert(1, '')
    from proto import OCT2L

    model = OCT2L()

    tic = time.time()
    volume = model.loading('../data/npy/ODD_079_OD.npy')   # make this a folder loop
    c_volume = model.preprocessing(volume)  # convolved volume

    # -- ilm segmentation
    surface = model.segmenting(c_volume)   # get ilm surface
    ilm_surface = model.correcting(surface)  # some more smoothing?

    # -- rpe segmentation
    cr_volume = model.cropping(volume, ilm_surface, height=150)  
    #surface = model.segmenting(c_volume)   # get ilm surface
    #ilm_surface = model.correcting(surface)  # some more smoothing?

    _ = model.saving(cr_volume, 'cropped_volume.mat')

    toc = time.time()
    print("time elapsed: %.1f seconds" % (toc-tic))

    #positions = model.canny(cr_volume)
    #positions+=150
    #fig = model.visualizing(volume, positions)

    """

    rpe_surface = model.segmenting(cr_volume)   # get rpe surface
    spikes = model.spikes_get(rpe_surface, stride=8)  # some more smoothing?
    rpe2 = model.spikes_remove(rpe_surface, spikes)
    rpe3 = model.surface_interpolate(rpe2)
    rpe3+=(ilm_surface+50)  # postprocessing

    #ilm_figure = model.visualizing(volume, ilm_surface)
    #print(' >> %s seconds ' % (time.time() - st))
    #cr_volume = model.cropping(volume, ilm_surface)
    #print(cr_volume.shape)

    import cv2
    %matplotlib inline
    for k in [50]:
        edge = cv2.Canny(np.uint8(cr_volume[k,:,:]), 50, 150, apertureSize=3)
        np.save('/tmp/canny2.npy', edge)

    """

'''
if __name__ == "__main__":
    args = parse_args()
    main(args)

'''
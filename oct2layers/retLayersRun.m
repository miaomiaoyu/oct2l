clear; close all
%% 
% surfaceGet.m loads OCT volumes and runs cannyEdgeDetector.m to obtain edges.
% ILM: Selects and reconstructs the edges closest to top of image.
% RPE: Selects and reconstructs edges found in within 100 pix of ILM.


%% Load OCT Volumes

volumePath = '~/workspace/oct2l/data/project-files-2022-11-30-mat';
volumeDir = dir(fullfile(volumePath, '*.mat'));

for thisVol = 1:3%5:length(volumeDir)
    f = volumeDir(thisVol).name;
    d = load(fullfile(volumePath, f));
    octVolume = d.d;
    octVolumeGamma = octVolume.^2;

    [nSlices, nRows, nCols] = size(octVolume);

    %% Run retLayerGet.m
    ilmEdges = cannyEdgeDetector(octVolumeGamma, .6);
    ilmLayers = retLayerGet(ilmEdges);
    ilmLayersSmooth = retLayerSmooth(ilmLayers);
    ilmLayersSmoothNoNans = fillmissing(ilmLayers, 'linear', 2, 'EndValues','nearest');
    
    meshBuilder(ilmLayersSmooth)

    height = 150;
    padding = 30;
    [octVolumeSample, rowIndices] = octVolumeSampleGet(octVolume, ilmLayersSmoothNoNans, height, padding);
      
    %% Run retLayerGet.m
    rpeEdges = cannyEdgeDetector(octVolumeSample, .95);
    rowStarts = squeeze(rowIndices(:,1,:));
    rpeLayersPreAdjust = retLayerGet(rpeEdges);
    rpeLayers = rpeLayersPreAdjust+rowStarts;
    rpeLayersSmooth = retLayerSmooth(rpeLayers);

    %% Check by plotting
    sliceNum = 21;

    octImageSample = squeeze(octVolumeSample(sliceNum,:,:));
    octImage = squeeze(octVolume(sliceNum,:,:));
    
    bw = rpeEdges;
    r1 = rpeLayers;
    r2 = rpeLayersSmooth;
    r3 = rpeLayersSmooth;

    BW = squeeze(bw(sliceNum,:,:));
    R1 = squeeze(r1(sliceNum,:,:));
    R2 = squeeze(r2(sliceNum,:,:));
    R3 = squeeze(r3(sliceNum,:,:));
  

    figure
    imo = imoverlay(uint8(octImageSample), BW, 'r');
    imshow(imo);
    
    figure
    hold on
    imagesc(octImage)
    plot(R1,'LineWidth',2,'Color','r')
    plot(R2,'LineWidth',2,'Color','y')
    plot(R3,'LineWidth',2,'Color','m')
    ylim([0,496])
    hold off

    %%

end



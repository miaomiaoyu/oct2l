function [] = retLayerPlot(octVolume, ilmLayers, rpeLayers)
%% plots surfaces constructed
% retLayerPlot plots surfaces against the octVolume
%
stride = 20;

[nSlices, ~, ~] = size(octVolume);
slicesToPlot = 1:stride:nSlices;

figure

for i = 1:length(slicesToPlot)

    thisSlice = slicesToPlot(i);
    img = squeeze(octVolume(thisSlice,:,:));

    subplot(1,5,i)
    t = ['Slice ', num2str(thisSlice)];
    title(t);
    hold on
    imshow(uint8(img)) % OCT B-scan image
    plot(ilmLayers(thisSlice,:), 'r')
    plot(rpeLayers(thisSlice,:), 'y')
    hold off

end

end

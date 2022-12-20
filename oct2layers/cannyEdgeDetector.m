function edges = cannyEdgeDetector(volume, threshold)
% cannyEdgeDetector: runs through volume and finds edges via the canny edge detector.
% returns edges, a 2-D matrixs

[nSlices, nRows, nCols] = size(volume);
edges = zeros([nSlices, nRows, nCols]);

for thisSlice = 1:nSlices
    img = squeeze(volume(thisSlice,:,:));
    img = uint8(img);
    bw = imbinarize(img);
    edges(thisSlice,:,:) = edge(bw, 'Canny', threshold);
end

end
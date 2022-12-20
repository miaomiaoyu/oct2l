function [octVolumeSample, rowIndices] = octVolumeSampleGet(octVolume, retLayer, height, padding)
%%
% octVolumeSampleGet returns the top layer of edges along 2nd dimension.

[nSlices, nRows, nCols] = size(octVolume);
octVolumeSample = zeros([nSlices, height, nCols]);
rowIndices = zeros([nSlices,2,nCols]);

for thisSlice = 1:nSlices
    for thisCol = 1:nCols
        rowStart = round(retLayer(thisSlice,thisCol)) + padding;
        rowEnd = rowStart+height-1;

        if rowEnd > nRows
            rowEnd = nRows; % so it won't go over
        end
        nRowsSample = rowEnd-rowStart+1;
        octVolumeSample(thisSlice,1:nRowsSample,thisCol) = octVolume(thisSlice,rowStart:rowEnd,thisCol);
        rowIndices(thisSlice,:,thisCol) = [rowStart;rowEnd];
    end
end
    
end

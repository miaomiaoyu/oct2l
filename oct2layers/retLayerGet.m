function [retLayer] = retLayerGet(edges)
%retLayerGet returns the top layer of edges along 2nd dimension.

[nSlices, ~, nCols] = size(edges);
retLayer = zeros([nSlices, nCols]);

for thisSlice = 1:nSlices
    for thisCol = 1:nCols
        j = find(edges(thisSlice,:,thisCol),1,'first');
        if isempty(j)
            j = NaN;
        end
        retLayer(thisSlice,thisCol) = j;
    end
end
    
end


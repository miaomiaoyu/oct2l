function [surfaceSmooth] = retLayerSmooth(surface)
%% 
% retLayerSmooth returns the top layer of edges along 2nd dimension.
% filters used: medfilt2

[nSlices, ~] = size(surface);
surfaceSmooth = zeros(size(surface));

for thisSlice = 1:nSlices

    layer = surface(thisSlice,:);

    % First derivative
    layerDiff = diff(layer);
    badIndex = find(abs(layerDiff) > 5);
    badIndex = badIndex+1;
    layer(badIndex) = NaN;
    
    % Median filter
    layer = medfilt2(layer, [1,7]);



%     % Moving median
%     layerNans = isnan(layer);
%     CC = bwconncomp(layerNans);
%     sizes = cellfun(@length, CC.PixelIdxList);
%     window = max(sizes)*2;
%     if window < 7
%         window = 7;
%     end
%     F = fillmissing(layer, 'movmedian', window=window);
%     % This is good for ILM but we cannot have NaNs for octVolumeSampleGet.m
% 
%     layer = F;
    surfaceSmooth(thisSlice,:) = layer;

end
    
end


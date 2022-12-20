function edges = cannyDetector(volume)
% cannyDetector: runs through volume and finds edges via the canny edge detector.
% returns edges, a 2-D matrix

[K,N,M] = size(volume);
threshold = .8;
edges = zeros([K,N,M]);

for k = 1:K
    img = squeeze(volume(k,:,:));
    img = uint8(img);
    bw = imbinarize(img);
    edges(k,:,:) = edge(bw, 'Canny', threshold);
end

end
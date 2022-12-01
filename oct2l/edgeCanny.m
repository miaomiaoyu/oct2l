function edges = edgeCanny(volume)
% edgeCanny: runs through volume and finds edges via the canny edge detector.
% returns a 

[K,N,M] = size(volume);
threshold = 95;
rpe = zeros([K,M]);

for k = 1:K
    img = squeeze(volume(k,:,:));
    img = uint8(img);
    bw = imbinarize(img);
    rpe(k,:) = edge(bw, 'Canny', threshold);
end



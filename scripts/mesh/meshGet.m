addpath(genpath())

%% Build mesh from surface
clear; close all

fileDir='surfaces/*surface.mat';
d = dir(fileDir);
for i = 1:length(d)
    a = load(d(i).name);
    surface = a.surface;
    meshBuilder(surface)
end


%% Cropped volume
close all

a = load("tmp/cropped_volume.mat");
volume = a.cropped_volume;
[K,N,M] = size(volume);

threshold = .95;

for k = 1:2

    img = squeeze(volume(k,:,:));
    img = uint8(img);
    bw = imbinarize(img);
    bw1 = edge(bw, 'Canny', threshold);
    se90 = strel('line',3,90);
    se0 = strel('line',3,0);
    bwsdil = imdilate(bw1,[se90 se0]); % dilate
    bwsfil = imfill(bwsdil, 'holes'); % fill
    bwnobord = imclearborder(bwsfil,4);

    figure
    subplot(4,1,1)
    imshow(bw1)
    subplot(4,1,2)
    imshow(bwsdil)
    subplot(4,1,3)
    imshow(bwsfil)
    subplot(4,1,4)
    imshow(labeloverlay(img,bw1,'Colormap','autumn','Transparency',0.15))

end




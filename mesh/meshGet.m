addpath(genpath())

%% Build mesh from surface
clear; close all

basePath = '~/workspace/oct2l/output';
inputFolder = '02';
inputPath = fullfile(basePath, inputFolder);
inputDir = dir(inputPath); % dir not path

for i = 1:3%length(inputDir)
    
    if contains(inputDir(i).name, '_02')  % ignore .DS_store
        fName = inputDir(i).name;
        inputFile = fullfile(inputPath, fName);
        
        f = load(inputFile);
        f = f.f; % idk why but somehow it's saved like this

        volume = f.octvolume;
        ilm = f.ilm;
        rpe = f.rpe;
        
        topSurface = zeros(97,768);

        for thisSlice=1:97
            for thisCol=1:768
                c = find(rpe(thisSlice,:,thisCol));
                if ~isempty(c)
                    topSurf = min(c);
                else
                    topSurf = NaN;
                end
                topSurface(thisSlice,thisCol) = topSurf;
            end
        end
    end
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




function [ilmVol] = ilmLayerGet()
% ilmLayer
% Loads
% @ilmLayerGet -> @cannyDetector -> save out to the same file. 
% MY, 2022

basePath = '~/workspace/oct2l/output';
inputFolder = '01';
inputPath = fullfile(basePath, inputFolder);
inputDir = dir(inputPath); % dir not path

for i = 1:length(inputDir)

    if contains(inputDir(i).name, 'ODD-')  % ignore .DS_store

        fName = inputDir(i).name;

        inputFile = fullfile(inputPath, fName);

        f = load(inputFile); 
        ilmSurf = f.ilm;
        octVol = f.octvolume;

        [nSlices,nRows,nCols] = size(octVol);

        ilmVol = cannyDetector(octVol); % Canny Edge Detector

        sliceIndex = 1:10:nSlices;

        rpeBin = imbinarize(ilmVol); % binarize rpe for plotting
        
        figure
        for ii = 1:length(sliceIndex)
            sliceNum = sliceIndex(ii);

            img = squeeze(octVol(sliceNum,:,:));
            subplot(2,5,ii)
            imshow(uint8(img)) % volume
            hold
            B = imoverlay(uint8(img), squeeze(ilmVol(sliceNum,:,:)), 'r');
            %B = imoverlay(uint8(img), topSurface(sliceNum,:), 'y');
            imshow(B); % rpe
            %plot(ilmVol(sliceNum,:), 'y')
            %plot(ilm(sliceNum,:), 'r') % ilm
            %legend('on')
        end


    end
end

   


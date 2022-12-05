function [] = rpe_layer_get()
% rpe_layer_get()
% Loads OCT volume and ILM surface, and segments RPE using Canny Edge 
% Detector in the area below the ILM layer on each slice image.
% @rpe_layer_get() -> @cannyDetector() -> save out to the same file. 
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
            octVolCropped = f.octvolumecropped;
   

            [K,N,M] = size(octVol);
            [~,height,~] = size(octVolCropped);
            padding = 30;

            rpeVolCropped = cannyDetector(octVolCropped); % Canny Edge Detector
            % size(edges) -> 97x150x768

            rpeVol = zeros(K,N,M);
            for k=1:K  % by slice
                startIndex = ilmSurf(k,:) + padding; % ILM + 30 allowance
                endIndex = startIndex + height; 
                rpeSlice = squeeze(rpeVolCropped(k,:,:)); % get slice of rpeVolCrop
                for m=1:M
                    rpeVol(k, round(startIndex(m)):round(endIndex(m))-1, m) = rpeSlice(:,m);
                end
            end

            % add rpeVol to struct f
            f.rpe = rpeVol;

            fNameOut = strrep(fName, '_01', '_02');
            outputFolder = '02';
            outputPath = fullfile(basePath, outputFolder);
            if ~(exist(outputPath, 'dir'))
                mkdir(outputPath)
            end
            outputFile = fullfile(outputPath, fNameOut);
            save(outputFile, 'f', '-mat');
            disp('%')
        end
    
    end
    
    fprintf('rpe_layer_get completed.')
   


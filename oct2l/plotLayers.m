function [] = plotLayers()
%plots surfaces constructed
%
%
%     arguments
%         layer   str {mustBeMember(layer,['ilm','both'])} = 'ilm'
%         stride  int {mustBeInRange(stride,1,10)} = 10
%     end
%%

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

            for thisSlice=1:97
                topSurface(thisSlice,:) = medfilt2(topSurface(thisSlice,:),[1,7]);
            end

            nSlices = size(volume,1); % aka K elsewhere
            sliceIndex = 1:10:nSlices;

            rpeBin = imbinarize(rpe); % binarize rpe for plotting
            
            figure
            for ii = 1:length(sliceIndex)
                sliceNum = sliceIndex(ii);

                img = squeeze(volume(sliceNum,:,:));
                subplot(2,5,ii)
                imshow(uint8(img)) % volume
                hold
                %B = imoverlay(uint8(img), squeeze(rpeBin(sliceNum,:,:)), 'y');
                %B = imoverlay(uint8(img), topSurface(sliceNum,:), 'y');
                %imshow(B); % rpe
                plot(topSurface(sliceNum,:), 'y')
                plot(ilm(sliceNum,:), 'r') % ilm
                %legend('on')
            end
        end
    end
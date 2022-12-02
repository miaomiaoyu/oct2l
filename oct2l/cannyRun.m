function [] = cannyRun()
%Runs @cannyDetector on cropped volume.
%

outputDir = '~/workspace/oct2l/output';
chdir(outputDir);
volumeDir = fullfile(cd, 'tmp', 'CVOLS', '*.mat'); 
volumeDir = dir(volumeDir);
tic
for i = 1:length(volumeDir)
    fName = fullfile(volumeDir(i).folder, volumeDir(i).name);
    volume = load(fName);
    volume = volume.f;
    edges = cannyDetector(volume);
    fNameOut = split(volumeDir(i).name, '_crop.mat');
    fNameOut = fNameOut{1,1};
    fullfileOut = fullfile(cd, 'surface', 'RPE_Canny', fNameOut);
    save(strcat(fullfileOut,'.mat'), 'edges', '-mat')
    
end
toc
% figure
% img = squeeze(volume(3,:,:));
% edge = squeeze(edges(3,:,:));
% %img = imbinarize(img);
% img = uint8(img);
% imshow(labeloverlay(img,edge,'Colormap','autumn','Transparency',0.15))
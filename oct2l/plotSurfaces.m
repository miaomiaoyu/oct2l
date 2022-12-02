function [] = plotSurfaces(layer, stride)
%plots surfaces constructed

arguments
    stride {mustBeInRange(stride,1,10)}
end
layersIndex = linspace(1, 97, stride);

outputDir = '~/workspace/oct2l/output';
chdir(outputDir);

volDir = fullfile(outputDir, 'vol');
ilmDir = fullfile(outputDir, 'ilm');
rpeDir = fullfile(outputDir, 'rpe');

for i = 1:length(volDir):


switch layer
    case 'ilm'
        plotILM()
    case 'rpe'
        disp('rpe')
    case 'both'
        figure
        img = squeeze(volume(3,:,:));
        edge = squeeze(edges(3,:,:));
        %img = imbinarize(img);
        img = uint8(img);
        imshow(labeloverlay(img,edge,'Colormap','autumn','Transparency',0.15))
    otherwise
        fprintf("layer has to be 'ilm', 'rpe' or 'both'")
end

for i = 1:length(ilmDir)
    fName = fullfile(ilmDir(i).folder, ilmDir(i).name);
    ilm = load(fName);
    ilm = ilm.f;
    imshow()


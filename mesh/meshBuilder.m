function [] = meshBuilder(surface)
% octMeshBuild Constructs meshes from segmented surfaces
%   Produces 4 kinds of subplots.

    conv2Surface = conv2(surface,(ones(3,3)/9), "same");
    imgaussfiltSurface = imgaussfilt(surface,1);
    medfilt2Surface = medfilt2(surface, [9,1]);
    
    K = size(surface,1);
    M = size(surface,2);

    kMicron = 30;
    mMicron = 6;
    
    kMesh = 1:kMicron:kMicron*K;
    mMesh = 1:mMicron:mMicron*M;
    
    [X,Y] = meshgrid(mMesh,kMesh);
    
    figure(1)

    subplot(2,4,1)
    imagesc(surface)
    title('Original Surface')

    subplot(2,4,2)
    imagesc(conv2Surface)
    title('conv2 (kernel=7,1)')

    subplot(2,4,3)
    imagesc(imgaussfiltSurface)
    title('imgaussfilt (sigma=2)')

    subplot(2,4,4)
    imagesc(medfilt2Surface)
    title('medfilt2 (winSize=9,1)')

    subplot(2,4,5)
    mesh(X,Y,surface)
    ylim([0,496])

    subplot(2,4,6)
    mesh(X,Y,conv2Surface)
    ylim([0,496])

    subplot(2,4,7)
    mesh(X,Y,imgaussfiltSurface)
    ylim([0,496])
    
    subplot(2,4,8)
    mesh(X,Y,medfilt2Surface)
    ylim([0,496])
    

end
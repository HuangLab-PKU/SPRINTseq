function cell_segmentation(img_dir,outdir)
%% read image
I=imread(img_dir);

I_adjust=imadjust(I,[0.015 1],[0 1],0.6);
I_adapt=adapthisteq(I_adjust);
I_sharpen=imsharpen(I_adjust,'Radius',25,'Amount',25);
Dapi= I_sharpen;
Dapi = imadjust(Dapi); % contrast enhancement
ImSz = size(Dapi);
Debug = 0;
bwDapi = im2bw(Dapi,0.6);
bwDapi=imerode(bwDapi, strel('disk',2));
%% find local maxima 
dist = bwdist(~bwDapi);
dist0 = dist;
dist0(dist<2)=0;

ddist = imdilate(dist0, strel('disk',7));
ddist4 = imdilate(dist0, strel('disk',4));

impim = imimposemin(-dist0, imregionalmax(ddist));
impimad = imadjust(impim);
clear dist0

%% segment
% remove pixels at watershed boundaries
bwDapi0 = bwDapi;
Iwatershed=watershed(impim);
bwDapi0(watershed(impim)==2)=0;
impim_watershed=impim;
impim_watershed(watershed(impim)==0)=0;
ddist0=ddist4;
ddist0(watershed(impim)==0)=0;
ddist0(ddist0>0)=1;
ddist0_lable= uint32(bwlabel(ddist0));
image(label2rgb(ddist0_lable, 'jet', 'w', 'shuffle'));
CellMap0=ddist0_lable;
rProps0 = regionprops(CellMap0); % returns XY coordinate and area
BigEnough = [rProps0.Area]>200;
NewNumber = zeros(length(rProps0),1);
NewNumber(~BigEnough) = 0;
NewNumber(BigEnough) = 1:sum(BigEnough);
CellMap = CellMap0;
CellMap(CellMap0>0) = NewNumber(CellMap0(CellMap0>0));
    image(label2rgb(CellMap, 'jet', 'w', 'shuffle'));
% assign all pixels a label
labels = uint32(bwlabel(bwDapi0));
[d, idx] = bwdist(bwDapi0);
% now expand the regions by a margin
CellMap0 = zeros(ImSz, 'uint32');
Expansions = (d<10);
CellMap0(Expansions) = labels(idx(Expansions));
% get rid of cells that are too small    
rProps0 = regionprops(CellMap0); % returns XY coordinate and area
BigEnough = [rProps0.Area]>200&[rProps0.Area]<25000;
NewNumber = zeros(length(rProps0),1);
NewNumber(~BigEnough) = 0;
NewNumber(BigEnough) = 1:sum(BigEnough);
CellMap = CellMap0;
CellMap(CellMap0>0) = NewNumber(CellMap0(CellMap0>0));

CellYX = fliplr(vertcat(rProps0(BigEnough).Centroid)); % because XY
    image(label2rgb(CellMap, 'jet', 'black', 'shuffle'));
num=size(CellYX,1);

outfile_CellYX=fullfile(outdir,char(strcat('CellYX','.mat')));
outfile_CellMap=fullfile(outdir,char(strcat('CellMap','.mat')));
save (outfile_CellYX,'CellYX');
save (outfile_CellMap,'CellMap');
end


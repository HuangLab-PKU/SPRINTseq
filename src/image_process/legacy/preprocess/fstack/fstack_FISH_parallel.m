function fstack_FISH_parallel(basedir,destdir1,stackNum)
%% get folder names

d = dir(fullfile(basedir));  
isub = [d(:).isdir]; %# returns logical vector  
nameFolds = {d(isub).name}';  
nameFolds(ismember(nameFolds,{'.','..'})) = []; 

%% 
for k=1:length(nameFolds)
%% get file names
samplename=nameFolds{k};
d = dir(fullfile(basedir,samplename,'*.tif')); %d = dir(fullfile(basedir,samplename,'*cy5*.tif'));  
isub = [d(:).isdir]; %# returns logical vector  
nameFiles = {d(~isub).name}';  
nameFiles(ismember(nameFiles,{'.','..'})) = []; 
NumOfFiles=length(nameFiles);
seriesNum=floor(NumOfFiles/stackNum);
imlist=cell(1,NumOfFiles);
destdir=fullfile(destdir1,samplename);
mkdir(destdir);
    for n=1:NumOfFiles
        imlist{n}=fullfile(basedir,samplename,nameFiles{n});
    end
%% read image
if rem(NumOfFiles,stackNum)~=0
    break;
end
for m=1:seriesNum
    im = fstack(imlist((m-1)*stackNum+1:m*stackNum));
    imwrite(uint16(im),fullfile(destdir,['FocalStack_',num2str(m,'%03d'),'.tif']),'tif','Compression','none');
end
end
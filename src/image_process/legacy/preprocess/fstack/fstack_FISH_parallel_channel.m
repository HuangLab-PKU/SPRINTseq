function fstack_FISH_parallel_channel(basedir,destdir1,stackNum,channel)
%% get folder names

d = dir(fullfile(basedir,'cyc*'));  
isub = [d(:).isdir]; %# returns logical vector  
nameFolds = {d(isub).name}';  
nameFolds(ismember(nameFolds,{'.','..'})) = []; 

%% 
for k=1:length(nameFolds)
%% get file names
samplename=nameFolds{k};
if stackNum <5
    d = dir(fullfile(basedir,samplename,['*',channel,'*.tif'])); %d = dir(fullfile(basedir,samplename,'*cy5*.tif'));  
else
    d = dir(fullfile(basedir,samplename,['*',channel,'*.tif']));
end
isub = [d(:).isdir]; %# returns logical vector  
nameFiles = {d(~isub).name}';  
nameFiles(ismember(nameFiles,{'.','..'})) = []; 
NumOfFiles=length(nameFiles);
seriesNum=floor(NumOfFiles/stackNum);
imlist=cell(1,NumOfFiles);
destdir=fullfile(destdir1,[samplename,'_',channel]);
if exist(destdir,'dir')==7
    fprintf('Folder %s already exists!\n',destdir);
    continue;
end
mkdir(destdir);
    for n=1:NumOfFiles
        imlist{n}=fullfile(basedir,samplename,nameFiles{n});
    end
%% read image
 if rem(NumOfFiles,stackNum)~=0
     spritf('sample %s image number not match!',samplename);
     break;
 end
for m=1:seriesNum
    if stackNum <5
        im = imread(imlist{(m-1)*stackNum+1});
    else
        im = fstack(imlist((m-1)*stackNum+1:m*stackNum));
    end
    namestring=imlist{(m-1)*stackNum+1};
    Tindex=strfind(namestring,'-T');
    Tstring=namestring(Tindex+2:Tindex+5);
    TileNum=str2double(Tstring);
    imwrite(uint16(im),fullfile(destdir,['FocalStack_',num2str(TileNum,'%03d'),'.tif']),'tif','Compression','none');
end
end
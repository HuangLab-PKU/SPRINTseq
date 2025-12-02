function fstack_FISH(basedir,destdir1,stackNum)
%% get folder names

d = dir(fullfile(basedir));  
isub = [d(:).isdir]; %# returns logical vector  
nameFolds = {d(isub).name}';  
nameFolds(ismember(nameFolds,{'.','..'})) = []; 

%% 
for k=1:length(nameFolds)
%% get file names
samplename=nameFolds{k};
d = dir(fullfile(basedir,samplename));  
isub = [d(:).isdir]; %# returns logical vector  
nameFiles = {d(~isub).name}';  
nameFiles(ismember(nameFiles,{'.','..'})) = []; 
destdir=fullfile(destdir1,samplename);
mkdir(destdir);
%% read image
if rem(length(nameFiles),stackNum)~=1
    break;
end
seriesNum=length(nameFiles)/stackNum;
for m=1:seriesNum
    for n=1:stackNum
        imlist{n}=fullfile(basedir,samplename,nameFiles{(m-1)*stackNum+n});
    end
    im = fstack(imlist);
    imwrite(uint16(im),fullfile(destdir,['FocalStack_',num2str(m,'%03d'),'.tif']),'tif','Compression','none');
end
end
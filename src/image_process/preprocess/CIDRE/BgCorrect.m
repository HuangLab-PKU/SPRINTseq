function BgCorrect(basedir,destdir1)
%% get folder names

d = dir(fullfile(basedir));  
isub = [d(:).isdir]; %# returns logical vector  
nameFolds = {d(isub).name}';  
nameFolds(ismember(nameFolds,{'.','..'})) = []; 
parfor k=1:length(nameFolds)
%% CIDRE Correction
samplename=nameFolds{k};
sampledir = fullfile(basedir,samplename);  
destdir=fullfile(destdir1,samplename);
if exist(destdir,'dir')==7
    continue;
end
cidre([sampledir,'\'],'destination',destdir);
delete(fullfile(destdir,'cidre_model.mat'));
end
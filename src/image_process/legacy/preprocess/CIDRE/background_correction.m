function background_correction(basedir,destbasedir)
%% get folder names

d = dir(fullfile(basedir));  
isub = [d(:).isdir]; %# returns logical vector  
nameFolds = {d(isub).name}';  
nameFolds(ismember(nameFolds,{'.','..'})) = []; 
parfor k=1:length(nameFolds)
%% CIDRE Correction
samplename=nameFolds{k};
sampledir = fullfile(basedir,samplename);  
sampledestdir=fullfile(destbasedir,samplename);
if exist(sampledestdir,'dir')==7
    continue;
end
cidre([sampledir,'\'],'destination',sampledestdir);
delete(fullfile(sampledestdir,'cidre_model.mat'));
end
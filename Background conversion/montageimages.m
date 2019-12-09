%% Montage images

NEURALOUTROOT = 'G:\My Drive\School\Year6\CS230\Project\Imgs\neuraltest';
addpath(NEURALOUTROOT);
fileFolder = fullfile(NEURALOUTROOT);
dirOutput = dir(fullfile(fileFolder,'outup*.jpg'));
fileNames = string({dirOutput.name});
%% crop all images to the same size 
%%
allimgs = {};
for i=1:length(fileNames)
    img = imread(fileNames{i});
    I = imresize(img,[512 512]);
    allimgs{i}  = I;
    
    f = figure('visible','off');
    imshow(I);
    fileName = ['crop_' fileNames{i}]; 
    saveas(f,fileName,'jpg');
end

%%
NEURALOUTROOT = 'G:\My Drive\School\Year6\CS230\Project\Imgs\neuraltest';
addpath(NEURALOUTROOT);
fileFolder = fullfile(NEURALOUTROOT);
dirOutput = dir(fullfile(fileFolder,'crop_outup*.jpg'));
fileNames = string({dirOutput.name});
%%
close all;
figure(1);
montage(fileNames)

%% FOR GAN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear fileNames;
GANOUTROOT = 'G:\My Drive\School\Year6\CS230\Project\CartoonGan-tensorflow\output_images\comparison';
addpath(GANOUTROOT);
fileFolder = fullfile(GANOUTROOT);
dirOutput = dir(fullfile(fileFolder,'*.jpg'));
fileNames = string({dirOutput.name});
%% crop all images to the same size 
%%
allimgs = {};
for i=1:length(fileNames)
    img = imread(fileNames{i});
    I = imresize(img,[512 512]);
    allimgs{i}  = I;
    
    f = figure('visible','off');
    imshow(I);
    fileName = ['cropgan_' fileNames{i}]; 
    saveas(f,fileName,'jpg');
end

%%
GANOUTROOT = 'G:\My Drive\School\Year6\CS230\Project\CartoonGan-tensorflow\output_images\comparison';
addpath(GANOUTROOT);
fileFolder = fullfile(GANOUTROOT);
dirOutput = dir(fullfile(fileFolder,'cropgan*.jpg'));
fileNames = string({dirOutput.name});
%%
close all;
figure(1);
montage(fileNames)
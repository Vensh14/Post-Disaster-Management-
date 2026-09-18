clc;
clear;
close all;


%% Image folder

imageFolder = "I:\EEE\SEM_6\FYP_EE405\Disaster\Disaster_Images";


%% Get images

files = dir(fullfile(imageFolder,'*.jpg'));

numImages = length(files);

fprintf("Total Images : %d\n",numImages);



%% Store results

methods = ["BRISQUE","NIQE","PIQE"];

totalTime = zeros(3,1);
peakMemory = zeros(3,1);



%% ================================
% BRISQUE
% ================================

tic;

for i = 1:numImages

    img = imread(fullfile(imageFolder,files(i).name));

    score = brisque(img);

end

totalTime(1)=toc;



%% ================================
% NIQE
% ================================

tic;

for i = 1:numImages

    img = imread(fullfile(imageFolder,files(i).name));

    score = niqe(img);

end

totalTime(2)=toc;



%% ================================
% PIQE
% ================================

tic;

for i = 1:numImages

    img = imread(fullfile(imageFolder,files(i).name));

    score = piqe(img);

end

totalTime(3)=toc;



%% ================================
% Calculate metrics
% ================================


averageTime = totalTime ./ numImages;


throughput = numImages ./ totalTime;



%% Storage requirement

% MATLAB built-in functions
% no separate model files stored by user

storage = [
    0;
    0;
    0
];



%% Display Results

fprintf("\n====================================\n");
fprintf("IQA Computational Comparison\n");
fprintf("====================================\n");


for i=1:length(methods)

    fprintf("\n%s\n",methods(i));

    fprintf("----------------------------\n");

    fprintf("Total Time          : %.2f seconds\n",totalTime(i));

    fprintf("Average Time/Image  : %.4f seconds\n",averageTime(i));

    fprintf("Average Time/Image  : %.2f ms\n",averageTime(i)*1000);

    fprintf("Throughput          : %.2f images/sec\n",throughput(i));

    fprintf("Storage Requirement : %.2f MB\n",storage(i));

end
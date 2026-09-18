clc;
clear;
close all;

%% ==========================
% USER SETTINGS
% ===========================

csvPath = "I:\EEE\SEM_6\FYP_EE405\Disaster data\Data\OpenCV_IQA_Results.csv";

imageFolder = "I:\EEE\SEM_6\FYP_EE405\Disaster data\ALL_Images";

outputCSV = "I:\EEE\SEM_6\FYP_EE405\Disaster data\Data\MATLAB_IQA_Results.csv";

%% ==========================
% READ CSV
% ===========================

T = readtable(csvPath);

N = height(T);

BRISQUE = nan(N,1);
NIQE = nan(N,1);
PIQE = nan(N,1);

%% ==========================
% PROCESS IMAGES
% ===========================

for i = 1:N

    imgPath = fullfile(imageFolder, T.Name{i});

    if ~isfile(imgPath)
        fprintf("Missing : %s\n",T.Name{i});
        continue;
    end

    img = imread(imgPath);

    BRISQUE(i) = brisque(img);

    NIQE(i) = niqe(img);

    PIQE(i) = piqe(img);

    fprintf("%d / %d\n",i,N);

end

%% ==========================
% SAVE SCORES
% ===========================

T.BRISQUE_Score = BRISQUE;
T.NIQE_Score = NIQE;
T.PIQE_Score = PIQE;

writetable(T,outputCSV);

disp("CSV Saved.");

%% ==========================
% GROUND TRUTH
% ===========================

truth = logical(T.Quality);

%% ==========================
% EVALUATE METHODS
% ===========================

evaluateMethod(BRISQUE,truth,"BRISQUE");

evaluateMethod(NIQE,truth,"NIQE");

evaluateMethod(PIQE,truth,"PIQE");

%% =====================================================
function evaluateMethod(score,truth,name)

valid = ~isnan(score);

score = score(valid);
truth = truth(valid);

% Lower score = Better image
score = -score;

[X,Y,T,AUC] = perfcurve(truth,score,1);

J = Y-X;

[~,idx] = max(J);

threshold = T(idx);

prediction = score>=threshold;

TP = sum(prediction==1 & truth==1);
TN = sum(prediction==0 & truth==0);
FP = sum(prediction==1 & truth==0);
FN = sum(prediction==0 & truth==1);

accuracy = (TP+TN)/(TP+TN+FP+FN);

precision = TP/(TP+FP);

recall = TP/(TP+FN);

F1 = 2*(precision*recall)/(precision+recall);

fprintf("\n=====================================\n");
fprintf("%s\n",name);
fprintf("=====================================\n");

fprintf("Threshold : %.4f\n",threshold);
fprintf("Accuracy  : %.4f\n",accuracy);
fprintf("Precision : %.4f\n",precision);
fprintf("Recall    : %.4f\n",recall);
fprintf("F1 Score  : %.4f\n",F1);
fprintf("AUC       : %.4f\n",AUC);

end
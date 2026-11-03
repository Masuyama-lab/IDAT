% Copyright (c) 2025-2026 Naoki Masuyama
% SPDX-License-Identifier: MIT

clear

numTrials = 10;  % Number of trials

% Load dataset ============
dataDir = fullfile(fileparts(mfilename('fullpath')), '..', 'datasets');
load(fullfile(dataDir, 'iris.mat'));
% load(fullfile(dataDir, 'optdigits.mat'));

data = double(data);
target = double(target);

rng(0)

% Split data by class (shuffled within each class)
classLabels = unique(target);
numClasses  = length(classLabels);
classData   = cell(numClasses, 1);
classTarget = cell(numClasses, 1);
for c = 1:numClasses
    idx            = find(target == classLabels(c));
    idx            = idx(randperm(length(idx)));
    classData{c}   = data(idx, :);
    classTarget{c} = target(idx);
end

% Initialize arrays to collect results
numNodesArray       = zeros(numTrials,1);
numActiveNodesArray = zeros(numTrials,1);
numClustersArray    = zeros(numTrials,1);
ariArray            = zeros(numTrials,1);
amiArray            = zeros(numTrials,1);
timeArray           = zeros(numTrials,1);
incAmiArray         = zeros(numTrials,1);
incAriArray         = zeros(numTrials,1);
bwtAmiArray         = zeros(numTrials,1);
bwtAriArray         = zeros(numTrials,1);

parfor n = 1:numTrials

    % Set random seed
    rng(n);

    % Class order of this trial
    classOrder = randperm(numClasses);

    % Scores after learning each class
    amiSteps = zeros(1, numClasses);
    ariSteps = zeros(1, numClasses);

    net = IDAT();
    timeTrain = 0;
    evalData   = [];
    evalTarget = [];

    for cIdx = 1:numClasses
        c = classOrder(cIdx);

        % Train IDAT on the current class
        tic;
        net.fit(classData{c});
        timeTrain = timeTrain + toc;

        % Evaluate on all classes learned so far
        evalData   = [evalData;   classData{c}];
        evalTarget = [evalTarget; classTarget{c}];
        if cIdx == 1
            amiSteps(cIdx) = 1.0;
            ariSteps(cIdx) = 1.0;
        else
            predictedLabels = net.predict(evalData);
            [amiSteps(cIdx), ariSteps(cIdx)] = clusteringEvaluationMetrics(evalTarget, predictedLabels);
        end
    end

    % Number of all nodes, of active nodes, and of clusters
    % (clusters with at least one active node) after learning all classes
    numNodesArray(n)       = net.numNodes;
    numActiveNodesArray(n) = nnz(net.isWeight);
    numClustersArray(n)    = numel(unique(net.labelCluster(net.isWeight)));
    amiArray(n)            = amiSteps(end);
    ariArray(n)            = ariSteps(end);
    timeArray(n)           = timeTrain;

    % Average incremental AMI/ARI and backward transfer (BWT)
    incAmiArray(n) = mean(amiSteps);
    incAriArray(n) = mean(ariSteps);
    bwtAmiArray(n) = mean(amiSteps(end) - amiSteps(1:end-1));
    bwtAriArray(n) = mean(ariSteps(end) - ariSteps(1:end-1));
end

% Display average and standard deviation results
disp('=========== Nonstationary ============');
disp(['# classes: ', num2str(numClasses)]);
disp(['Average results over ', num2str(numTrials), ' runs']);
disp(['Average # all nodes:     ', num2str(mean(numNodesArray)),       ' ± ', num2str(std(numNodesArray))]);
disp(['Average # active nodes:  ', num2str(mean(numActiveNodesArray)), ' ± ', num2str(std(numActiveNodesArray))]);
disp(['Average # clusters:      ', num2str(mean(numClustersArray)),    ' ± ', num2str(std(numClustersArray))]);
disp(['         Average AMI:    ', num2str(mean(amiArray)),            ' ± ', num2str(std(amiArray))]);
disp(['         Average ARI:    ', num2str(mean(ariArray)),            ' ± ', num2str(std(ariArray))]);
disp(['        Average Time:    ', num2str(mean(timeArray)),           ' ± ', num2str(std(timeArray))]);
disp(' ');
disp(['Average Incremental AMI: ', num2str(mean(incAmiArray)), ' ± ', num2str(std(incAmiArray))]);
disp(['Average Incremental ARI: ', num2str(mean(incAriArray)), ' ± ', num2str(std(incAriArray))]);
disp(['  Backward Transfer AMI: ', num2str(mean(bwtAmiArray)), ' ± ', num2str(std(bwtAmiArray))]);
disp(['  Backward Transfer ARI: ', num2str(mean(bwtAriArray)), ' ± ', num2str(std(bwtAriArray))]);
disp(' ');

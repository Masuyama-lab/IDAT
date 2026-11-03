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

% Avoid label value == 0
if min(target)==0
    target = target + 1;
end

% Keep original data for parfor broadcast
data0 = data;
target0 = target;

% Initialize arrays to collect results
numNodesArray = zeros(numTrials,1);
numActiveNodesArray = zeros(numTrials,1);
numClustersArray = zeros(numTrials,1);
ariArray = zeros(numTrials,1);
amiArray = zeros(numTrials,1);
timeArray = zeros(numTrials,1);

parfor n = 1:numTrials

    % Set random seed
    rng(n);

    % Randomization
    ran = randperm(size(data0,1));
    data = data0(ran,:);
    target = target0(ran);

    % Train IDAT
    net = IDAT();
    tic;
    net.fit(data);
    timeTrain = toc;

    % Test IDAT
    predictedLabels = net.predict(data);

    % Evaluation
    [ami, ari] = clusteringEvaluationMetrics(target, predictedLabels);

    % Number of all nodes, of active nodes, and of clusters
    % (clusters with at least one active node)
    numNodesArray(n) = net.numNodes;
    numActiveNodesArray(n) = nnz(net.isWeight);
    numClustersArray(n) = numel(unique(net.labelCluster(net.isWeight)));
    ariArray(n) = ari;
    amiArray(n) = ami;
    timeArray(n) = timeTrain;
end

% Display average and standard deviation results
disp('=========== Stationary ============');
disp(['# classes: ', num2str(numel(unique(target0)))]);
disp(['Average results over ', num2str(numTrials), ' runs']);
disp(['Average # all nodes:     ', num2str(mean(numNodesArray)),       ' ± ', num2str(std(numNodesArray))]);
disp(['Average # active nodes:  ', num2str(mean(numActiveNodesArray)), ' ± ', num2str(std(numActiveNodesArray))]);
disp(['Average # clusters:      ', num2str(mean(numClustersArray)),    ' ± ', num2str(std(numClustersArray))]);
disp(['         Average AMI:    ', num2str(mean(amiArray)),            ' ± ', num2str(std(amiArray))]);
disp(['         Average ARI:    ', num2str(mean(ariArray)),            ' ± ', num2str(std(ariArray))]);
disp(['        Average Time:    ', num2str(mean(timeArray)),           ' ± ', num2str(std(timeArray))]);
disp(' ');

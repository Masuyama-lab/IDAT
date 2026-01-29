% Copyright (c) 2025-2026 Naoki Masuyama
% SPDX-License-Identifier: MIT

clear all

% Load dataset ============
load Iris
% load OptDigits


rng(0)



% Get unique class labels
classLabels = unique(target);
numClasses  = length(classLabels);

% Cell arrays to store data for each class
classData   = cell(numClasses, 1);
classTarget = cell(numClasses, 1);
for c = 1:numClasses
    currentClass = classLabels(c);
    idx          = find(target == currentClass);
    idx          = idx(randperm(length(idx)));  % shuffle within class
    classData{c}   = data(idx, :);
    classTarget{c} = target(idx);
end

% Initialize arrays to collect results
numTrials            = 10;
numNodes_array     = zeros(numTrials,1);
numClusters_array  = zeros(numTrials,1);
ARI_array          = zeros(numTrials,1);
AMI_array          = zeros(numTrials,1);
Time_array         = zeros(numTrials,1);
lambda             = zeros(numTrials,1);
minSim             = zeros(numTrials,1);

% Prepare storage for incremental metrics
ami_steps_array    = cell(numTrials,1);
ari_steps_array    = cell(numTrials,1);

% Preallocate cell arrays to store net for each trial
hold_net = cell(numTrials, 1);


parfor n = 1:numTrials
    
    % per-step performance
    ami_steps = zeros(1, numClasses);
    ari_steps = zeros(1, numClasses);

    % Initialize net
    IDATnet = IDAT();

    % Set random seed
    rng(n);
   
    % Initialize timer
    time_idat_train = 0;

    randomClassOrder = randperm(numClasses);

    for cIdx = 1:numClasses
        % Current class being handled
        c        = randomClassOrder(cIdx);
        data_c   = classData{c};
        target_c = classTarget{c};
        
        % Train IDAT
        tic;
        IDATnet = IDATnet.fit(data_c);
        time_idat_train = time_idat_train + toc;

        % incremental evaluation
        if cIdx == 1
            ami_steps(cIdx) = 1.0;
            ari_steps(cIdx) = 1.0;

        else
            % build eval set for classes learned so far
            eval_data   = [];
            eval_target = [];
            for eIdx = 1:cIdx
                ctmp        = randomClassOrder(eIdx);
                eval_data   = [eval_data;   classData{ctmp}];
                eval_target = [eval_target; classTarget{ctmp}];
            end
            predicted_inc = IDATnet.predict(eval_data);
            % Evaluation (unified): returns AMI, ARI
            [AMI_val, ARI_val] = Clustering_Evaluation_Metrics(eval_target, predicted_inc);
            ami_steps(cIdx) = AMI_val;
            ari_steps(cIdx) = ARI_val;
        end
    end

    hold_net{n} = IDATnet; % Save trained net

    % Hold step metrics
    ami_steps_array{n} = ami_steps;
    ari_steps_array{n} = ari_steps;

    % Consider isWeight ========================
    isWeight_for_predict = IDATnet.isWeight;
    if ~isempty(isWeight_for_predict) && ~any(isWeight_for_predict)
        isWeight_for_predict(:) = true;
    end
    weightIndices = find(isWeight_for_predict); % Indices of isWeight == 1 nodes
    tmpWeight     = IDATnet.weight(weightIndices, :);
    tmpEdge       = IDATnet.edge(weightIndices, weightIndices); % Extract edges
    connection    = graph(tmpEdge ~= 0); % Create graph structure
    tmpClusters   = conncomp(connection); % Compute clusters
    % ==========================================
    numNodes    = size(tmpWeight, 1);
    numClusters = max(tmpClusters);

    lambda(n) = IDATnet.Lambda;
    minSim(n) = IDATnet.similarityTh;

    % Collect results
    numNodes_array(n)    = numNodes;
    numClusters_array(n) = numClusters;
    ARI_array(n)         = ari_steps(end);
    AMI_array(n)         = ami_steps(end);
    Time_array(n)        = time_idat_train;
end

% Calculate mean and standard deviation for net
mean_numNodes    = mean(numNodes_array);
std_numNodes     = std(numNodes_array);
mean_numClusters = mean(numClusters_array);
std_numClusters  = std(numClusters_array);
mean_ARI         = mean(ARI_array);
std_ARI          = std(ARI_array);
mean_AMI         = mean(AMI_array);
std_AMI          = std(AMI_array);
mean_Time        = mean(Time_array);
std_Time         = std(Time_array);

% Calculate Average Incremental AMI/ARI and Backward Transfer
incAMI_list = zeros(numTrials,1);
incARI_list = zeros(numTrials,1);
bwtAMI_list = zeros(numTrials,1);
bwtARI_list = zeros(numTrials,1);
for n = 1:numTrials
    per_ami = num2cell(ami_steps_array{n});
    per_ari = num2cell(ari_steps_array{n});

    [incAMI, bwtAMI] = computeContinualClusteringMetrics(per_ami);
    [incARI, bwtARI] = computeContinualClusteringMetrics(per_ari);

    incAMI_list(n) = incAMI;
    incARI_list(n) = incARI;
    bwtAMI_list(n) = bwtAMI;
    bwtARI_list(n) = bwtARI;
end
mean_incAMI = mean(incAMI_list);  std_incAMI = std(incAMI_list);
mean_incARI = mean(incARI_list);  std_incARI = std(incARI_list);
mean_bwtAMI = mean(bwtAMI_list);  std_bwtAMI = std(bwtAMI_list);
mean_bwtARI = mean(bwtARI_list);  std_bwtARI = std(bwtARI_list);

% Display mean and standard deviation for net
disp('=========== Nonstationary ============');
disp(['# classes: ', num2str(max(target))]);
disp(['Average results over ', num2str(numTrials), ' runs']);
disp(['Average # nodes: ', num2str(mean_numNodes), ' ± ', num2str(std_numNodes)]);
disp(['Average # clusters: ', num2str(mean_numClusters), ' ± ', num2str(std_numClusters)]);
disp(['     Average AMI: ', num2str(mean_AMI), ' ± ', num2str(std_AMI)]);
disp(['     Average ARI: ', num2str(mean_ARI), ' ± ', num2str(std_ARI)]);
disp(['    Average Time: ', num2str(mean_Time), ' ± ', num2str(std_Time)]);
disp(' ');
disp(['     Average Incremental AMI: ', num2str(mean_incAMI), ' ± ', num2str(std_incAMI)]);
disp(['     Average Incremental ARI: ', num2str(mean_incARI), ' ± ', num2str(std_incARI)]);
disp(['    Backward Transfer AMI:   ', num2str(mean_bwtAMI), ' ± ', num2str(std_bwtAMI)]);
disp(['    Backward Transfer ARI:   ', num2str(mean_bwtARI), ' ± ', num2str(std_bwtARI)]);

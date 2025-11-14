clear all

% Load dataset ============
load Zoo
load Iris
% load Wine
% load Seeds
% load Glass
% load Newthyroid
% load Pima
% load Yeast
% load ImageSegmentation
% load Phoneme
% load OptDigits
% load Shuttle

rng(0)

% Avoid label value == 0
if min(target)==0
    target = target + 1;
end

% Store original data
OriginalData = data;
OriginalTarget = target;

% Initialize arrays to collect results
numFold = 10;
numNodes_array = zeros(numFold,1);
numClusters_array = zeros(numFold,1);
ARI_array = zeros(numFold,1);
AMI_array = zeros(numFold,1);
NID_array = zeros(numFold,1);
NVI_array = zeros(numFold,1);
Time_array = zeros(numFold,1);
lambda = zeros(numFold,1);
minSim = zeros(numFold,1);

% Preallocate cell arrays to store net for each trial
hold_net = cell(numFold, 1);





% for n = 1:numTrials
parfor n = 1:numFold

    % Initialize net
    IDATnet = IDAT();

    % Set random seed
    rng(n);

    % Reset data
    data = OriginalData;
    target = OriginalTarget;

    % Randomization
    ran = randperm(size(data,1));
    data = data(ran,:);
    target = target(ran);

    % Train IDAT
    tic;
    IDATnet = IDATnet.fit(data);
    time_idat_train = toc;
    nets0{n} = IDATnet; % Store trained net

    % Test IDAT
    predicted_labels = IDATnet.predict(data);

    % Evaluation
    [AMI_val, ARI_val, NID_val, NVI_val] = Clustering_Evaluation_Metrics(target, predicted_labels);

    % Consider isWeight ========================
    isWeight_for_predict = IDATnet.isWeight;
    if ~isempty(isWeight_for_predict) && ~any(isWeight_for_predict)
        isWeight_for_predict(:) = true;
    end
    weightIndices = find(isWeight_for_predict); % Indices of isWeight == 1 nodes
    tmpWeight = IDATnet.weight(weightIndices, :);
    tmpEdge = IDATnet.edge(weightIndices, weightIndices); % Extract edges corresponding to nodes
    connection = graph(tmpEdge ~= 0); % Create graph structure
    tmpClusters = conncomp(connection); % Compute cluster labels
    % ==========================================
    numNodes1 = size(tmpWeight, 1);
    numClusters1 = max(tmpClusters);

    lambda(n) = IDATnet.Lambda;
    minSim(n) = IDATnet.similarityTh;

    % Collect results
    numNodes_array(n) = numNodes1;
    numClusters_array(n) = numClusters1;
    ARI_array(n) = ARI_val;
    AMI_array(n) = AMI_val;

    %  For intuitive use, define NID' = 1.0 - NID
    NID_array(n) = 1.0 - NID_val;

    % For intuitive use, define NVI' = 1.0 - NVI
    NVI_array(n) = 1.0 - NVI_val;

    Time_array(n) = time_idat_train;

end

% Compute average and standard deviation results for net
mean_numNodes = mean(numNodes_array);
std_numNodes = std(numNodes_array);
mean_numClusters = mean(numClusters_array);
std_numClusters = std(numClusters_array);
mean_ARI = mean(ARI_array);
std_ARI = std(ARI_array);
mean_AMI = mean(AMI_array);
std_AMI = std(AMI_array);
mean_NID = mean(NID_array);
std_NID  = std(NID_array);
mean_NVI = mean(NVI_array);
std_NVI  = std(NVI_array);
mean_Time = mean(Time_array);
std_Time = std(Time_array);

% Display average and standard deviation results for net
disp('=========== Stationary ============');
disp(['# classes: ', num2str(max(target))]);
disp(['Average results over ', num2str(numFold), ' runs']);
disp(['Average # nodes:     ', num2str(mean_numNodes),    ' ± ', num2str(std_numNodes)]);
disp(['Average # clusters:  ', num2str(mean_numClusters),' ± ', num2str(std_numClusters)]);
disp(['     Average AMI:    ', num2str(mean_AMI),         ' ± ', num2str(std_AMI)]);
disp(['     Average ARI:    ', num2str(mean_ARI),         ' ± ', num2str(std_ARI)]);
disp(['     Average NID:    ', num2str(mean_NID),         ' ± ', num2str(std_NID)]);
disp(['     Average NVI:    ', num2str(mean_NVI),         ' ± ', num2str(std_NVI)]);
disp(['    Average Time:    ', num2str(mean_Time),        ' ± ', num2str(std_Time)]);
disp(' ');

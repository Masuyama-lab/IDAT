clear all

% Load dataset ============
load Iris
% load OptDigits

rng(0)

% Avoid label value == 0
if min(target)==0
    target = target + 1;
end

% Keep original data for parfor broadcast
data0 = data;
target0 = target;

% Initialize arrays to collect results
numTrials = 10;
numNodes_array = zeros(numTrials,1);
numClusters_array = zeros(numTrials,1);
ARI_array = zeros(numTrials,1);
AMI_array = zeros(numTrials,1);
Time_array = zeros(numTrials,1);



% for n = 1:numTrials
parfor n = 1:numTrials

    % Initialize net
    IDATnet = IDAT();

    % Set random seed
    rng(n);

    % Reset data
    data = data0;
    target = target0;

    % Randomization
    ran = randperm(size(data,1));
    data = data(ran,:);
    target = target(ran);

    % Train IDAT
    tic;
    IDATnet = IDATnet.fit(data);
    time_idat_train = toc;

    % Test IDAT
    predicted_labels = IDATnet.predict(data);

    % Evaluation
    [ami, ari] = Clustering_Evaluation_Metrics(target, predicted_labels);

    isWeight_for_predict = IDATnet.isWeight;
    if ~isempty(isWeight_for_predict) && ~any(isWeight_for_predict)
        isWeight_for_predict(:) = true;
    end
    weightIndices = find(isWeight_for_predict); % Indices of isWeight == 1 nodes
    tmpWeight = IDATnet.weight(weightIndices, :);
    tmpEdge = IDATnet.edge(weightIndices, weightIndices); % Extract edges corresponding to nodes
    connection = graph(tmpEdge ~= 0); % Create graph structure
    tmpClusters = conncomp(connection); % Compute cluster labels
    numNodes = size(tmpWeight, 1);
    numClusters = max(tmpClusters);


    % Collect results
    numNodes_array(n) = numNodes;
    numClusters_array(n) = numClusters;
    ARI_array(n) = ari;
    AMI_array(n) = ami;
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
mean_Time = mean(Time_array);
std_Time = std(Time_array);

% Display average and standard deviation results for net
disp('=========== Stationary ============');
disp(['# classes: ', num2str(max(target0))]);
disp(['Average results over ', num2str(numTrials), ' runs']);
disp(['Average # nodes:     ', num2str(mean_numNodes),    ' ± ', num2str(std_numNodes)]);
disp(['Average # clusters:  ', num2str(mean_numClusters),' ± ', num2str(std_numClusters)]);
disp(['     Average AMI:    ', num2str(mean_AMI),         ' ± ', num2str(std_AMI)]);
disp(['     Average ARI:    ', num2str(mean_ARI),         ' ± ', num2str(std_ARI)]);
disp(['    Average Time:    ', num2str(mean_Time),        ' ± ', num2str(std_Time)]);
disp(' ');

close all
clear all


set(gcf, 'Color', 'w'); % Figure全体の背景色を白に設定
axesHandle = gca;
set(axesHandle, 'Color', 'w'); % Axes（描画範囲）の背景色を白に設定

rng(1)

TRIAL = 10;    % Number of trials
 
NR = 0.1; % Noise Rate [0-1]


load 2D_ClusteringDATASET; numD = 15000; 
% load 2D_ClusteringDATASET_SOINN_numD10500; numD = 52500;
DATA = [data(1:end,1) data(1:end,2)];
originDATA = DATA;

DATA = [data(1:end,1) data(1:end,2)];

d1 = DATA(1:numD,:);
d2 = DATA(15001:15000+numD,:);
d3 = DATA(30001:30000+numD,:);
d4 = DATA(45001:45000+numD,:);
d5 = DATA(60001:60000+numD,:);
d6 = DATA(75001:75000+numD,:);
D = [d1; d2; d3; d4; d5; d6];
data = D;


% Normalization [0-1]
% DATA = normalize(DATA,'range');


% Initialization
net = IDAT();


time_idt_train = 0;

% Noise Setting [0,1]
if NR > 0
    noiseDATA = rand(size(data,1)*NR, size(data,2));
    data(1:size(noiseDATA,1),:) = noiseDATA;
end


for traial = 1:TRIAL
    
    fprintf('Iterations: %d/%d\n',traial,TRIAL);

    % Randamize data
    rng(traial)
    ran = randperm(size(data,1));
    data = data(ran,:);
    
    
    

    % IDAT ==============================================
    tic
    net = net.fit(data);
    time_idt_train = time_idt_train + toc;
    
    figure(1);
    myPlot_isWeight(data, net, 'Stationary IDAT');
    drawnow
    % ===================================================
    
    
    % Results
    % isWeightを考慮 ============================
    weightIndices = find(net.isWeight); % Indices of isWeight == 1 nodes
    numWeight = net.weight(weightIndices, :);
    filteredEdge = net.edge(weightIndices, weightIndices); % Extract edges corresponding to nodes
    connection = graph(filteredEdge ~= 0); % Create graph structure
    numClusters = conncomp(connection); % Compute cluster labels
    % ==========================================
    disp(['   Num. Nodes: ', num2str(size(numWeight, 1))]);
    disp(['Num. Clusters: ', num2str(max(numClusters))]);
    disp([' Processing Time:  IDAT: ', num2str(time_idt_train)]);
    disp('');
    
    
end




% Copyright (c) 2025-2026 Naoki Masuyama
% SPDX-License-Identifier: MIT

close all
clear all


set(gcf, 'Color', 'w');
axesHandle = gca;
set(axesHandle, 'Color', 'w');

rng(1)

TRIAL = 10;    % Number of trials
 
NR = 0.1; % Noise Rate [0-1]


load 2D_ClusteringDATASET;
data = [data(1:end,1) data(1:end,2)];

% Noise Setting [0,1]
if NR > 0
    noiseDATA = rand(size(data,1)*NR, size(data,2));
    data(1:size(noiseDATA,1),:) = noiseDATA;
end



% Initialization
net = IDAT();


time_idt_train = 0;

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
    weightIndices = find(net.isWeight); % Indices of isWeight == 1 nodes
    numWeight = net.weight(weightIndices, :);
    filteredEdge = net.edge(weightIndices, weightIndices); % Extract edges corresponding to nodes
    connection = graph(filteredEdge ~= 0); % Create graph structure
    numClusters = conncomp(connection); % Compute cluster labels
    disp(['   Num. Nodes: ', num2str(size(numWeight, 1))]);
    disp(['Num. Clusters: ', num2str(max(numClusters))]);
    disp([' Processing Time:  IDAT: ', num2str(time_idt_train)]);
    disp('');
    
    
end




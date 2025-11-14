close all
clear all

set(gcf, 'Color', 'w'); % Figure全体の背景色を白に設定
axesHandle = gca;
set(axesHandle, 'Color', 'w'); % Axes（描画範囲）の背景色を白に設定

rng(1)

TRIAL = 10;    % Number of trials

NR = 0.1; % Noise Rate [0-1]


load 2Ddataset_6C_90000_Stream;

% Initialization
net = IDAT();


time_idt_train = 0;

for traial = 1:(TRIAL * size(data,2))

    fprintf('Iterations: %d/%d\n',traial,TRIAL*size(data,2));
    
    % for 2Ddataset_6C_90000_Stream ---------
    idx = mod(traial-1, size(data,2)) + 1; % Cycle through 1..6 based on remainder
    data_train = data{idx};
    % ---------------------------------------

    % Randamize data
    ran = randperm(size(data_train,1));
    data_train = data_train(ran,:);
    
    % Noise Setting [0,1]
    if NR > 0
        noiseDATA = rand(size(data_train,1)*NR, size(data_train,2));
        data_train(1:size(noiseDATA,1),:) = noiseDATA;
    end

    % IDAT ==============================================
    tic
    net = net.fit(data_train);
    time_idt_train = time_idt_train + toc;

    figure(1);
    myPlot_isWeight(data_train, net, 'Non-Stationary IDAT');
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

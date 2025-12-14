function myPlot_isWeight(DATA, net, num)
    cla;
    
    w = net.weight;
    edge = net.edge;
    % edgeAge = net.edgeAge; % edgeAgeを利用
    CountNode = net.CountNode;
    isWeight = net.isWeight; % isCandidateを取得
    [N, D] = size(w);

    connection = graph(net.edge ~= 0);
    label = conncomp(connection);

    % Prepare colors and handle grid setup outside the loop for efficiency
    color = [
        [1 0 0];        % Red
        [0 1 0];        % Green
        [0 0 1];        % Blue
        [1 0 1];        % Magenta
        [0.8500 0.3250 0.0980];  % Orange
        [0.9290 0.6940 0.1250];  % Yellow
        [0.4940 0.1840 0.5560];  % Purple
        [0.4660 0.6740 0.1880];  % Olive
        [0.3010 0.7450 0.9330];  % Light Blue
        [0.6350 0.0780 0.1840];  % Dark Red
        [0.3 0.3 0.3];  % Gray
        [0.5 0.5 0];    % Olive Green
        [0.5 0 0.5];    % Dark Magenta
        [0 0.5 0.5];    % Dark Cyan
        [0.75 0.75 0];  % Mustard
        [0.25 0.25 0.25];  % Dark Gray
        [0 0.4470 0.7410]; % MATLAB Blue
        [0.8500 0.3250 0.0980]; % MATLAB Orange
        [0.9290 0.6940 0.1250]; % MATLAB Yellow
        [0.4940 0.1840 0.5560]; % MATLAB Purple
        [0.1 0.6 0.4];   % Teal
        [0.9 0.3 0.5];   % Pink
        [0.2 0.2 0.6];   % Navy Blue
        [0.6 0.2 0.2];   % Brown
        [0.4 0.4 0.6];   % Lavender
        [0.6 0.6 0.2];   % Light Olive
        [0.2 0.6 0.2];   % Lime Green
        [0.7 0.4 0.1];   % Bronze
        [0.1 0.4 0.7];   % Sky Blue
        [0.9 0.6 0.2];   % Apricot
        [0.7 0.1 0.7];   % Violet
    ];
    m = length(color);

    % Filter nodes with isCandidate = 0
    nonCandidateIndices = find(isWeight == 1); % isCandidate = 0 のインデックスを取得
    w_nonCandidate = w(nonCandidateIndices, :); % ノードの座標を取得
    CountNode_nonCandidate = CountNode(nonCandidateIndices); % CountNode もフィルタリング
    label_nonCandidate = label(nonCandidateIndices); % クラスタラベルもフィルタリング

    % Map labels to colors in advance for non-candidate nodes
    colors_nonCandidate = color(mod(label_nonCandidate - 1, m) + 1, :);

    hold on;

    % データポイント描画
    plot(DATA(:,1), DATA(:,2), 'o', 'MarkerEdgeColor', [0.8, 0.8, 0.8], 'MarkerFaceColor', [0.7, 0.7, 0.7], 'MarkerSize', 2);

    % エッジとそのAgeの描画用インデックス取得
    [i_idx, j_idx] = find(triu(edge, 1)); % 上三角行列を使って冗長な計算を削除
    ages = edge(sub2ind(size(edge), i_idx, j_idx)); % 対応するエッジAgeを取得

    if D == 2
        X = [w(i_idx,1) w(j_idx,1)]';
        Y = [w(i_idx,2) w(j_idx,2)]';
        plot(X, Y, 'k', 'LineWidth', 2);
        
        % エッジの中点にAgeを表示
        % midX = mean(X,1); % 各エッジごとの中点X座標
        % midY = mean(Y,1); % 各エッジごとの中点Y座標
        % for k = 1:length(ages)
        %     text(midX(k), midY(k), num2str(ages(k)), ...
        %          'Color', 'r', 'FontSize', 12, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
        % end
    elseif D == 3
        X = [w(i_idx,1) w(j_idx,1)]';
        Y = [w(i_idx,2) w(j_idx,2)]';
        Z = [w(i_idx,3) w(j_idx,3)]';
        plot3(X, Y, Z, 'w', 'LineWidth', 2);
        
        % エッジの中点にAgeを表示 (3D)
        % midX = mean(X,1); % 各エッジごとの中点X座標
        % midY = mean(Y,1); % 各エッジごとの中点Y座標
        % midZ = mean(Z,1); % 各エッジごとの中点Z座標
        % for k = 1:length(ages)
        %     text(midX(k), midY(k), midZ(k), num2str(ages(k)), ...
        %          'Color', 'k', 'FontSize', 8, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
        % end
    end

    % ノードを一括で描画 (isCandidate = 0 のノードのみ)
    if D == 2
        scatter(w_nonCandidate(:,1), w_nonCandidate(:,2), 80, colors_nonCandidate, 'filled');
        
        % % 各ノードの右上にCountNodeを表示
        % for k = 1:length(nonCandidateIndices)
        %     text(w_nonCandidate(k,1) + 0.005, w_nonCandidate(k,2) + 0.005, num2str(CountNode_nonCandidate(k)), ...
        %          'Color', 'k', 'FontSize', 10, 'HorizontalAlignment', 'left', ...
        %          'VerticalAlignment', 'bottom');
        % end
    elseif D == 3
        scatter3(w_nonCandidate(:,1), w_nonCandidate(:,2), w_nonCandidate(:,3), 280, colors_nonCandidate, 'filled');
        
        % 各ノードの右上にCountNodeを表示 (3D)
        % for k = 1:length(nonCandidateIndices)
        %     text(w_nonCandidate(k,1) + 0.005, w_nonCandidate(k,2) + 0.005, w_nonCandidate(k,3) + 0.02, ...
        %          num2str(CountNode_nonCandidate(k)), ...
        %          'Color', 'k', 'FontSize', 10, 'HorizontalAlignment', 'left', ...
        %          'VerticalAlignment', 'bottom');
        % end
    end

    % 軸とグリッドの設定
    ytickformat('%.1f');
    xtickformat('%.1f');
    xticks(0:0.2:1);
    yticks(0:0.2:1);
    set(gca, 'GridColor', 'k');
    set(gca, 'layer', 'bottom');
    set(gca, 'FontSize', 25); 
    set(gca, 'XMinorTick', 'on', 'YMinorTick', 'on');

    axis equal
    grid on
    box on
    hold off
    axis([-0.05 1.05 -0.05 1.05]);
end

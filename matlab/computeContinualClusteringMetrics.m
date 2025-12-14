function [avgInc, bwt] = computeContinualClusteringMetrics(per_metric)
% computeContinualClusteringMetrics  Compute average incremental & backward transfer
%   [avgInc, bwt] = computeContinualClusteringMetrics(per_metric)
%
%   INPUT:
%     per_metric : 1×C cell array of phase-wise scores for a given metric
%
%   OUTPUT:
%     avgInc - scalar, average incremental metric
%     bwt    - scalar, backward transfer metric

    % Convert input cell array to numeric vector
    scores = cell2mat(per_metric);

    % Number of phases
    C = numel(scores);

    % --- Average Incremental computation ---
    avgInc = mean(scores);

    % --- Backward Transfer computation ---
    if C > 1
        final_score = scores(end);
        deltas = final_score - scores(1:C-1);
        bwt = mean(deltas);
    else
        bwt = 0;  % no previous phase to compare with
    end
end

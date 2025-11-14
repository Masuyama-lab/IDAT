function [AMI, ARI, NVI] = Clustering_Evaluation_Metrics(ground_truth, predicted)
    % 
    % https://jp.mathworks.com/help/matlab/matlab_external/install-supported-python-implementation.html
    % 

    % This function calculates AMI and ARI by calling Python's sklearn functions

    % Convert MATLAB arrays to Python-compatible 1-D lists
    % If truth or predicted is a row vector or column vector,
    % we ensure they are 1-D by using (:)
    truth = double(ground_truth(:));
    predicted = double(predicted(:));

    % Convert them to Python array
    py_truth = py.numpy.array(truth);
    py_pred = py.numpy.array(predicted);

    % Flatten to true 1-D array
    py_truth = py_truth.flatten();            % shape: (N,)
    py_pred  = py_pred.flatten();             % shape: (N,)

    % Calculate AMI（1.0 is the best, 0.0 is the worst）
    amic = py.sklearn.metrics.adjusted_mutual_info_score(py_truth, py_pred);
    AMI = double(amic);

    % Calculate ARI（1.0 is the best, -0.5 is the worst）
    aric = py.sklearn.metrics.adjusted_rand_score(py_truth, py_pred);
    ARI = double(aric);

    % Compute Normalized Variation of Information（0.0 is the best, 1.0 is the worst）
    NVI = NormalizedVariationInformation( ground_truth, predicted );

end


% NVI: Normalized Variation of Information
% Computes the normalized variation of information between two label vectors
function nvi = NormalizedVariationInformation(c1, c2)
    % Input validation
    if numel(c1) ~= numel(c2)
        error('c1 and c2 must have the same length.');
    end

    % Map labels to consecutive integers starting at 1
    [~, ~, ic1] = unique(c1);
    [~, ~, ic2] = unique(c2);

    N = numel(ic1);
    K1 = max(ic1);
    K2 = max(ic2);

    % Build contingency table
    nij = accumarray([ic1, ic2], 1, [K1, K2]);
    ni = sum(nij, 2);
    nj = sum(nij, 1);

    % Convert to probabilities
    p_ij = nij / N;
    p_i  = ni  / N;
    p_j  = nj  / N;

    % Compute entropies (ignore zero probabilities)
    Huv = -sum(p_ij(p_ij>0) .* log(p_ij(p_ij>0)));
    Hu  = -sum(p_i(p_i>0)   .* log(p_i(p_i>0)));
    Hv  = -sum(p_j(p_j>0)   .* log(p_j(p_j>0)));

    % Mutual information: MI = Hu + Hv - Huv
    MI = Hu + Hv - Huv;

    % Normalized Variation of Information: 1 - MI / Huv
    nvi = 1 - MI / Huv;
end



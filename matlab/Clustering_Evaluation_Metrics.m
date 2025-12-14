function [AMI, ARI] = Clustering_Evaluation_Metrics(ground_truth, predicted)

% Compute Adjusted Mutual Information
AMI = AdjustedMutualInformation(ground_truth, predicted);

% Compute Adjusted Rand Index
ARI = AdjustedRandIndex(ground_truth, predicted);

end


function AMI = AdjustedMutualInformation(gt, pred)
% Compute Adjusted Mutual Information (AMI) aligned with scikit-learn default
% average_method = 'arithmetic' and natural logarithm.

gt = gt(:);
pred = pred(:);

[~, ~, gt] = unique(gt);
[~, ~, pred] = unique(pred);

n = length(gt);
if n == 0
    AMI = NaN;
    return;
end

classes_gt = unique(gt);
classes_pred = unique(pred);

n_gt = length(classes_gt);
n_pred = length(classes_pred);

contingency = zeros(n_gt, n_pred);
for i = 1:n_gt
    gi = (gt == classes_gt(i));
    for j = 1:n_pred
        contingency(i, j) = sum(gi & (pred == classes_pred(j)));
    end
end

a = sum(contingency, 2);
b = sum(contingency, 1);

% Mutual Information (natural log)
MI = 0;
for i = 1:n_gt
    for j = 1:n_pred
        nij = contingency(i, j);
        if nij > 0
            MI = MI + (nij / n) * log((n * nij) / (a(i) * b(j)));
        end
    end
end

% Entropies (natural log)
pa = a / n;
pb = b / n;

H_gt = -sum(pa(pa > 0) .* log(pa(pa > 0)));
H_pred = -sum(pb(pb > 0) .* log(pb(pb > 0)));

% Expected Mutual Information under hypergeometric model
EMI = ExpectedMutualInformation(a, b, n);

% scikit-learn default: average_method = 'arithmetic'
normalizer = 0.5 * (H_gt + H_pred);

if abs(normalizer - EMI) < 1.0e-15
    % Degenerate case: both clusterings have zero entropy
    if abs(MI - EMI) < 1.0e-15
        AMI = 1.0;
    else
        AMI = 0.0;
    end
    return;
end

AMI = (MI - EMI) / (normalizer - EMI);

% Numerical safety clamp
if AMI > 1
    AMI = 1;
end
if AMI < -1
    AMI = -1;
end

end


function EMI = ExpectedMutualInformation(a, b, n)
% Compute expected mutual information E[MI] for contingency marginals a, b.

a = a(:);
b = b(:);

R = length(a);
C = length(b);

EMI = 0;

logC_n_b = lognchoosek(n, b); % 1 x C

for i = 1:R
    ai = a(i);

    for j = 1:C
        bj = b(j);

        nij_min = max(0, ai + bj - n);
        nij_max = min(ai, bj);

        if nij_min > nij_max
            continue;
        end

        for nij = nij_min:nij_max
            if nij == 0
                continue;
            end

            % Hypergeometric probability:
            % P(nij) = C(ai, nij) * C(n-ai, bj-nij) / C(n, bj)
            logP = lognchoosek(ai, nij) + lognchoosek(n - ai, bj - nij) - logC_n_b(j);
            P = exp(logP);

            EMI = EMI + (nij / n) * log((n * nij) / (ai * bj)) * P;
        end
    end
end

end


function ARI = AdjustedRandIndex(gt, pred)
% Compute Adjusted Rand Index (ARI)

gt = gt(:);
pred = pred(:);

[~, ~, gt] = unique(gt);
[~, ~, pred] = unique(pred);

n = length(gt);
if n < 2
    ARI = 1.0;
    return;
end

classes_gt = unique(gt);
classes_pred = unique(pred);

n_gt = length(classes_gt);
n_pred = length(classes_pred);

contingency = zeros(n_gt, n_pred);
for i = 1:n_gt
    gi = (gt == classes_gt(i));
    for j = 1:n_pred
        contingency(i, j) = sum(gi & (pred == classes_pred(j)));
    end
end

sum_comb = sum(nchoosek2(contingency(:)));
sum_comb_c = sum(nchoosek2(sum(contingency, 2)));
sum_comb_k = sum(nchoosek2(sum(contingency, 1)));

total_comb = nchoosek(n, 2);

expected_index = (sum_comb_c * sum_comb_k) / total_comb;
max_index = 0.5 * (sum_comb_c + sum_comb_k);

den = (max_index - expected_index);
if abs(den) < 1.0e-15
    if abs(sum_comb - expected_index) < 1.0e-15
        ARI = 1.0;
    else
        ARI = 0.0;
    end
else
    ARI = (sum_comb - expected_index) / den;
end

end


function y = nchoosek2(x)
% Compute nC2 element-wise without calling nchoosek repeatedly.

x = double(x);
y = zeros(size(x));
mask = (x >= 2);
y(mask) = x(mask) .* (x(mask) - 1) / 2;

end


function lc = lognchoosek(n, k)
% Compute log(n choose k) safely using gammaln.
% n can be a scalar. k can be a scalar or a vector.

k = double(k);
lc = -Inf(size(k));

valid = (k >= 0) & (k <= n);
if any(valid(:))
    kv = k(valid);
    lc(valid) = gammaln(n + 1) - gammaln(kv + 1) - gammaln(n - kv + 1);
end

end

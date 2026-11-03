% Copyright (c) 2025-2026 Naoki Masuyama
% SPDX-License-Identifier: MIT

function [AMI, ARI] = clusteringEvaluationMetrics(groundTruth, predicted)

% Compute Adjusted Mutual Information
AMI = adjustedMutualInformation(groundTruth, predicted);

% Compute Adjusted Rand Index
ARI = adjustedRandIndex(groundTruth, predicted);

end


function AMI = adjustedMutualInformation(gt, pred)
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

classesGt = unique(gt);
classesPred = unique(pred);

nGt = length(classesGt);
nPred = length(classesPred);

contingency = zeros(nGt, nPred);
for i = 1:nGt
    gi = (gt == classesGt(i));
    for j = 1:nPred
        contingency(i, j) = sum(gi & (pred == classesPred(j)));
    end
end

a = sum(contingency, 2);
b = sum(contingency, 1);

% Mutual Information (natural log)
MI = 0;
for i = 1:nGt
    for j = 1:nPred
        nij = contingency(i, j);
        if nij > 0
            MI = MI + (nij / n) * log((n * nij) / (a(i) * b(j)));
        end
    end
end

% Entropies (natural log)
pa = a / n;
pb = b / n;

entropyGt = -sum(pa(pa > 0) .* log(pa(pa > 0)));
entropyPred = -sum(pb(pb > 0) .* log(pb(pb > 0)));

% Expected Mutual Information under hypergeometric model
EMI = expectedMutualInformation(a, b, n);

% scikit-learn default: average_method = 'arithmetic'
normalizer = 0.5 * (entropyGt + entropyPred);

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


function EMI = expectedMutualInformation(a, b, n)
% Compute expected mutual information E[MI] for contingency marginals a, b.

a = a(:);
b = b(:);

R = length(a);
C = length(b);

EMI = 0;

logNChooseB = lognchoosek(n, b); % 1 x C

for i = 1:R
    ai = a(i);

    for j = 1:C
        bj = b(j);

        nijMin = max(0, ai + bj - n);
        nijMax = min(ai, bj);

        if nijMin > nijMax
            continue;
        end

        for nij = nijMin:nijMax
            if nij == 0
                continue;
            end

            % Hypergeometric probability:
            % P(nij) = C(ai, nij) * C(n-ai, bj-nij) / C(n, bj)
            logP = lognchoosek(ai, nij) + lognchoosek(n - ai, bj - nij) - logNChooseB(j);
            P = exp(logP);

            EMI = EMI + (nij / n) * log((n * nij) / (ai * bj)) * P;
        end
    end
end

end


function ARI = adjustedRandIndex(gt, pred)
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

classesGt = unique(gt);
classesPred = unique(pred);

nGt = length(classesGt);
nPred = length(classesPred);

contingency = zeros(nGt, nPred);
for i = 1:nGt
    gi = (gt == classesGt(i));
    for j = 1:nPred
        contingency(i, j) = sum(gi & (pred == classesPred(j)));
    end
end

sumComb = sum(nchoosek2(contingency(:)));
sumCombC = sum(nchoosek2(sum(contingency, 2)));
sumCombK = sum(nchoosek2(sum(contingency, 1)));

totalComb = nchoosek(n, 2);

expectedIndex = (sumCombC * sumCombK) / totalComb;
maxIndex = 0.5 * (sumCombC + sumCombK);

den = (maxIndex - expectedIndex);
if abs(den) < 1.0e-15
    if abs(sumComb - expectedIndex) < 1.0e-15
        ARI = 1.0;
    else
        ARI = 0.0;
    end
else
    ARI = (sumComb - expectedIndex) / den;
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

// Copyright (c) 2025-2026 Naoki Masuyama
// SPDX-License-Identifier: MIT

// Adjusted Mutual Information (AMI) and Adjusted Rand Index (ARI) of a predicted clustering.
//
//     auto [ami, ari] = clustering_evaluation_metrics(ground_truth, predicted);
//
// AMI follows the default of scikit-learn: the arithmetic mean of the two entropies as the
// normalizer and the natural logarithm.

#ifndef CLUSTERING_EVALUATION_METRICS_HPP
#define CLUSTERING_EVALUATION_METRICS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

// Contingency table of two labelings: entry (i, j) is the number of samples with the i-th
// smallest label of gt and the j-th smallest label of pred.
inline std::vector<std::vector<double>> contingency_table(const std::vector<int>& gt,
                                                          const std::vector<int>& pred) {
    std::vector<int> classes_gt(gt);
    std::sort(classes_gt.begin(), classes_gt.end());
    classes_gt.erase(std::unique(classes_gt.begin(), classes_gt.end()), classes_gt.end());
    std::vector<int> classes_pred(pred);
    std::sort(classes_pred.begin(), classes_pred.end());
    classes_pred.erase(std::unique(classes_pred.begin(), classes_pred.end()), classes_pred.end());

    std::vector<std::vector<double>> contingency(classes_gt.size(),
                                                 std::vector<double>(classes_pred.size(), 0.0));
    for (std::size_t n = 0; n < gt.size(); ++n) {
        const std::size_t i =
            std::lower_bound(classes_gt.begin(), classes_gt.end(), gt[n]) - classes_gt.begin();
        const std::size_t j = std::lower_bound(classes_pred.begin(), classes_pred.end(), pred[n]) -
                              classes_pred.begin();
        contingency[i][j] += 1;
    }
    return contingency;
}

// Expected mutual information E[MI] for the row sums a and the column sums b of a contingency
// table of n samples (hypergeometric model).
inline double expected_mutual_information(const std::vector<double>& a,
                                          const std::vector<double>& b, std::size_t n) {
    // log(k!) for k = 0..n
    std::vector<double> log_factorial(n + 1, 0.0);
    for (std::size_t k = 1; k <= n; ++k) {
        log_factorial[k] = log_factorial[k - 1] + std::log(static_cast<double>(k));
    }
    auto log_n_choose_k = [&log_factorial](std::size_t N, std::size_t k) {
        return log_factorial[N] - log_factorial[k] - log_factorial[N - k];
    };

    const double N = static_cast<double>(n);
    double emi = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const std::size_t ai = static_cast<std::size_t>(a[i]);
        for (std::size_t j = 0; j < b.size(); ++j) {
            const std::size_t bj = static_cast<std::size_t>(b[j]);
            const std::size_t nij_min = ai + bj > n ? ai + bj - n : 0;
            const std::size_t nij_max = std::min(ai, bj);
            const double log_n_choose_bj = log_n_choose_k(n, bj);
            for (std::size_t nij = std::max<std::size_t>(nij_min, 1); nij <= nij_max; ++nij) {
                // P(nij) = C(ai, nij) * C(n - ai, bj - nij) / C(n, bj)
                const double log_p =
                    log_n_choose_k(ai, nij) + log_n_choose_k(n - ai, bj - nij) - log_n_choose_bj;
                const double p = std::exp(log_p);
                const double x = static_cast<double>(nij);
                emi += (x / N) * std::log((N * x) / (a[i] * b[j])) * p;
            }
        }
    }
    return emi;
}

inline double adjusted_mutual_information(const std::vector<int>& gt,
                                          const std::vector<int>& pred) {
    const std::size_t n = gt.size();
    if (n == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const std::vector<std::vector<double>> contingency = contingency_table(gt, pred);
    const std::size_t num_gt = contingency.size();
    const std::size_t num_pred = contingency[0].size();
    std::vector<double> a(num_gt, 0.0);
    std::vector<double> b(num_pred, 0.0);
    for (std::size_t i = 0; i < num_gt; ++i) {
        for (std::size_t j = 0; j < num_pred; ++j) {
            a[i] += contingency[i][j];
            b[j] += contingency[i][j];
        }
    }
    const double N = static_cast<double>(n);

    // mutual information (natural logarithm)
    double mi = 0.0;
    for (std::size_t i = 0; i < num_gt; ++i) {
        for (std::size_t j = 0; j < num_pred; ++j) {
            const double nij = contingency[i][j];
            if (nij > 0) {
                mi += (nij / N) * std::log((N * nij) / (a[i] * b[j]));
            }
        }
    }

    // entropies (natural logarithm)
    double entropy_gt = 0.0;
    for (double ai : a) {
        entropy_gt -= (ai / N) * std::log(ai / N);
    }
    double entropy_pred = 0.0;
    for (double bj : b) {
        entropy_pred -= (bj / N) * std::log(bj / N);
    }

    const double emi = expected_mutual_information(a, b, n);

    // arithmetic mean of the two entropies
    const double normalizer = 0.5 * (entropy_gt + entropy_pred);
    if (std::abs(normalizer - emi) < 1.0e-15) {
        // both clusterings have zero entropy
        return std::abs(mi - emi) < 1.0e-15 ? 1.0 : 0.0;
    }
    const double ami = (mi - emi) / (normalizer - emi);
    return std::min(std::max(ami, -1.0), 1.0);
}

inline double adjusted_rand_index(const std::vector<int>& gt, const std::vector<int>& pred) {
    const std::size_t n = gt.size();
    if (n < 2) {
        return 1.0;
    }
    const std::vector<std::vector<double>> contingency = contingency_table(gt, pred);
    const std::size_t num_gt = contingency.size();
    const std::size_t num_pred = contingency[0].size();
    auto choose2 = [](double x) { return x * (x - 1) / 2; };

    double sum_comb = 0.0;
    std::vector<double> a(num_gt, 0.0);
    std::vector<double> b(num_pred, 0.0);
    for (std::size_t i = 0; i < num_gt; ++i) {
        for (std::size_t j = 0; j < num_pred; ++j) {
            sum_comb += choose2(contingency[i][j]);
            a[i] += contingency[i][j];
            b[j] += contingency[i][j];
        }
    }
    double sum_comb_c = 0.0;
    for (double ai : a) {
        sum_comb_c += choose2(ai);
    }
    double sum_comb_k = 0.0;
    for (double bj : b) {
        sum_comb_k += choose2(bj);
    }
    const double total_comb = choose2(static_cast<double>(n));

    const double expected_index = (sum_comb_c * sum_comb_k) / total_comb;
    const double max_index = 0.5 * (sum_comb_c + sum_comb_k);
    const double den = max_index - expected_index;
    if (std::abs(den) < 1.0e-15) {
        return std::abs(sum_comb - expected_index) < 1.0e-15 ? 1.0 : 0.0;
    }
    return (sum_comb - expected_index) / den;
}

// (AMI, ARI) of the predicted labels against the ground truth labels.
inline std::pair<double, double> clustering_evaluation_metrics(const std::vector<int>& ground_truth,
                                                               const std::vector<int>& predicted) {
    return {adjusted_mutual_information(ground_truth, predicted),
            adjusted_rand_index(ground_truth, predicted)};
}

#endif  // CLUSTERING_EVALUATION_METRICS_HPP

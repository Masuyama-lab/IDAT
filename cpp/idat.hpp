// Copyright (c) 2025-2026 Naoki Masuyama
// SPDX-License-Identifier: MIT

// IDAT: Inverse Distance-based ART with Topology.
//
//     IDAT net;                                  // a new model, with no hyperparameter to set
//     net.fit(X);                                // learn the rows of X one at a time, in order
//     std::vector<int> labels = net.predict(Y);  // cluster label of every row of Y
//
// X and Y are std::vector<std::vector<double>>, one inner vector per sample.
// fit can be called again with new samples. The model keeps learning from where it stopped, so a
// data stream can be given chunk by chunk. fit_predict(X) is fit(X) followed by predict(X).
//
// How it works (one sample x at a time)
//   Category choice   Every node k has a prototype w_k and a scale sigma_k. The similarity of x to
//                     node k is s_k = 1 / (1 + ||x - w_k|| / sigma_k). The two most similar nodes
//                     are s1 and s2. The first three samples become nodes.
//   Vigilance test    V is the similarity threshold (the vigilance parameter).
//                       s_1 <  V           : x becomes a new node.
//                       s_1 >= V, s_2 <= V : w_s1 moves towards x.
//                       s_1 >= V, s_2 >  V : w_s1 and w_s2 move towards x. When both nodes have
//                     won more often than the average of the nodes that have won at least twice,
//                     both become active (is_weight()) and their co-activation count
//                     E^cand(s1, s2) grows. An edge is created once that count exceeds the average
//                     of the positive co-activation counts.
//   Maintenance       Every Lambda samples: edges longer than Q3 + 1.5 IQR of the edge lengths are
//                     removed, nodes that have never won since their creation are removed except
//                     the newest Lambda of them, and an active node that is isolated and has
//                     stayed idle for two maintenances becomes inactive.
//   Adaptation        Every Lambda samples, once there are more than two nodes: Lambda is set to
//                     the largest number L of recent samples whose similarity matrix S_L is still
//                     diverse, det(S_L) >= EPS_DET, and V is set to a quantile of the similarities
//                     within these samples. The quantile level follows the ratio of the number of
//                     clusters to the number of nodes, smoothed over the adaptations.
//   Prediction        A sample gets the label of its most similar active node (of all nodes if
//                     none is active). The labels are the connected components of the edge graph,
//                     numbered from 1 in the order of their smallest node index.
//
// Accessors (read only)
//   num_nodes()              K, the number of nodes, active or not
//   weight()                 [K x d] prototypes
//   count_node()             [K] number of wins of every node
//   buffer_std()             [K] sigma_k
//   is_weight()              [K] true for the active nodes, the nodes used by predict
//   edges()                  the edges, as pairs (i, j) of node indices with i < j
//   lambda()                 current window length and maintenance interval
//   similarity_threshold()   current similarity threshold V
//   label_cluster()          [K] cluster label of every node, set by predict
//   sample_counts()          number of samples learned so far
//
// Network size (the three counts printed by the main programs)
//   all nodes      num_nodes()
//   active nodes   the number of true entries of is_weight()
//   clusters       the number of distinct values of label_cluster() over the active nodes after
//                  predict, that is, the clusters that contain at least one active node
//
// The class constants are the values used in the paper. They are not meant to be tuned.
//
// Requires a C++17 compiler and no other library.
//
// Reference
//   N. Masuyama, Y. Toda, Y. Nojima, and H. Ishibuchi, "An adaptive resonance theory-based
//   topological clustering algorithm with a self-adjusting vigilance parameter,"
//   arXiv:2511.17983, 2025. https://arxiv.org/abs/2511.17983

#ifndef IDAT_HPP
#define IDAT_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

class IDAT {
public:
    // initial Lambda
    static constexpr std::size_t LAMBDA_INIT = 2;
    // a window of L samples is diverse while det(S_L) >= EPS_DET
    static constexpr double EPS_DET = 1.0e-6;
    // c of the bounds Q3 + c IQR (edge pruning) and Q1 - c IQR (demotion)
    static constexpr double IQR_FACTOR = 1.5;
    // idle maintenances before an isolated active node is demoted
    static constexpr int DEMOTE_TOLERANCE = 2;
    // the buffer keeps the newest BUFFER_FACTOR * Lambda samples
    static constexpr std::size_t BUFFER_FACTOR = 2;
    // lower bound of sigma_k
    static constexpr double SIGMA_MIN = 1.0e-6;

    // ---------------------------------------------------------------------------------------------
    // Learning and prediction
    // ---------------------------------------------------------------------------------------------

    // Learn the rows of X one at a time, in the given order.
    IDAT& fit(const std::vector<std::vector<double>>& X) {
        for (const std::vector<double>& x : X) {
            if (sample_counts_ == 0) {
                d_ = x.size();
            } else if (x.size() != d_) {
                throw std::invalid_argument("IDAT: all samples must have the same length");
            }
            append_to_buffer(x);
            learn_one_sample(x);
            sample_counts_ += 1;
            recalc_counter_ += 1;
            if (recalc_counter_ >= lambda_ && num_nodes_ > 2) {
                adapt_lambda_and_threshold();
                recalc_counter_ = 0;
                update_buffer_keep_len();
            }
        }
        return *this;
    }

    // Cluster label of every row of X.
    std::vector<int> predict(const std::vector<std::vector<double>>& X) {
        const std::size_t K = num_nodes_;
        if (K == 0) {
            label_cluster_.clear();
            return {};
        }
        std::vector<int> labels;
        connected_components(labels);
        label_cluster_ = labels;
        std::vector<std::size_t> nodes;
        for (std::size_t k = 0; k < K; ++k) {
            if (active_[k]) {
                nodes.push_back(k);
            }
        }
        if (nodes.empty()) {
            for (std::size_t k = 0; k < K; ++k) {
                nodes.push_back(k);
            }
        }
        std::vector<double> alpha(nodes.size());
        for (std::size_t q = 0; q < nodes.size(); ++q) {
            alpha[q] = 1.0 / sigma_[nodes[q]];
        }
        std::vector<int> out(X.size());
        for (std::size_t n = 0; n < X.size(); ++n) {
            if (X[n].size() != d_) {
                throw std::invalid_argument("IDAT: all samples must have the same length");
            }
            // the most similar node (the first of equal similarities)
            std::size_t best = 0;
            double best_s = -std::numeric_limits<double>::infinity();
            for (std::size_t q = 0; q < nodes.size(); ++q) {
                const double s = similarity(alpha[q], distance(&w_[nodes[q] * d_], X[n].data()));
                if (s > best_s) {
                    best_s = s;
                    best = q;
                }
            }
            out[n] = labels[nodes[best]];
        }
        return out;
    }

    // fit(X) followed by predict(X).
    std::vector<int> fit_predict(const std::vector<std::vector<double>>& X) {
        fit(X);
        return predict(X);
    }

    // ---------------------------------------------------------------------------------------------
    // Accessors
    // ---------------------------------------------------------------------------------------------
    std::size_t num_nodes() const { return num_nodes_; }

    std::vector<std::vector<double>> weight() const {
        std::vector<std::vector<double>> W(num_nodes_);
        for (std::size_t k = 0; k < num_nodes_; ++k) {
            W[k].assign(w_.begin() + k * d_, w_.begin() + (k + 1) * d_);
        }
        return W;
    }

    std::vector<double> count_node() const { return first_entries(count_); }
    std::vector<double> buffer_std() const { return first_entries(sigma_); }
    std::vector<bool> is_weight() const { return first_entries(active_); }

    std::vector<std::pair<std::size_t, std::size_t>> edges() const {
        std::vector<std::pair<std::size_t, std::size_t>> E;
        for (std::size_t p = 0; p < num_pairs_; ++p) {
            if (pair_edge_[p] > 0) {
                E.emplace_back(pair_i_[p], pair_j_[p]);
            }
        }
        std::sort(E.begin(), E.end());
        return E;
    }

    std::size_t lambda() const { return lambda_; }
    double similarity_threshold() const { return similarity_threshold_; }
    const std::vector<int>& label_cluster() const { return label_cluster_; }
    std::size_t sample_counts() const { return sample_counts_; }

private:
    // In the arithmetic below, a product is stored in a variable before it is added, so that the
    // compiler does not merge the product and the sum into one fused multiply add.

    std::size_t d_ = 0;  // number of features
    std::size_t num_nodes_ = 0;
    std::size_t lambda_ = LAMBDA_INIT;
    double similarity_threshold_ = 0.0;
    std::vector<int> label_cluster_;
    std::size_t sample_counts_ = 0;

    std::size_t recalc_counter_ = 0;  // samples learned since the last adaptation
    double ratio_nc_ = 0.0;  // smoothed ratio of the number of clusters to the number of nodes

    // nodes: arrays with one entry per node. They have spare rows, and the first num_nodes_ are
    // used.
    std::size_t capacity_ = 0;
    std::vector<double> w_;  // prototypes, [capacity_ x d_] row by row
    std::vector<double> count_;  // number of wins
    std::vector<double> sigma_;  // sigma_k
    std::vector<bool> active_;  // true for the active nodes
    std::vector<double> degree_;  // number of edges of every node
    std::vector<double> cand_row_sum_;  // row sums of E^cand
    std::vector<double> prev_count_;  // count_ at the last maintenance
    std::vector<double> prev_cand_row_sum_;  // cand_row_sum_ at the last maintenance
    std::vector<double> idle_;  // consecutive maintenances as a demotion candidate
    // creation number of every node, kept when other nodes are removed
    std::vector<std::uint64_t> id_;
    std::uint64_t next_id_ = 0;

    // node pairs (i < j) that have been co-activated, with their edge count and E^cand.
    // A pair is looked up by the creation numbers of its two nodes.
    struct PairHash {
        std::size_t operator()(const std::pair<std::uint64_t, std::uint64_t>& key) const noexcept {
            return std::hash<std::uint64_t>()(key.first * 0x9E3779B97F4A7C15ULL ^ key.second);
        }
    };
    std::unordered_map<std::pair<std::uint64_t, std::uint64_t>, std::size_t, PairHash> pair_slot_;
    std::size_t num_pairs_ = 0;
    std::vector<std::size_t> pair_i_;
    std::vector<std::size_t> pair_j_;
    std::vector<double> pair_edge_;
    std::vector<double> pair_cand_;

    // running sums behind the two thresholds of the vigilance test (over the symmetric matrix)
    std::size_t sum_count_above1_ = 0;  // sum of the counts > 1
    std::size_t num_count_above1_ = 0;  // number of the counts > 1
    std::size_t sum_cand_ = 0;  // sum of the positive entries of E^cand
    std::size_t num_cand_ = 0;  // number of these entries

    long long num_components_ = 0;  // number of connected components (valid when not stale)
    bool components_stale_ = true;

    // running mean, second moment and count of the learned samples (Welford)
    std::vector<double> run_mean_;
    std::vector<double> run_m2_;
    std::size_t run_count_ = 0;

    // ring buffer of the most recent samples, [buf_rows_ x d_] row by row
    std::vector<double> buf_;
    std::size_t buf_rows_ = 0;
    std::size_t buf_start_ = 0;  // row of the oldest kept sample
    std::size_t buf_count_ = 0;  // number of kept samples
    std::size_t buf_keep_ = 0;  // samples to keep (0 before the first adaptation: all)

    template <typename T>
    std::vector<T> first_entries(const std::vector<T>& a) const {
        return std::vector<T>(a.begin(), a.begin() + num_nodes_);
    }

    // Euclidean distance, with the features added from the first to the last.
    double distance(const double* a, const double* b) const {
        double s = 0.0;
        for (std::size_t f = 0; f < d_; ++f) {
            const double diff = a[f] - b[f];
            const double sq = diff * diff;
            s += sq;
        }
        return std::sqrt(s);
    }

    // 1 / (1 + alpha * distance)
    static double similarity(double alpha, double dist) {
        const double scaled = alpha * dist;
        return 1.0 / (1.0 + scaled);
    }

    // w_k += rate * (x - w_k)
    void move_towards(std::size_t k, const std::vector<double>& x, double rate) {
        double* w = &w_[k * d_];
        for (std::size_t f = 0; f < d_; ++f) {
            const double step = rate * (x[f] - w[f]);
            w[f] += step;
        }
    }

    // ---------------------------------------------------------------------------------------------
    // One sample: category choice, vigilance test and maintenance
    // ---------------------------------------------------------------------------------------------
    void learn_one_sample(const std::vector<double>& x) {
        if (num_nodes_ < 3) {
            add_node(x);
            if (num_nodes_ == 2) {
                // the first node takes the scale of the first two samples
                sigma_[0] = sigma_[1];
            }
        } else {
            // category choice: the two nearest nodes (the first of equal distances is taken)
            const std::size_t K = num_nodes_;
            std::vector<double> distances(K);
            for (std::size_t k = 0; k < K; ++k) {
                distances[k] = distance(&w_[k * d_], x.data());
            }
            const std::size_t s1 = argmin(distances);
            const double d1 = distances[s1];
            distances[s1] = std::numeric_limits<double>::infinity();
            const std::size_t s2 = argmin(distances);
            const double d2 = distances[s2];
            const double sim1 = similarity(1.0 / sigma_[s1], d1);
            const double sim2 = similarity(1.0 / sigma_[s2], d2);
            const double V = similarity_threshold_;
            if (sim1 < V) {
                add_node(x);  // Case I
            } else {
                increment_count(s1);  // Case II
                move_towards(s1, x, 1.0 / count_[s1]);
                sigma_[s1] = std::max(update_running_std(x), SIGMA_MIN);
                if (sim2 > V) {  // Case III
                    increment_count(s2);
                    move_towards(s2, x, 1.0 / (count_[s1] + count_[s2]));
                    update_topology(s1, s2);
                }
            }
        }
        // maintenance every Lambda samples
        // (sample_counts_ is the number of samples learned before x)
        if (sample_counts_ % lambda_ == 0) {
            maintain();
        }
    }

    static std::size_t argmin(const std::vector<double>& v) {
        std::size_t best = 0;
        for (std::size_t k = 1; k < v.size(); ++k) {
            if (v[k] < v[best]) {
                best = k;
            }
        }
        return best;
    }

    // Activation of s1 and s2, their co-activation count and, possibly, an edge (Case III).
    void update_topology(std::size_t s1, std::size_t s2) {
        // mean of the counts > 1
        const double count_th =
            static_cast<double>(sum_count_above1_) / static_cast<double>(num_count_above1_);
        if (count_[s1] > count_th && count_[s2] > count_th) {
            active_[s1] = true;
            active_[s2] = true;
            const std::size_t slot = find_pair(s1, s2);
            const double c = pair_cand_[slot];
            pair_cand_[slot] = c + 1;
            cand_row_sum_[s1] += 1;
            cand_row_sum_[s2] += 1;
            sum_cand_ += 2;
            if (c == 0) {
                num_cand_ += 2;
            }
            // mean of the positive entries of E^cand
            const double edge_th = static_cast<double>(sum_cand_) / static_cast<double>(num_cand_);
            if (c + 1 > edge_th) {
                if (pair_edge_[slot] == 0) {
                    components_stale_ = true;
                    degree_[s1] += 1;
                    degree_[s2] += 1;
                }
                pair_edge_[slot] += 1;
            }
        }
    }

    void maintain() {
        const std::size_t K = num_nodes_;
        // 1) edge pruning: Tukey's rule on the edge lengths
        const std::size_t n = num_pairs_;
        std::vector<std::size_t> e;
        for (std::size_t p = 0; p < n; ++p) {
            if (pair_edge_[p] > 0) {
                e.push_back(p);
            }
        }
        if (!e.empty()) {
            std::vector<double> length(e.size());
            for (std::size_t q = 0; q < e.size(); ++q) {
                length[q] = distance(&w_[pair_i_[e[q]] * d_], &w_[pair_j_[e[q]] * d_]);
            }
            std::vector<double> both(length);  // every edge counts as (i, j) and (j, i)
            both.insert(both.end(), length.begin(), length.end());
            std::sort(both.begin(), both.end());
            const double Q1 = quantile(both, 0.25);
            const double Q3 = quantile(both, 0.75);
            const double IQR = Q3 - Q1;
            const double margin = IQR_FACTOR * IQR;
            const double upper = Q3 + margin;
            bool any_too_long = false;
            for (std::size_t q = 0; q < e.size(); ++q) {
                if (length[q] > upper) {
                    pair_edge_[e[q]] = 0;
                    pair_cand_[e[q]] = 0;
                    any_too_long = true;
                }
            }
            if (any_too_long) {
                refresh_candidate_sums();
                components_stale_ = true;
                std::fill(degree_.begin(), degree_.begin() + K, 0.0);
                std::fill(cand_row_sum_.begin(), cand_row_sum_.begin() + K, 0.0);
                for (std::size_t p = 0; p < n; ++p) {
                    if (pair_edge_[p] > 0) {
                        degree_[pair_i_[p]] += 1;
                        degree_[pair_j_[p]] += 1;
                    }
                    cand_row_sum_[pair_i_[p]] += pair_cand_[p];
                    cand_row_sum_[pair_j_[p]] += pair_cand_[p];
                }
                // many pruned pairs: drop them
                if (n - num_cand_ / 2 > std::max<std::size_t>(1024, n / 2)) {
                    compact_pairs();
                }
            }
        }
        // 2) removal of the nodes that have never won: the newest Lambda of them are kept
        const std::size_t num_count1 = K - num_count_above1_;
        if (num_count_above1_ > 0 && num_count1 > lambda_) {
            std::vector<std::size_t> r;
            for (std::size_t k = 0; k < K; ++k) {
                if (count_[k] == 1) {
                    r.push_back(k);
                }
            }
            r.resize(r.size() - lambda_);
            remove_nodes(r);
        }
        // 3) demotion of the isolated active nodes that stay idle
        demote_idle_nodes();
    }

    // The nodes r have count 1, so they have no edge and no co-activation. Every one of them is an
    // isolated component, and the degrees and row sums of the other nodes and the sums behind the
    // thresholds do not change.
    void remove_nodes(const std::vector<std::size_t>& r) {
        const std::size_t K = num_nodes_;
        std::vector<bool> keep(K, true);
        for (std::size_t k : r) {
            keep[k] = false;
        }
        std::vector<std::size_t> new_index(K);
        std::size_t m = 0;
        for (std::size_t k = 0; k < K; ++k) {
            new_index[k] = m;
            if (keep[k]) {
                for (std::size_t f = 0; f < d_; ++f) {
                    w_[m * d_ + f] = w_[k * d_ + f];
                }
                count_[m] = count_[k];
                sigma_[m] = sigma_[k];
                active_[m] = active_[k];
                degree_[m] = degree_[k];
                cand_row_sum_[m] = cand_row_sum_[k];
                prev_count_[m] = prev_count_[k];
                prev_cand_row_sum_[m] = prev_cand_row_sum_[k];
                idle_[m] = idle_[k];
                id_[m] = id_[k];
                m += 1;
            }
        }
        num_nodes_ = m;
        for (std::size_t p = 0; p < num_pairs_; ++p) {
            pair_i_[p] = new_index[pair_i_[p]];
            pair_j_[p] = new_index[pair_j_[p]];
        }
        num_components_ -= static_cast<long long>(r.size());
    }

    // The activity of a node is its number of wins and co-activations since the last maintenance.
    // An active node that is isolated and whose activity is at most max(Q1 - 1.5 IQR, 0) of the
    // positive activities for DEMOTE_TOLERANCE consecutive maintenances is demoted (is_weight()
    // becomes false).
    void demote_idle_nodes() {
        const std::size_t K = num_nodes_;
        std::vector<double> activity(K);
        bool any_activity = false;
        for (std::size_t k = 0; k < K; ++k) {
            activity[k] = (count_[k] - prev_count_[k]) + (cand_row_sum_[k] - prev_cand_row_sum_[k]);
            prev_count_[k] = count_[k];
            prev_cand_row_sum_[k] = cand_row_sum_[k];
            any_activity = any_activity || activity[k] > 0;
        }
        if (!any_activity) {
            std::fill(idle_.begin(), idle_.begin() + K, 0.0);
            return;
        }
        std::vector<bool> isolated_active(K);
        bool any_isolated_active = false;
        for (std::size_t k = 0; k < K; ++k) {
            isolated_active[k] = active_[k] && degree_[k] == 0;
            any_isolated_active = any_isolated_active || isolated_active[k];
        }
        std::vector<bool> candidate(K, false);
        if (any_isolated_active) {
            std::vector<double> a;
            for (std::size_t k = 0; k < K; ++k) {
                if (activity[k] > 0) {
                    a.push_back(activity[k]);
                }
            }
            std::sort(a.begin(), a.end());
            const double Q1 = quantile(a, 0.25);
            const double Q3 = quantile(a, 0.75);
            const double IQR = Q3 - Q1;
            const double margin = IQR_FACTOR * IQR;
            const double bound = std::max(Q1 - margin, 0.0);
            for (std::size_t k = 0; k < K; ++k) {
                candidate[k] = isolated_active[k] && activity[k] <= bound;
            }
        }
        for (std::size_t k = 0; k < K; ++k) {
            if (candidate[k]) {
                idle_[k] += 1;
            } else {
                idle_[k] = 0;
            }
            if (idle_[k] >= DEMOTE_TOLERANCE) {
                active_[k] = false;
                idle_[k] = 0;
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Adaptation of Lambda and V
    // ---------------------------------------------------------------------------------------------

    // New Lambda: the largest L such that det(S_L) >= EPS_DET, where S_L is the similarity matrix
    // of the L most recent samples. One Cholesky factorisation of the similarity matrix of the
    // whole window gives every det(S_L) as a product of squared pivots.
    // New V: the (1 - r) quantile of the largest similarity of every one of the Lambda most recent
    // samples to another one, where r is the smoothed ratio of the number of clusters to the number
    // of nodes.
    void adapt_lambda_and_threshold() {
        if (components_stale_) {
            num_components_ = static_cast<long long>(count_components());
            components_stale_ = false;
        }
        const std::size_t K = num_nodes_;
        const std::size_t lam = lambda_;
        const double alpha = 1.0 / *std::max_element(sigma_.begin(), sigma_.begin() + K);
        const std::size_t L = std::min(BUFFER_FACTOR * lam, buf_count_);
        // the rows and columns start from the newest sample
        const std::vector<double> rows = newest_rows(L);
        std::vector<double> S(L * L);
        for (std::size_t i = 0; i < L; ++i) {
            S[i * L + i] = similarity(alpha, 0.0);
            for (std::size_t j = i + 1; j < L; ++j) {
                const double s = similarity(alpha, distance(&rows[i * d_], &rows[j * d_]));
                S[i * L + j] = s;
                S[j * L + i] = s;
            }
        }
        // Cholesky factorisation S = R R^T, row by row. It stops at the first pivot that is not
        // positive, so that only the leading m rows are factorised.
        std::vector<double> R(L * L, 0.0);
        std::size_t m = L;
        for (std::size_t i = 0; i < L && m == L; ++i) {
            for (std::size_t j = 0; j < i; ++j) {
                double dot = 0.0;
                for (std::size_t k = 0; k < j; ++k) {
                    const double prod = R[i * L + k] * R[j * L + k];
                    dot += prod;
                }
                R[i * L + j] = (S[i * L + j] - dot) / R[j * L + j];
            }
            double dot = 0.0;
            for (std::size_t k = 0; k < i; ++k) {
                const double prod = R[i * L + k] * R[i * L + k];
                dot += prod;
            }
            const double pivot = S[i * L + i] - dot;
            if (pivot > 0.0) {
                R[i * L + i] = std::sqrt(pivot);
            } else {
                m = i;
            }
        }
        // det(S_L) for L = 1..m is the square of the product of the first L diagonal entries
        std::size_t new_lambda = m;
        double cp = 1.0;
        for (std::size_t q = 0; q < m; ++q) {
            cp *= R[q * L + q];
            if (cp * cp < EPS_DET) {
                new_lambda = q;
                break;
            }
        }
        if (new_lambda < 2) {
            return;  // not even two samples are diverse: keep Lambda, V and r
        }
        double beta;
        if (new_lambda < lam) {
            const double base = 1.0 - 1.0 / static_cast<double>(lam);
            beta = 1.0 - std::pow(base, static_cast<double>(new_lambda - 1));
        } else {
            beta = 1.0 / static_cast<double>(lam);
        }
        const double ratio_now = static_cast<double>(num_components_) / static_cast<double>(K);
        const double part_now = beta * ratio_now;
        const double part_old = (1.0 - beta) * ratio_nc_;
        ratio_nc_ = part_now + part_old;
        std::vector<double> u(new_lambda, -std::numeric_limits<double>::infinity());
        for (std::size_t i = 0; i < new_lambda; ++i) {
            for (std::size_t j = 0; j < new_lambda; ++j) {
                if (j != i) {
                    u[i] = std::max(u[i], S[i * L + j]);
                }
            }
        }
        std::sort(u.begin(), u.end());
        lambda_ = new_lambda;
        similarity_threshold_ = quantile(u, 1.0 - ratio_nc_);
    }

    // Labels of the connected components of the edge graph, numbered from 1 in the order of their
    // smallest node index. Returns the number of components.
    std::size_t connected_components(std::vector<int>& labels) const {
        const std::size_t K = num_nodes_;
        // adjacency lists of the edge graph
        std::vector<std::size_t> start(K + 1, 0);
        for (std::size_t p = 0; p < num_pairs_; ++p) {
            if (pair_edge_[p] > 0) {
                start[pair_i_[p] + 1] += 1;
                start[pair_j_[p] + 1] += 1;
            }
        }
        for (std::size_t k = 0; k < K; ++k) {
            start[k + 1] += start[k];
        }
        std::vector<std::size_t> neighbor(start[K]);
        std::vector<std::size_t> next(start.begin(), start.end() - 1);
        for (std::size_t p = 0; p < num_pairs_; ++p) {
            if (pair_edge_[p] > 0) {
                neighbor[next[pair_i_[p]]++] = pair_j_[p];
                neighbor[next[pair_j_[p]]++] = pair_i_[p];
            }
        }
        // breadth first search from every node that has no label yet
        labels.assign(K, 0);
        int num = 0;
        std::vector<std::size_t> queue;
        for (std::size_t v = 0; v < K; ++v) {
            if (labels[v] != 0) {
                continue;
            }
            num += 1;
            labels[v] = num;
            queue.assign(1, v);
            for (std::size_t h = 0; h < queue.size(); ++h) {
                const std::size_t a = queue[h];
                for (std::size_t q = start[a]; q < start[a + 1]; ++q) {
                    const std::size_t b = neighbor[q];
                    if (labels[b] == 0) {
                        labels[b] = num;
                        queue.push_back(b);
                    }
                }
            }
        }
        return static_cast<std::size_t>(num);
    }

    // Number of connected components of the edge graph.
    std::size_t count_components() const {
        std::vector<int> labels;
        return connected_components(labels);
    }

    // ---------------------------------------------------------------------------------------------
    // Nodes, node pairs and running sums
    // ---------------------------------------------------------------------------------------------
    void add_node(const std::vector<double>& x) {
        const std::size_t k = num_nodes_;
        if (k == capacity_) {
            grow_node_arrays();
        }
        num_nodes_ = k + 1;
        std::copy(x.begin(), x.end(), w_.begin() + k * d_);
        count_[k] = 1;
        sigma_[k] = std::max(update_running_std(x), SIGMA_MIN);
        active_[k] = false;
        degree_[k] = 0;
        cand_row_sum_[k] = 0;
        prev_count_[k] = 0;
        prev_cand_row_sum_[k] = 0;
        idle_[k] = 0;
        id_[k] = next_id_;
        next_id_ += 1;
        num_components_ += 1;  // a new node is an isolated component
    }

    void grow_node_arrays() {
        capacity_ = capacity_ == 0 ? 64 : 2 * capacity_;
        w_.resize(capacity_ * d_, 0.0);
        count_.resize(capacity_, 0.0);
        sigma_.resize(capacity_, 0.0);
        active_.resize(capacity_, false);
        degree_.resize(capacity_, 0.0);
        cand_row_sum_.resize(capacity_, 0.0);
        prev_count_.resize(capacity_, 0.0);
        prev_cand_row_sum_.resize(capacity_, 0.0);
        idle_.resize(capacity_, 0.0);
        id_.resize(capacity_, 0);
    }

    void increment_count(std::size_t k) {
        const double c = count_[k];
        count_[k] = c + 1;
        if (c == 1) {  // the node now has a count > 1
            num_count_above1_ += 1;
            sum_count_above1_ += 2;
        } else {
            sum_count_above1_ += 1;
        }
    }

    // Slot of the node pair (a, b), created when the pair is new.
    std::size_t find_pair(std::size_t a, std::size_t b) {
        const std::size_t i = a < b ? a : b;
        const std::size_t j = a < b ? b : a;
        const std::pair<std::uint64_t, std::uint64_t> key(id_[i], id_[j]);
        const auto found = pair_slot_.find(key);
        if (found != pair_slot_.end()) {
            return found->second;
        }
        const std::size_t slot = num_pairs_;
        if (slot == pair_i_.size()) {
            const std::size_t capacity = std::max<std::size_t>(256, 2 * slot);
            pair_i_.resize(capacity, 0);
            pair_j_.resize(capacity, 0);
            pair_edge_.resize(capacity, 0.0);
            pair_cand_.resize(capacity, 0.0);
        }
        pair_i_[slot] = i;
        pair_j_[slot] = j;
        pair_edge_[slot] = 0;
        pair_cand_[slot] = 0;
        pair_slot_[key] = slot;
        num_pairs_ += 1;
        return slot;
    }

    // Keep only the pairs with a positive co-activation count (a pruned pair has none).
    void compact_pairs() {
        std::size_t m = 0;
        for (std::size_t p = 0; p < num_pairs_; ++p) {
            if (pair_cand_[p] > 0) {
                pair_i_[m] = pair_i_[p];
                pair_j_[m] = pair_j_[p];
                pair_edge_[m] = pair_edge_[p];
                pair_cand_[m] = pair_cand_[p];
                m += 1;
            }
        }
        num_pairs_ = m;
        pair_slot_.clear();
        for (std::size_t p = 0; p < m; ++p) {
            pair_slot_[std::make_pair(id_[pair_i_[p]], id_[pair_j_[p]])] = p;
        }
    }

    void refresh_candidate_sums() {
        double total = 0.0;
        std::size_t num = 0;
        for (std::size_t p = 0; p < num_pairs_; ++p) {
            if (pair_cand_[p] > 0) {
                total += pair_cand_[p];
                num += 1;
            }
        }
        // every pair is two entries of the symmetric E^cand
        sum_cand_ = 2 * static_cast<std::size_t>(total);
        num_cand_ = 2 * num;
    }

    // Welford's update of the running mean and variance of the learned samples. Returns the largest
    // standard deviation over the features.
    double update_running_std(const std::vector<double>& x) {
        if (run_count_ == 0) {
            run_mean_ = x;
            run_m2_.assign(x.size(), 0.0);
            run_count_ = 1;
            return 0.0;
        }
        run_count_ += 1;
        const double n = static_cast<double>(run_count_);
        double m2_max = -std::numeric_limits<double>::infinity();
        for (std::size_t f = 0; f < d_; ++f) {
            const double delta = x[f] - run_mean_[f];
            run_mean_[f] += delta / n;
            const double prod = delta * (x[f] - run_mean_[f]);
            run_m2_[f] += prod;
            m2_max = std::max(m2_max, run_m2_[f]);
        }
        // the division by a positive number and the square root keep the order of the features
        return std::sqrt(m2_max / static_cast<double>(run_count_ - 1));
    }

    // ---------------------------------------------------------------------------------------------
    // Ring buffer of the most recent samples
    // ---------------------------------------------------------------------------------------------
    void append_to_buffer(const std::vector<double>& x) {
        if (buf_rows_ == 0) {
            buf_rows_ = 500;
            buf_.assign(buf_rows_ * d_, 0.0);
        } else if (buf_count_ == buf_rows_) {
            grow_buffer(2 * buf_rows_);
        }
        const std::size_t row = (buf_start_ + buf_count_) % buf_rows_;
        std::copy(x.begin(), x.end(), buf_.begin() + row * d_);
        buf_count_ += 1;
        if (0 < buf_keep_ && buf_keep_ < buf_count_) {
            drop_oldest(buf_count_ - buf_keep_);
        }
    }

    void update_buffer_keep_len() {
        buf_keep_ = BUFFER_FACTOR * lambda_;
        if (buf_count_ > buf_keep_) {
            drop_oldest(buf_count_ - buf_keep_);
        }
    }

    void drop_oldest(std::size_t n) {
        buf_start_ = (buf_start_ + n) % buf_rows_;
        buf_count_ -= n;
    }

    void grow_buffer(std::size_t capacity) {
        const std::vector<double> rows = newest_rows(buf_count_);
        buf_.assign(capacity * d_, 0.0);
        for (std::size_t q = 0; q < buf_count_; ++q) {  // the oldest sample first
            const std::size_t src = buf_count_ - 1 - q;
            std::copy_n(rows.begin() + src * d_, d_, buf_.begin() + q * d_);
        }
        buf_rows_ = capacity;
        buf_start_ = 0;
    }

    // The L most recent samples, the newest first, [L x d_] row by row.
    std::vector<double> newest_rows(std::size_t L) const {
        std::vector<double> rows(L * d_);
        for (std::size_t q = 0; q < L; ++q) {
            const std::size_t row = (buf_start_ + buf_count_ - 1 - q) % buf_rows_;
            std::copy_n(buf_.begin() + row * d_, d_, rows.begin() + q * d_);
        }
        return rows;
    }

    // The p quantile of x, sorted in ascending order: the k-th smallest value is the (k - 0.5) / n
    // quantile, with linear interpolation in between.
    static double quantile(const std::vector<double>& x, double p) {
        const double n = static_cast<double>(x.size());
        const double scaled = p * n;
        const double pos = std::min(std::max(scaled - 0.5, 0.0), n - 1.0);
        const std::size_t k = static_cast<std::size_t>(pos);
        const std::size_t k1 = std::min(k + 1, x.size() - 1);
        const double step = (pos - static_cast<double>(k)) * (x[k1] - x[k]);
        return x[k] + step;
    }
};

#endif  // IDAT_HPP

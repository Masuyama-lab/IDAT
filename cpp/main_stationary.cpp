// Copyright (c) 2025-2026 Naoki Masuyama
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "clustering_evaluation_metrics.hpp"
#include "idat.hpp"

const int NUM_TRIALS = 10;  // Number of trials

// Load dataset ============
const std::string DATA_FILE = "../datasets/iris.csv";
// const std::string DATA_FILE = "../datasets/optdigits.csv";

// Every line of the file holds the features of one sample and its class label in the last column.
bool load_csv(const std::string& path, std::vector<std::vector<double>>& data,
              std::vector<int>& target) {
    std::ifstream file(path);
    if (!file) {
        return false;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<double> row;
        std::stringstream fields(line);
        std::string field;
        while (std::getline(fields, field, ',')) {
            row.push_back(std::strtod(field.c_str(), nullptr));
        }
        target.push_back(static_cast<int>(row.back()));
        row.pop_back();
        data.push_back(row);
    }
    return true;
}

// Number of all nodes, of active nodes, and of clusters, AMI, ARI, and training time
std::vector<double> run_trial(int t, const std::vector<std::vector<double>>& data,
                              const std::vector<int>& target) {
    // Randomization
    std::vector<std::size_t> order(data.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(t);
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    for (std::size_t n : order) {
        X.push_back(data[n]);
        y.push_back(target[n]);
    }

    // Train IDAT
    IDAT model;
    const auto start = std::chrono::steady_clock::now();
    model.fit(X);
    const double train_time =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    // Test IDAT
    const std::vector<int> predicted_labels = model.predict(X);

    // Evaluation
    const auto [ami, ari] = clustering_evaluation_metrics(y, predicted_labels);

    // Number of all nodes, of active nodes, and of clusters
    // (clusters with at least one active node)
    const std::vector<bool> is_weight = model.is_weight();
    std::set<int> clusters;
    double num_active_nodes = 0;
    for (std::size_t k = 0; k < is_weight.size(); ++k) {
        if (is_weight[k]) {
            num_active_nodes += 1;
            clusters.insert(model.label_cluster()[k]);
        }
    }
    return {static_cast<double>(model.num_nodes()), num_active_nodes,
            static_cast<double>(clusters.size()), ami, ari, train_time};
}

int main() {
    std::vector<std::vector<double>> data;
    std::vector<int> target;
    if (!load_csv(DATA_FILE, data, target)) {
        std::fprintf(stderr, "Cannot open %s\n", DATA_FILE.c_str());
        return 1;
    }
    const std::size_t num_classes = std::set<int>(target.begin(), target.end()).size();

    std::vector<std::vector<double>> results;
    for (int t = 1; t <= NUM_TRIALS; ++t) {
        results.push_back(run_trial(t, data, target));
    }

    // Mean and standard deviation over the trials
    const std::size_t num_values = results[0].size();
    std::vector<double> mean(num_values, 0.0);
    std::vector<double> std_dev(num_values, 0.0);
    for (std::size_t v = 0; v < num_values; ++v) {
        for (const std::vector<double>& r : results) {
            mean[v] += r[v];
        }
        mean[v] /= NUM_TRIALS;
        for (const std::vector<double>& r : results) {
            std_dev[v] += (r[v] - mean[v]) * (r[v] - mean[v]);
        }
        std_dev[v] = std::sqrt(std_dev[v] / (NUM_TRIALS - 1));
    }

    // Display average and standard deviation results
    std::printf("=========== Stationary ============\n");
    std::printf("# classes: %zu\n", num_classes);
    std::printf("Average results over %d runs\n", NUM_TRIALS);
    std::printf("Average # all nodes:     %.4f ± %.4f\n", mean[0], std_dev[0]);
    std::printf("Average # active nodes:  %.4f ± %.4f\n", mean[1], std_dev[1]);
    std::printf("Average # clusters:      %.4f ± %.4f\n", mean[2], std_dev[2]);
    std::printf("         Average AMI:    %.4f ± %.4f\n", mean[3], std_dev[3]);
    std::printf("         Average ARI:    %.4f ± %.4f\n", mean[4], std_dev[4]);
    std::printf("        Average Time:    %.4f ± %.4f\n", mean[5], std_dev[5]);
    std::printf("\n");
    return 0;
}

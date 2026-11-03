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

// Samples and labels of every class (shuffled within each class)
std::vector<std::vector<std::vector<double>>> class_data;
std::vector<std::vector<int>> class_target;

// Number of all nodes, of active nodes, and of clusters, AMI, ARI, training time,
// incremental AMI and ARI, and BWT of AMI and ARI
std::vector<double> run_trial(int t) {
    const std::size_t num_classes = class_data.size();

    // Class order of this trial
    std::vector<std::size_t> class_order(num_classes);
    std::iota(class_order.begin(), class_order.end(), 0);
    std::mt19937 rng(t);
    std::shuffle(class_order.begin(), class_order.end(), rng);

    // Scores after learning each class
    std::vector<double> ami_steps(num_classes, 0.0);
    std::vector<double> ari_steps(num_classes, 0.0);

    IDAT model;
    double train_time = 0.0;
    std::vector<std::vector<double>> eval_data;
    std::vector<int> eval_target;
    for (std::size_t i = 0; i < num_classes; ++i) {
        const std::size_t c = class_order[i];

        // Train IDAT on the current class
        const auto start = std::chrono::steady_clock::now();
        model.fit(class_data[c]);
        train_time +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

        // Evaluate on all classes learned so far
        eval_data.insert(eval_data.end(), class_data[c].begin(), class_data[c].end());
        eval_target.insert(eval_target.end(), class_target[c].begin(), class_target[c].end());
        if (i == 0) {
            ami_steps[i] = ari_steps[i] = 1.0;
        } else {
            const std::vector<int> predicted_labels = model.predict(eval_data);
            const auto [ami, ari] = clustering_evaluation_metrics(eval_target, predicted_labels);
            ami_steps[i] = ami;
            ari_steps[i] = ari;
        }
    }

    // Number of all nodes, of active nodes, and of clusters
    // (clusters with at least one active node) after learning all classes
    const std::vector<bool> is_weight = model.is_weight();
    std::set<int> clusters;
    double num_active_nodes = 0;
    for (std::size_t k = 0; k < is_weight.size(); ++k) {
        if (is_weight[k]) {
            num_active_nodes += 1;
            clusters.insert(model.label_cluster()[k]);
        }
    }

    // Average incremental AMI/ARI and backward transfer (BWT)
    double inc_ami = 0.0, inc_ari = 0.0, bwt_ami = 0.0, bwt_ari = 0.0;
    for (std::size_t i = 0; i < num_classes; ++i) {
        inc_ami += ami_steps[i];
        inc_ari += ari_steps[i];
    }
    inc_ami /= num_classes;
    inc_ari /= num_classes;
    for (std::size_t i = 0; i + 1 < num_classes; ++i) {
        bwt_ami += ami_steps[num_classes - 1] - ami_steps[i];
        bwt_ari += ari_steps[num_classes - 1] - ari_steps[i];
    }
    bwt_ami /= num_classes - 1;
    bwt_ari /= num_classes - 1;
    return {static_cast<double>(model.num_nodes()),
            num_active_nodes,
            static_cast<double>(clusters.size()),
            ami_steps[num_classes - 1],
            ari_steps[num_classes - 1],
            train_time,
            inc_ami,
            inc_ari,
            bwt_ami,
            bwt_ari};
}

int main() {
    std::vector<std::vector<double>> data;
    std::vector<int> target;
    if (!load_csv(DATA_FILE, data, target)) {
        std::fprintf(stderr, "Cannot open %s\n", DATA_FILE.c_str());
        return 1;
    }

    // Split data by class (shuffled within each class)
    std::mt19937 rng(0);
    const std::set<int> class_labels(target.begin(), target.end());
    const std::size_t num_classes = class_labels.size();
    for (int c : class_labels) {
        std::vector<std::size_t> idx;
        for (std::size_t n = 0; n < target.size(); ++n) {
            if (target[n] == c) {
                idx.push_back(n);
            }
        }
        std::shuffle(idx.begin(), idx.end(), rng);
        class_data.emplace_back();
        class_target.emplace_back();
        for (std::size_t n : idx) {
            class_data.back().push_back(data[n]);
            class_target.back().push_back(target[n]);
        }
    }

    std::vector<std::vector<double>> results;
    for (int t = 1; t <= NUM_TRIALS; ++t) {
        results.push_back(run_trial(t));
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
    std::printf("=========== Nonstationary ============\n");
    std::printf("# classes: %zu\n", num_classes);
    std::printf("Average results over %d runs\n", NUM_TRIALS);
    std::printf("Average # all nodes:     %.4f ± %.4f\n", mean[0], std_dev[0]);
    std::printf("Average # active nodes:  %.4f ± %.4f\n", mean[1], std_dev[1]);
    std::printf("Average # clusters:      %.4f ± %.4f\n", mean[2], std_dev[2]);
    std::printf("         Average AMI:    %.4f ± %.4f\n", mean[3], std_dev[3]);
    std::printf("         Average ARI:    %.4f ± %.4f\n", mean[4], std_dev[4]);
    std::printf("        Average Time:    %.4f ± %.4f\n", mean[5], std_dev[5]);
    std::printf("\n");
    std::printf("Average Incremental AMI: %.4f ± %.4f\n", mean[6], std_dev[6]);
    std::printf("Average Incremental ARI: %.4f ± %.4f\n", mean[7], std_dev[7]);
    std::printf("  Backward Transfer AMI: %.4f ± %.4f\n", mean[8], std_dev[8]);
    std::printf("  Backward Transfer ARI: %.4f ± %.4f\n", mean[9], std_dev[9]);
    std::printf("\n");
    return 0;
}

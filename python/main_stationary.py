# Copyright (c) 2025-2026 Naoki Masuyama
# SPDX-License-Identifier: MIT

import os
import time

import numpy as np
from joblib import Parallel, delayed
from scipy.io import loadmat
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from idat import IDAT

N_TRIALS = 10  # Number of trials

# Load dataset ============
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")
mat = loadmat(os.path.join(data_dir, "iris.mat"))
# mat = loadmat(os.path.join(data_dir, "optdigits.mat"))

data = mat["data"].astype(np.float64)
target = mat["target"].ravel().astype(int)


def run_trial(t):
    # Randomization
    order = np.random.RandomState(t).permutation(data.shape[0])
    X, y = data[order], target[order]

    # Train IDAT
    model = IDAT()
    start = time.perf_counter()
    model.fit(X)
    train_time = time.perf_counter() - start

    # Test IDAT
    predicted_labels = model.predict(X)

    # Evaluation
    ami = adjusted_mutual_info_score(y, predicted_labels)
    ari = adjusted_rand_score(y, predicted_labels)

    # Number of all nodes, of active nodes, and of clusters
    # (clusters with at least one active node)
    num_nodes = model.num_nodes_
    num_active_nodes = np.count_nonzero(model.is_weight_)
    num_clusters = np.unique(model.label_cluster_[model.is_weight_]).size
    return num_nodes, num_active_nodes, num_clusters, ami, ari, train_time


results = np.array(Parallel(n_jobs=-1)(delayed(run_trial)(t) for t in range(1, N_TRIALS + 1)))
mean, std = results.mean(axis=0), results.std(axis=0, ddof=1)

# Display average and standard deviation results
print("=========== Stationary ============")
print(f"# classes: {np.unique(target).size}")
print(f"Average results over {N_TRIALS} runs")
print(f"Average # all nodes:     {mean[0]:.4f} ± {std[0]:.4f}")
print(f"Average # active nodes:  {mean[1]:.4f} ± {std[1]:.4f}")
print(f"Average # clusters:      {mean[2]:.4f} ± {std[2]:.4f}")
print(f"         Average AMI:    {mean[3]:.4f} ± {std[3]:.4f}")
print(f"         Average ARI:    {mean[4]:.4f} ± {std[4]:.4f}")
print(f"        Average Time:    {mean[5]:.4f} ± {std[5]:.4f}")
print()

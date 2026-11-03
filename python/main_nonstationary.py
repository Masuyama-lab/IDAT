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

# Split data by class (shuffled within each class)
rng = np.random.RandomState(0)
class_labels = np.unique(target)
num_classes = class_labels.size
class_data, class_target = [], []
for c in class_labels:
    idx = rng.permutation(np.flatnonzero(target == c))
    class_data.append(data[idx])
    class_target.append(target[idx])


def run_trial(t):
    # Class order of this trial
    class_order = np.random.RandomState(t).permutation(num_classes)

    # Scores after learning each class
    ami_steps = np.zeros(num_classes)
    ari_steps = np.zeros(num_classes)

    model = IDAT()
    train_time = 0.0
    for i, c in enumerate(class_order):
        # Train IDAT on the current class
        start = time.perf_counter()
        model.fit(class_data[c])
        train_time += time.perf_counter() - start

        # Evaluate on all classes learned so far
        if i == 0:
            ami_steps[i] = ari_steps[i] = 1.0
        else:
            eval_data = np.vstack([class_data[k] for k in class_order[: i + 1]])
            eval_target = np.concatenate([class_target[k] for k in class_order[: i + 1]])
            predicted_labels = model.predict(eval_data)
            ami_steps[i] = adjusted_mutual_info_score(eval_target, predicted_labels)
            ari_steps[i] = adjusted_rand_score(eval_target, predicted_labels)

    # Number of all nodes, of active nodes, and of clusters
    # (clusters with at least one active node) after learning all classes
    num_nodes = model.num_nodes_
    num_active_nodes = np.count_nonzero(model.is_weight_)
    num_clusters = np.unique(model.label_cluster_[model.is_weight_]).size

    # Average incremental AMI/ARI and backward transfer (BWT)
    inc_ami, inc_ari = ami_steps.mean(), ari_steps.mean()
    bwt_ami = np.mean(ami_steps[-1] - ami_steps[:-1])
    bwt_ari = np.mean(ari_steps[-1] - ari_steps[:-1])
    return (
        num_nodes,
        num_active_nodes,
        num_clusters,
        ami_steps[-1],
        ari_steps[-1],
        train_time,
        inc_ami,
        inc_ari,
        bwt_ami,
        bwt_ari,
    )


results = np.array(Parallel(n_jobs=-1)(delayed(run_trial)(t) for t in range(1, N_TRIALS + 1)))
mean, std = results.mean(axis=0), results.std(axis=0, ddof=1)

# Display average and standard deviation results
print("=========== Nonstationary ============")
print(f"# classes: {num_classes}")
print(f"Average results over {N_TRIALS} runs")
print(f"Average # all nodes:     {mean[0]:.4f} ± {std[0]:.4f}")
print(f"Average # active nodes:  {mean[1]:.4f} ± {std[1]:.4f}")
print(f"Average # clusters:      {mean[2]:.4f} ± {std[2]:.4f}")
print(f"         Average AMI:    {mean[3]:.4f} ± {std[3]:.4f}")
print(f"         Average ARI:    {mean[4]:.4f} ± {std[4]:.4f}")
print(f"        Average Time:    {mean[5]:.4f} ± {std[5]:.4f}")
print()
print(f"Average Incremental AMI: {mean[6]:.4f} ± {std[6]:.4f}")
print(f"Average Incremental ARI: {mean[7]:.4f} ± {std[7]:.4f}")
print(f"  Backward Transfer AMI: {mean[8]:.4f} ± {std[8]:.4f}")
print(f"  Backward Transfer ARI: {mean[9]:.4f} ± {std[9]:.4f}")
print()

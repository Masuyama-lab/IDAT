# eval_metrics.py
# -*- coding: utf-8 -*-
"""
Clustering evaluation metrics in pure Python:
- AMI, ARI via scikit-learn
- NVI (Normalized Variation of Information)
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score


def clustering_evaluation_metrics(ground_truth, predicted):
    """Return AMI, ARI, NID, NVI for given ground-truth and predicted labels."""
    gt = np.asarray(ground_truth).ravel()
    pr = np.asarray(predicted).ravel()
    if gt.shape[0] != pr.shape[0]:
        raise ValueError("ground_truth and predicted must have the same length.")

    ami = adjusted_mutual_info_score(gt, pr)
    ari = adjusted_rand_score(gt, pr)
    nvi = normalized_variation_information(gt, pr)
    return ami, ari, nvi


def _contingency(ic1: np.ndarray, ic2: np.ndarray):
    """Build contingency matrix nij from integer-coded labels ic1, ic2 (both >=0)."""
    n = ic1.shape[0]
    k1 = int(ic1.max()) + 1
    k2 = int(ic2.max()) + 1
    nij = np.zeros((k1, k2), dtype=np.int64)
    # Efficient scatter-add
    for i in range(n):
        nij[ic1[i], ic2[i]] += 1
    return nij


def _entropies_from_contingency(nij: np.ndarray):
    """Return H(U,V), H(U), H(V) using natural log, ignoring zero-probabilities."""
    n = float(nij.sum())
    p_ij = nij / n
    p_i = p_ij.sum(axis=1)
    p_j = p_ij.sum(axis=0)

    # Use boolean masks to avoid log(0)
    mask_ij = p_ij > 0
    mask_i = p_i > 0
    mask_j = p_j > 0

    huv = -np.sum(p_ij[mask_ij] * np.log(p_ij[mask_ij]))
    hu = -np.sum(p_i[mask_i] * np.log(p_i[mask_i]))
    hv = -np.sum(p_j[mask_j] * np.log(p_j[mask_j]))
    return huv, hu, hv


def normalized_variation_information(c1, c2):
    """Normalized Variation of Information: 1 - MI / H(U,V). Lower is better (0 best)."""
    c1 = np.asarray(c1).ravel()
    c2 = np.asarray(c2).ravel()
    if c1.shape[0] != c2.shape[0]:
        raise ValueError("c1 and c2 must have the same length.")

    _, ic1 = np.unique(c1, return_inverse=True)
    _, ic2 = np.unique(c2, return_inverse=True)

    nij = _contingency(ic1, ic2)
    huv, hu, hv = _entropies_from_contingency(nij)
    mi = hu + hv - huv
    denom = huv if huv > 0 else 1.0
    return 1.0 - (mi / denom)

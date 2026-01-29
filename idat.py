# Copyright (c) 2025-2026 Naoki Masuyama
# SPDX-License-Identifier: MIT
#
# This file is part of IDAT.
# Licensed under the MIT License; see LICENSE in the project root for details.

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.spatial.distance import cdist
import networkx as nx


class IDAT(BaseEstimator, ClusterMixin):
    def __init__(self):
        # Initialize internal fields and parameters for the IDAT clustering model.
        self.num_nodes_ = 0                 # Number of nodes.
        self.weight_ = None                 # Prototype vectors for each node.
        self.cluster_counts_ = None         # Number of samples assigned to each node.
        self.buffer_std_ = None             # Per-node std used for distance scaling (alpha_k = 1 / std_k).
        self.buffer_data_ = None            # Ring buffer for incremental data storage (oldest → newest).
        self.sample_counts_ = 0             # Total number of processed samples.
        self.G = nx.Graph()                 # Main graph of permanent edges among nodes.
        self.lambda_ = 2                    # Maintenance interval; triggers (Lambda, threshold) recalculation.
        self.similarity_threshold_ = 0.0    # Vigilance threshold for resonance (match if sim ≥ threshold).
        self.recalc_counter_ = 0            # Counter for triggering recalculation at cadence lambda_.
        self.candidate_G = nx.Graph()       # Candidate edges before promotion to the main graph.
        self.is_weight_ = None              # Boolean mask; if any True, restrict prediction to these nodes.
        self.label_cluster_ = None          # Connected-component labels over the current graph.

        # Online variance (Welford)
        self.mean_data_ = None              # Running mean (global, feature-wise).
        self.M2_data_ = None                # Running second central moment (global).
        self.count_data_ = 0                # Running sample count.

        # Ring buffer internals
        self.buffer_data_capacity_ = 0      # Physical capacity (≥ logical retention).
        self.buffer_data_start_ = 0         # Logical head index.
        self.buffer_data_count_ = 0         # Logical length.

        # Logical retention (keep 2*lambda_ most recent samples; do not shrink capacity)
        self.buffer_keep_len_ = 0

        # Smoothed ratio of clusters to nodes for threshold selection
        self.past_ratioNC_ = 0.0

        # Demotion state snapshots and counters
        self.prev_cluster_counts_ = None
        self.prev_edge_row_sum_ = None
        self.demotion_tolerance_count_ = None

    def fit(self, samples, y=None):
        """
        Incrementally train on samples; process strictly sequentially (no lookahead).
        Inputs: samples [N x d].
        Notes:
          - Initialize lambda_/threshold on first call.
          - Recalculate when recalc_counter_ reaches lambda_, using only past data windows.
        """
        samples = samples.astype(np.float64)

        # Sequential processing (increment counter before recalculation).
        for sample_num in range(samples.shape[0]):

            sample = samples[sample_num:sample_num + 1, :]

            # Learn current sample
            self._append_to_buffer(sample)
            self._cluster_step(sample)

            self.sample_counts_ += 1
            self.recalc_counter_ += 1

            # Recalculate thresholds on schedule using only past data.
            if (self.recalc_counter_ >= int(self.lambda_)) and (self.weight_.shape[0] > 2):
                num_clusters = nx.number_connected_components(self.G.subgraph(range(self.num_nodes_)))

                # Decremental direction: search for first instability; adopt last stable window.
                Lambda_new, similarity_th_new, is_incremental, ratioNC_candidate = self._calculate_lambda_decremental_direction(
                    self.buffer_std_,
                    int(self.lambda_),
                    self.similarity_threshold_,
                    self._get_from_buffer(int(self.lambda_)),
                    self.num_nodes_,
                    num_clusters
                )

                # Incremental direction: try to widen up to 2*lambda_ using older past only.
                if is_incremental:
                    Lambda_new, similarity_th_new, ratioNC_candidate = self._calculate_lambda_incremental_direction(
                        self.buffer_std_,
                        int(self.lambda_),
                        self.similarity_threshold_,
                        self.num_nodes_,
                        num_clusters
                    )

                # Commit once; reset cadence.
                self.lambda_ = int(Lambda_new)
                self.similarity_threshold_ = float(similarity_th_new)
                self.past_ratioNC_ = float(ratioNC_candidate)
                self.recalc_counter_ = 0

                # Update retention (2*lambda_) and trim logically if necessary.
                self._update_buffer_keep_len()

        return self

    def predict(self, samples):
        samples = samples.astype(np.float64, copy=False)

        if getattr(self, "num_nodes_", 0) == 0 or self.weight_ is None or self.weight_.shape[0] == 0:
            return np.array([], dtype=int)

        if self.is_weight_ is None or not np.any(self.is_weight_):
            center_indices = np.arange(self.num_nodes_)
        else:
            center_indices = np.where(self.is_weight_)[0]

        # Assign a unique integer label to each connected component in the current graph.
        subG = self.G.subgraph(range(self.num_nodes_))
        labels = np.zeros(self.num_nodes_, dtype=int)
        for lbl, comp in enumerate(nx.connected_components(subG), start=1):
            for n in comp:
                labels[n] = lbl
        self.label_cluster_ = self.labels_ = labels

        if center_indices.size == 0:
            return np.array([], dtype=int)

        bs = self.buffer_std_[center_indices, 0] if self.buffer_std_ is not None else np.ones(center_indices.size)
        bs[~np.isfinite(bs) | (bs <= 0.0)] = 1.0e-6
        alpha_vec = 1.0 / bs

        distances = cdist(samples, self.weight_[center_indices, :], metric='euclidean')
        S = 1.0 / (1.0 + distances * alpha_vec.reshape(1, -1))
        winner_idx = np.argmax(S, axis=1)
        predicted_labels = labels[center_indices[winner_idx]]
        return predicted_labels.astype(int, copy=False)

    def _cluster_step(self, sample):
        """
        Assign a single sample to a node or create a new node; update prototypes and topology evidence.
        Notes:
          - If K<3, seed nodes quickly and mirror std for the first two nodes.
          - Otherwise perform resonance test with the vigilance threshold (primary and secondary winners).
          - Maintain candidate edges and promote to the main graph when evidence exceeds a running mean.
          - Periodically prune long edges, remove underused nodes, and demote isolated low-activity nodes.
        """
        if self.num_nodes_ < 3:
            # Create a new node immediately.
            self._add_cluster_node(sample)

            # Mirror std between first two nodes (stabilize early alpha).
            if self.num_nodes_ == 2:
                self.buffer_std_[0, 0] = self.buffer_std_[1, 0]
        else:
            # Category choice by Euclidean distance.
            distances = np.sqrt(np.sum((self.weight_ - sample) ** 2, axis=1))
            sorted_idx = np.argsort(distances)  # default argsort order

            # Precompute alpha for each node (distance → similarity scaling).
            # buffer_std_ is assumed positive; guard elsewhere during updates.
            alpha = 1.0 / self.buffer_std_[:, 0]
            all_sims = 1.0 / (1.0 + alpha * distances)

            # Direct two-nearest decision without while-loop
            s1 = sorted_idx[0]
            s2 = sorted_idx[1]
            sim_s1 = all_sims[s1]
            sim_s2 = all_sims[s2]

            if sim_s1 < self.similarity_threshold_:
                # No node satisfies vigilance: create a new node.
                self._add_cluster_node(sample)
            else:
                # Update s1 node (prototype and usage).
                self.cluster_counts_[s1] += 1
                self.weight_[s1, :] = self.weight_[s1, :] + (1.0 / self.cluster_counts_[s1]) * (sample[0, :] - self.weight_[s1, :])
                self.G.nodes[s1]['weight'] = self.weight_[s1]

                # Update per-node std (Welford).
                _, current_std = self._update_online_variance(sample)
                self.buffer_std_[s1, 0] = max(np.max(current_std), 1.0e-6)

                # If the second nearest also resonates, slightly update it and manage edge evidence.
                if sim_s2 > self.similarity_threshold_:
                    self.cluster_counts_[s2] += 1
                    self.weight_[s2, :] += (1.0 / (self.cluster_counts_[s1] + self.cluster_counts_[s2])) * (sample[0, :] - self.weight_[s2, :])
                    self.G.nodes[s2]['weight'] = self.weight_[s2]

                    valid_counts = self.cluster_counts_[self.cluster_counts_ > 1]
                    count_node_threshold = np.mean(valid_counts) if valid_counts.size > 0 else np.nan

                    if (self.cluster_counts_[s1] > count_node_threshold) and (self.cluster_counts_[s2] > count_node_threshold):
                        self.is_weight_[s1] = True
                        self.is_weight_[s2] = True

                        # Candidate edge updates → threshold by mean of positive candidate counts.
                        self._increment_candidate_edge(s1, s2)
                        vals_over = self._candidate_edge_values()
                        if vals_over.size > 0:
                            edge_threshold = np.mean(vals_over)
                            if self._get_candidate_edge_count(s1, s2) > edge_threshold:
                                self._increment_main_edge(s1, s2)

        # Periodic maintenance every lambda_ samples (including at 0).
        if (self.sample_counts_ % self.lambda_) == 0:
            self._prune_long_edges()
            if not np.all(self.cluster_counts_ == 1):
                self._remove_underutilized_nodes()
            self._demote_isweight()

    def _calculate_lambda_similarity_threshold(self, buffer_std, buffer1, current_idx, data_size, num_nodes, num_clusters, finalize_on_exhaustion=True, suppress_update=False, ratioNC_in=None):
        """
        Decide new (Lambda, similarity threshold) from buffered data.
        Stability: build inverse-distance similarity matrix S=1/(1+α d); use Cholesky and tiny determinant threshold (1e-6).
        Side effects: when suppress_update=True, do not write to past_ratioNC_ (use ratioNC_in).
        """
        Lambda_new = None
        similarityTh_new = None
        isRenewed = False

        if buffer1 is None or buffer1.size == 0 or buffer1.shape[0] < 2:
            return (Lambda_new, similarityTh_new, isRenewed, self.past_ratioNC_ if ratioNC_in is None else ratioNC_in)

        # Normalize buffer_std (scalar or array) and compute alpha.
        if buffer_std is None:
            max_std = 1.0
        elif isinstance(buffer_std, (int, float, np.floating)):
            max_std = float(buffer_std)
        else:
            max_std = float(np.max(buffer_std))
        alpha = 1.0 / max(max_std, 1.0e-6)

        _, inv_dist_mat = self.compute_inverse_distance_matrix(buffer1, alpha)

        # Cholesky-based stability check.
        unstable = False
        try:
            L = np.linalg.cholesky(inv_dist_mat)
            det_approx = float((np.prod(np.diag(L))) ** 2)  # Use Cholesky factor to approximate determinant
            if det_approx < 1.0e-6:
                unstable = True
        except np.linalg.LinAlgError:
            # Non-PD: declare instability without determinant fallback
            unstable = True

        ratioNC_local = self.past_ratioNC_ if ratioNC_in is None else ratioNC_in

        if unstable or (finalize_on_exhaustion and (current_idx == data_size)):
            Lambda_new = int(buffer1.shape[0])
            n = inv_dist_mat.shape[0]
            # Discard self-similarity for row maxima.
            inv_dist_mat[np.arange(n), np.arange(n)] = -np.inf
            max_values = np.max(inv_dist_mat, axis=1)
            if (num_nodes is None) or (num_clusters is None):
                similarityTh_new = float(np.mean(max_values))
            else:
                similarityTh_new, ratioNC_local = self._candidate_similarity_threshold(
                    num_clusters, num_nodes, max_values,
                    max(int(self.lambda_) if self.lambda_ is not None else 1, 1),
                    suppress_update=suppress_update, ratioNC_in=ratioNC_local
                )
            isRenewed = True

        return (Lambda_new, similarityTh_new, isRenewed, ratioNC_local)

    def _calculate_lambda_decremental_direction(self, buffer_std, Lambda, similarity_th, buffer1, num_nodes, num_clusters):
        """
        Try reducing lambda_ when inverse-distance stability degrades.
        Strategy:
          - Reverse the buffer (most recent first), factor once by Cholesky, and inspect growing principal minors.
          - If full Cholesky fails, test each prefix by Cholesky only (no determinant fallback).
          - Stop at first instability; return last valid window and vigilance threshold.
        """
        isIncremental = True
        lastValidLambda = int(Lambda)
        lastValidsimilarityTh = float(similarity_th)

        # Exit when buffer is too small.
        if buffer1 is None or buffer1.size == 0 or buffer1.shape[0] < 2:
            return (lastValidLambda, lastValidsimilarityTh, isIncremental, self.past_ratioNC_)

        max_std = float(np.max(buffer_std)) if (buffer_std is not None and np.size(buffer_std) > 0) else 1.0
        alpha = 1.0 / max(max_std, 1.0e-6)

        buffer1_reversed = buffer1[::-1]
        dist_mat, inv_dist_mat_full = self.compute_inverse_distance_matrix(buffer1_reversed, alpha)

        # Try Cholesky factorization for the full matrix.
        full_pd = True
        try:
            L = np.linalg.cholesky(inv_dist_mat_full)
        except np.linalg.LinAlgError:
            full_pd = False

        startN = 2
        endN = buffer1.shape[0]
        ratioNC_local = self.past_ratioNC_

        if full_pd:
            # Inspect leading principal minors using the full Cholesky factor.
            for currentN in range(startN, endN + 1):
                subDetValue = float((np.prod(np.diag(L[:currentN, :currentN]))) ** 2)
                if subDetValue < 1.0e-6:
                    return (lastValidLambda, lastValidsimilarityTh, False, ratioNC_local)
                lastValidLambda = currentN
                subMat = inv_dist_mat_full[0:currentN, 0:currentN].copy()
                np.fill_diagonal(subMat, -np.inf)
                max_values = np.max(subMat, axis=1)
                lastValidsimilarityTh, ratioNC_local = self._candidate_similarity_threshold(
                    num_clusters, num_nodes, max_values, max(int(self.lambda_), 1),
                    suppress_update=True, ratioNC_in=ratioNC_local
                )
        else:
            # Full Cholesky failed: test each prefix using Cholesky only.
            for currentN in range(startN, endN + 1):
                subDistMat = dist_mat[0:currentN, 0:currentN]
                invDistMat = 1.0 / (1.0 + alpha * subDistMat)
                try:
                    Lsub = np.linalg.cholesky(invDistMat)
                    subDetValue = float((np.prod(np.diag(Lsub))) ** 2)
                    if subDetValue < 1.0e-6:
                        return (lastValidLambda, lastValidsimilarityTh, False, ratioNC_local)
                except np.linalg.LinAlgError:
                    # Non-PD prefix: declare instability at currentN
                    return (lastValidLambda, lastValidsimilarityTh, False, ratioNC_local)
                lastValidLambda = currentN
                np.fill_diagonal(invDistMat, -np.inf)
                max_values = np.max(invDistMat, axis=1)
                lastValidsimilarityTh, ratioNC_local = self._candidate_similarity_threshold(
                    num_clusters, num_nodes, max_values, max(int(self.lambda_), 1),
                    suppress_update=True, ratioNC_in=ratioNC_local
                )

        return (lastValidLambda, lastValidsimilarityTh, isIncremental, ratioNC_local)

    def _calculate_lambda_incremental_direction(self, bufferStd, Lambda, similarityTh, numNodes, numClusters):
        """
        Increase lambda_ causally using only past samples up to 2*lambda_.
        Strategy:
          - Exponential search to bracket the first renewal trigger.
          - Adopt the last stable length immediately before the trigger (no lookahead).
        """
        Lambda = int(Lambda)
        Lambda_new = int(Lambda)
        similarityTh_new = float(similarityTh)

        availPast = int(self.buffer_data_count_)
        if availPast <= Lambda:
            return Lambda_new, similarityTh_new, self.past_ratioNC_

        endIndex = min(2 * Lambda, availPast)

        # Exponential search to bracket first triggering k.
        step = 1
        lowK = Lambda
        highK = min(Lambda + step, endIndex)
        found = False
        ratioNC_local = self.past_ratioNC_

        while highK <= endIndex:
            extend_buffer = self._get_from_buffer(highK)
            Lcand, Scand, ok, ratioNC_local = self._calculate_lambda_similarity_threshold(
                bufferStd, extend_buffer, highK, endIndex, numNodes, numClusters,
                finalize_on_exhaustion=False, suppress_update=True, ratioNC_in=ratioNC_local
            )
            if ok:
                found = True
                break  # [lowK, highK] brackets the first trigger
            else:
                lowK = highK
                step = min(step * 2, endIndex - Lambda)
                highK = min(Lambda + step, endIndex)
                if highK <= lowK:
                    break

        if found:
            # Adopt the last stable k (= lowK).
            Lambda_new = int(lowK)
            stable_buffer = self._get_from_buffer(lowK)
            # Recompute similarity threshold on a buffer of length lowK.
            max_std = float(np.max(bufferStd)) if (bufferStd is not None and np.size(bufferStd) > 0) else 1.0
            alpha = 1.0 / max(max_std, 1.0e-6)
            _, inv_dist_mat = self.compute_inverse_distance_matrix(stable_buffer, alpha)
            np.fill_diagonal(inv_dist_mat, -np.inf)
            max_values = np.max(inv_dist_mat, axis=1)
            if (numNodes is None) or (numClusters is None):
                similarityTh_new = float(np.mean(max_values))
            else:
                similarityTh_new, ratioNC_local = self._candidate_similarity_threshold(
                    numClusters, numNodes, max_values, lowK,
                    suppress_update=True, ratioNC_in=ratioNC_local
                )
            return Lambda_new, similarityTh_new, ratioNC_local
        else:
            # Finalize at the cap.
            cap_buffer = self._get_from_buffer(endIndex)
            Lcap, Scap, _, ratioNC_local = self._calculate_lambda_similarity_threshold(
                bufferStd, cap_buffer, endIndex, endIndex, numNodes, numClusters,
                finalize_on_exhaustion=True, suppress_update=True, ratioNC_in=ratioNC_local
            )
            return int(Lcap), float(Scap), ratioNC_local

    def _candidate_similarity_threshold(self, num_clusters, num_nodes, max_values, Lambda, suppress_update=False, ratioNC_in=None):
        """
        Compute threshold from row-wise maxima and the cluster/node ratio.
        Mixing: ratioNC ← (1/Lambda)*current + (1-1/Lambda)*history.
        Side effects: when suppress_update=True, do not update past_ratioNC_.
        """
        current_ratioNC = float(num_clusters) / float(num_nodes) if num_nodes > 0 else 0.0
        mixing_ratio = 1.0 / float(Lambda) if Lambda != 0 else 0.0
        base_ratio = self.past_ratioNC_ if ratioNC_in is None else ratioNC_in
        mixed_ratioNC = mixing_ratio * current_ratioNC + (1.0 - mixing_ratio) * base_ratio

        if not suppress_update:
            self.past_ratioNC_ = mixed_ratioNC

        qValue = (1.0 - mixed_ratioNC)
        # Use Hazen interpolation to match Matlab-like quantile behavior.
        new_threshold = np.quantile(max_values,  qValue, method='hazen')
        return new_threshold, mixed_ratioNC

    def _prune_long_edges(self):
        """
        Remove edges whose lengths are outliers based on IQR; reset candidate counts for those pairs.
        IQR quantiles are computed with Hazen interpolation to mirror Matlab quantile behavior.
        """
        edges = list(self.G.edges())
        if not edges:
            return

        # Compute Euclidean distance for each edge.
        distances = np.array([
            np.linalg.norm(self.weight_[u] - self.weight_[v])
            for u, v in edges
        ])

        # Identify outlier edges via interquartile range (Hazen quantiles).
        Q1 = np.quantile(distances, 0.25, method='hazen')
        Q3 = np.quantile(distances, 0.75, method='hazen')
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR

        # Remove edges exceeding the upper bound and reset candidate counters.
        for (u, v), dist in zip(edges, distances):
            if dist > upper_bound and self.G.has_edge(u, v):
                self.G.remove_edge(u, v)
                if self.candidate_G.has_edge(u, v):
                    self.candidate_G.remove_edge(u, v)

    def _remove_underutilized_nodes(self):
        """
        Remove nodes used only once when their count exceeds lambda_; keep at most lambda_ such nodes.
        Rebuild graphs and state arrays consistently after removal.
        """
        if self.cluster_counts_ is None or self.cluster_counts_.size == 0:
            return
        candidates = np.where(self.cluster_counts_ == 1)[0]
        if len(candidates) == 0:
            return

        # Determine how many extra nodes beyond lambda_ to remove.
        if len(candidates) > int(self.lambda_):
            nodes_to_remove = candidates[:len(candidates) - int(self.lambda_)]
        else:
            return

        # Backup original arrays.
        orig_weight = self.weight_.copy()
        orig_cluster_counts = self.cluster_counts_.copy()
        orig_buffer_std = self.buffer_std_.copy()
        orig_is_weight = None if (self.is_weight_ is None) else self.is_weight_.copy()

        # Build mask and mapping from old indices to new indices.
        mask = np.ones(self.num_nodes_, dtype=bool)
        mask[nodes_to_remove] = False
        kept_indices = np.where(mask)[0]
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_indices)}

        # Apply mask to node-specific arrays.
        self.weight_ = orig_weight[kept_indices, :]
        self.cluster_counts_ = orig_cluster_counts[kept_indices]
        self.buffer_std_ = orig_buffer_std[kept_indices, :]
        if orig_is_weight is None:
            self.is_weight_ = np.zeros(self.weight_.shape[0], dtype=bool)
        else:
            self.is_weight_ = orig_is_weight[kept_indices]
        self.num_nodes_ = self.weight_.shape[0]

        if self.prev_cluster_counts_ is not None and self.prev_cluster_counts_.shape[0] > 0:
            self.prev_cluster_counts_ = self.prev_cluster_counts_[kept_indices]
        if self.prev_edge_row_sum_ is not None and self.prev_edge_row_sum_.shape[0] > 0:
            self.prev_edge_row_sum_ = self.prev_edge_row_sum_[kept_indices]
        if self.demotion_tolerance_count_ is not None and self.demotion_tolerance_count_.shape[0] > 0:
            self.demotion_tolerance_count_ = self.demotion_tolerance_count_[kept_indices]

        # Rebuild the main graph with updated node indices using subgraph + relabel.
        kept_set = set(kept_indices.tolist())
        new_G = self.G.subgraph(kept_set).copy()
        new_G = nx.relabel_nodes(new_G, old_to_new, copy=True)
        # Ensure per-node 'weight' attribute is up-to-date.
        for new_idx in range(self.num_nodes_):
            new_G.nodes[new_idx]['weight'] = self.weight_[new_idx]
        self.G = new_G

        # Rebuild candidate_G similarly.
        if hasattr(self, 'candidate_G'):
            new_candidate = self.candidate_G.subgraph(kept_set).copy()
            new_candidate = nx.relabel_nodes(new_candidate, old_to_new, copy=True)
            self.candidate_G = new_candidate

    def _update_online_variance(self, new_samples):
        """
        Incrementally update mean and variance using Welford's algorithm.
        Returns: (self, current_std) where current_std is feature-wise std after update.
        """
        if new_samples.ndim == 1:
            new_samples = new_samples.reshape(1, -1)
        num_new = new_samples.shape[0]

        if num_new == 0:
            if self.mean_data_ is None or self.mean_data_.size == 0:
                num_features = new_samples.shape[1] if new_samples.ndim == 2 else 0
                current_std = np.zeros(num_features, dtype=np.float64)
            elif self.count_data_ <= 1:
                current_std = np.zeros_like(self.mean_data_)
            else:
                variance = self.M2_data_ / (self.count_data_ - 1)
                variance[variance < 0] = 0
                current_std = np.sqrt(variance)
            return (self, current_std)

        if self.mean_data_ is None or self.mean_data_.size == 0:
            self.mean_data_ = np.mean(new_samples, axis=0)
            self.M2_data_ = np.zeros_like(self.mean_data_)
            self.count_data_ = num_new
            current_std = np.zeros_like(self.mean_data_)
        else:
            new_count = self.count_data_ + num_new
            delta = new_samples - self.mean_data_
            self.mean_data_ = self.mean_data_ + np.sum(delta, axis=0) / new_count
            self.M2_data_ = self.M2_data_ + np.sum(
                delta * (new_samples - self.mean_data_), axis=0
            )
            self.count_data_ = new_count
            if self.count_data_ <= 1:
                current_std = np.zeros_like(self.mean_data_)
            else:
                variance = self.M2_data_ / (self.count_data_ - 1)
                variance[variance < 0] = 0
                current_std = np.sqrt(variance)
        return (self, current_std)

    def _reset_online_variance(self):
        """
        Reset accumulated statistics for online variance computation.
        """
        self.mean_data_ = None
        self.M2_data_ = None
        self.count_data_ = 0

    def _append_to_buffer(self, new_data):
        """
        Append new_data to the ring buffer.
        Policy:
          - Expand capacity with headroom only; never shrink (amortized O(1) appends).
          - After insertion, trim logically to keep only the last buffer_keep_len_ samples.
        """
        if new_data.ndim == 1:
            new_data = new_data.reshape(1, -1)
        num_new, num_cols = new_data.shape

        if self.buffer_data_count_ == 0:
            # Initial capacity with headroom against current keep_len.
            initial_capacity = max(
                500,
                num_new,
                int(np.ceil(2 * max(1, self.buffer_keep_len_)))
            )
            self.buffer_data_capacity_ = initial_capacity
            self.buffer_data_ = np.zeros((initial_capacity, num_cols), dtype=np.float64)
            self.buffer_data_start_ = 0
            self.buffer_data_count_ = 0

        required_capacity = self.buffer_data_count_ + num_new
        if required_capacity > self.buffer_data_capacity_:
            # Expand to max of: double current, double required, or 2*keep_len.
            new_capacity = max(
                self.buffer_data_capacity_ * 2,
                required_capacity * 2,
                int(np.ceil(2 * max(1, self.buffer_keep_len_)))
            )
            new_buffer = np.zeros((new_capacity, num_cols), dtype=np.float64)
            current_data = self._get_from_buffer()
            current_count = current_data.shape[0]
            if current_count > 0:
                new_buffer[0:current_count, :] = current_data
            self.buffer_data_ = new_buffer
            self.buffer_data_capacity_ = new_capacity
            self.buffer_data_start_ = 0
            self.buffer_data_count_ = current_count

        indices = ((self.buffer_data_start_ + np.arange(self.buffer_data_count_, self.buffer_data_count_ + num_new))
                   % self.buffer_data_capacity_)
        self.buffer_data_[indices, :] = new_data
        self.buffer_data_count_ += num_new

        # Logical trim to buffer_keep_len_.
        keep_len = int(self.buffer_keep_len_) if self.buffer_keep_len_ is not None else self.buffer_data_count_
        if keep_len <= 0 or not np.isfinite(keep_len):
            keep_len = self.buffer_data_count_
        if self.buffer_data_count_ > keep_len:
            excess = self.buffer_data_count_ - keep_len
            self.buffer_data_start_ = (self.buffer_data_start_ + excess) % self.buffer_data_capacity_
            self.buffer_data_count_ -= excess

    def _get_from_buffer(self, req_num=None):
        """
        Retrieve the latest req_num samples from the ring buffer (oldest → newest).
        Returns fewer if req_num exceeds current stored count.
        """
        if self.buffer_data_ is None:
            return np.array([])
        if req_num is None:
            req_num = self.buffer_data_count_
        if req_num > self.buffer_data_count_:
            req_num = self.buffer_data_count_
        if req_num == 0:
            return np.zeros((0, self.buffer_data_.shape[1]))

        start_idx = (self.buffer_data_start_ + self.buffer_data_count_ - req_num) % self.buffer_data_capacity_
        end_idx = start_idx + req_num

        if end_idx <= self.buffer_data_capacity_:
            data = self.buffer_data_[start_idx:end_idx, :]
        else:
            first_part = self.buffer_data_[start_idx:self.buffer_data_capacity_, :]
            second_part_count = end_idx - self.buffer_data_capacity_
            second_part = self.buffer_data_[0:second_part_count, :]
            data = np.vstack((first_part, second_part))
        return data

    def _clear_buffer(self):
        """
        Clear ring buffer contents without altering allocated capacity.
        """
        self.buffer_data_count_ = 0
        self.buffer_data_start_ = 0

    def _update_buffer_keep_len(self):
        """
        Recompute retention length after lambda_ update.
        Retention length = 2*lambda_ (at least lambda_). Perform only logical trimming.
        Proactively expand physical capacity to 2*retention if needed.
        """
        if self.lambda_ is None:
            self.buffer_keep_len_ = self.buffer_data_count_
            return
        keep_len = max(int(self.lambda_), 2 * int(self.lambda_))
        self.buffer_keep_len_ = keep_len

        # Logical trim if current stored count exceeds keep_len.
        if self.buffer_data_count_ > self.buffer_keep_len_:
            excess = self.buffer_data_count_ - self.buffer_keep_len_
            self.buffer_data_start_ = (self.buffer_data_start_ + excess) % self.buffer_data_capacity_
            self.buffer_data_count_ -= excess

        # Proactive capacity growth to reserve headroom (no shrink).
        target_capacity = int(np.ceil(2 * self.buffer_keep_len_))
        if self.buffer_data_capacity_ < target_capacity:
            if self.buffer_data_ is None:
                self.buffer_data_capacity_ = target_capacity
                return
            num_cols = self.buffer_data_.shape[1]
            new_buffer = np.zeros((target_capacity, num_cols), dtype=np.float64)
            current_data = self._get_from_buffer()
            current_count = current_data.shape[0]
            if current_count > 0:
                new_buffer[0:current_count, :] = current_data
            self.buffer_data_ = new_buffer
            self.buffer_data_capacity_ = target_capacity
            self.buffer_data_start_ = 0
            self.buffer_data_count_ = current_count

    def _add_cluster_node(self, sample):
        """
        Create a new node initialized at the given sample; leave unweighted until evidence accrues.
        Also updates per-node std from global Welford statistics (guarded by 1e-6).
        """
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
        self.num_nodes_ += 1
        idx = self.num_nodes_ - 1

        if self.weight_ is None or self.weight_ .size == 0:
            self.weight_ = sample.copy()
        else:
            self.weight_ = np.vstack([self.weight_, sample])

        if self.cluster_counts_ is None or self.cluster_counts_.size == 0:
            self.cluster_counts_ = np.array([1], dtype=np.float64)
        else:
            self.cluster_counts_ = np.hstack([self.cluster_counts_, [1.0]])

        if self.is_weight_ is None or self.is_weight_.size == 0:
            self.is_weight_ = np.array([False], dtype=bool)
        else:
            self.is_weight_ = np.hstack([self.is_weight_, [False]])

        if self.buffer_std_ is None or self.buffer_std_.size == 0:
            self.buffer_std_ = np.zeros((1, 1), dtype=np.float64)
        else:
            self.buffer_std_ = np.vstack([self.buffer_std_, np.zeros((1, 1), dtype=np.float64)])

        _, current_std = self._update_online_variance(sample)
        self.buffer_std_[idx, 0] = max(np.max(current_std), 1.0e-6)

        # Add the new node to main and candidate graphs, with weight info in main graph.
        self.G.add_node(idx)
        self.candidate_G.add_node(idx)
        self.G.nodes[idx]['weight'] = self.weight_[idx]

        # Matlab準拠：ノード追加時に3配列をゼロで同時拡張する（demote内で整合しない）
        if self.prev_cluster_counts_ is None or self.prev_cluster_counts_.size == 0:
            self.prev_cluster_counts_ = np.zeros(self.num_nodes_, dtype=float)
        else:
            self.prev_cluster_counts_ = np.hstack([self.prev_cluster_counts_, [0.0]])

        if self.prev_edge_row_sum_ is None or self.prev_edge_row_sum_.size == 0:
            self.prev_edge_row_sum_ = np.zeros(self.num_nodes_, dtype=float)
        else:
            self.prev_edge_row_sum_ = np.hstack([self.prev_edge_row_sum_, [0.0]])

        if self.demotion_tolerance_count_ is None or self.demotion_tolerance_count_.size == 0:
            self.demotion_tolerance_count_ = np.zeros(self.num_nodes_, dtype=int)
        else:
            self.demotion_tolerance_count_ = np.hstack([self.demotion_tolerance_count_, [0]])

    @staticmethod
    def compute_inverse_distance_matrix(X, alpha):
        """
        Compute pairwise Euclidean distances and inverse-distance similarities.
        Outputs:
          - dist_mat[i,j] = ||x_i - x_j||_2
          - inv_dist_mat[i,j] = 1 / (1 + alpha * dist_mat[i,j])
        Notes: Clamp small negative values before sqrt for numerical stability; O(n^2 d) time, O(n^2) memory.
        """
        XX = np.sum(X ** 2, axis=1, keepdims=True)
        D_sq = XX + XX.T - 2.0 * np.dot(X, X.T)
        D_sq[D_sq < 0] = 0.0
        dist_mat = np.sqrt(D_sq)
        inv_dist_mat = 1.0 / (1.0 + alpha * dist_mat)
        return (dist_mat, inv_dist_mat)

    def _increment_candidate_edge(self, u, v):
        """
        Increase candidate edge count; create edge with count=1 if absent (undirected).
        """
        if self.candidate_G.has_edge(u, v):
            self.candidate_G[u][v]['count'] = self.candidate_G[u][v].get('count', 0) + 1
        else:
            self.candidate_G.add_edge(u, v, count=1)

    def _candidate_edge_values(self):
        """
        Return counts of all positive candidate edges (>=1).
        """
        vals = []
        for (u, v, d) in self.candidate_G.edges(data=True):
            c = d.get('count', 0)
            if c > 0:
                vals.append(c)
        return np.array(vals)

    def _get_candidate_edge_count(self, u, v):
        """
        Return how many times a candidate edge has been incremented.
        """
        if self.candidate_G.has_edge(u, v):
            return self.candidate_G[u][v].get('count', 0)
        return 0.0

    def _increment_main_edge(self, u, v):
        """
        Create or increase weight of an edge in the main graph based on candidate evidence.
        """
        if self.G.has_edge(u, v):
            w_val = self.G[u][v].get('weight', 0)
            self.G[u][v]['weight'] = w_val + 1
        else:
            self.G.add_edge(u, v, weight=1)

    def _demote_isweight(self):
        """
        Demote is_weight=True nodes that stay isolated (degree==0) and comparatively inactive.
        Policy: maintain a tolerance counter and demote when it reaches ≈ max(1, round(buffer_keep_len_/lambda_)).
        Note: Matlab準拠のため，ここでは整合サイズ調整を行わない（追加／削除時にのみ整合を取る）．
        """
        K = self.num_nodes_
        if K == 0:
            return

        # Degree per node in the main graph.
        deg_dict = dict(self.G.degree(range(K)))
        degree = np.fromiter((deg_dict.get(i, 0) for i in range(K)), dtype=int, count=K)

        usage_count = self.cluster_counts_.astype(float)
        delta_usage = usage_count - self.prev_cluster_counts_

        # Row-sum of candidate co-activations.
        coact_row_sum = self._candidate_row_sums()
        delta_coact = coact_row_sum - self.prev_edge_row_sum_

        # Activity in this maintenance interval.
        activity_score = delta_usage + delta_coact

        # If no positive activity, reset tolerance and snapshots.
        has_activity = activity_score > 0
        if not np.any(has_activity):
            self.prev_cluster_counts_ = usage_count.copy()
            self.prev_edge_row_sum_ = coact_row_sum.copy()
            self.demotion_tolerance_count_.fill(0)
            return

        # Robust lower bound via Tukey's rule on positive activities (Hazen quantiles).
        activity_vals = activity_score[has_activity]
        Q1 = np.quantile(activity_vals, 0.25, method='hazen')
        Q3 = np.quantile(activity_vals, 0.75, method='hazen')
        IQR = Q3 - Q1
        lower_bound = max(Q1 - 1.5 * IQR, 0.0)

        # Candidates: is_weight & degree==0 & low activity.
        if self.is_weight_ is None or self.is_weight_.size == 0:
            is_weighted = np.zeros(K, dtype=bool)
        else:
            is_weighted = self.is_weight_.astype(bool)
        is_candidate = is_weighted & (degree == 0) & (activity_score <= lower_bound)

        # Update tolerance counters; reset when condition not met.
        self.demotion_tolerance_count_[is_candidate] += 1
        self.demotion_tolerance_count_[~is_candidate] = 0

        # Threshold derived from retention.
        # denom = int(self.lambda_) if (self.lambda_ is not None and int(self.lambda_) > 0) else 1
        # demotion_threshold = max(1, round(max(1, int(self.buffer_keep_len_)) / max(1, denom)))
        demotion_threshold = 2

        demote_idx = np.where(self.demotion_tolerance_count_ >= demotion_threshold)[0]
        if (demote_idx.size > 0) and (self.is_weight_ is not None):
            self.is_weight_[demote_idx] = False
            self.demotion_tolerance_count_[demote_idx] = 0

        # Roll snapshots forward.
        self.prev_cluster_counts_ = usage_count.copy()
        self.prev_edge_row_sum_ = coact_row_sum.copy()

    def _candidate_row_sums(self):
        """
        Sum candidate co-activation counts per node (undirected).
        """
        K = self.num_nodes_
        if K == 0 or self.candidate_G.number_of_edges() == 0:
            return np.zeros(K, dtype=float)
        # Use weighted degree (weight='count') to sum 'count' over incident edges.
        deg_w = dict(self.candidate_G.degree(range(K), weight='count'))
        return np.array([float(deg_w.get(i, 0.0)) for i in range(K)], dtype=float)

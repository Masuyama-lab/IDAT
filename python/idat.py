# Copyright (c) 2025-2026 Naoki Masuyama
# SPDX-License-Identifier: MIT

"""
IDAT: Inverse Distance-based ART with Topology.

    model = IDAT()               # a new model, with no hyperparameter to set
    model.fit(X)                 # learn the rows of X one at a time, in the given order
    labels = model.predict(Y)    # cluster label of every row of Y, as an array of length len(Y)

fit can be called again with new samples. The model keeps learning from where it stopped, so a
data stream can be given chunk by chunk. fit_predict(X) is fit(X) followed by predict(X).

How it works (one sample x at a time)
  Category choice   Every node k has a prototype w_k and a scale sigma_k. The similarity of x to
                    node k is s_k = 1 / (1 + ||x - w_k|| / sigma_k). The two most similar nodes
                    are s1 and s2. The first three samples become nodes.
  Vigilance test    V is the similarity threshold (the vigilance parameter).
                      s_1 <  V           : x becomes a new node.
                      s_1 >= V, s_2 <= V : w_s1 moves towards x.
                      s_1 >= V, s_2 >  V : w_s1 and w_s2 move towards x. When both nodes have
                    won more often than the average of the nodes that have won at least twice,
                    both become active (is_weight_) and their co-activation count
                    E^cand(s1, s2) grows. An edge is created once that count exceeds the average
                    of the positive co-activation counts.
  Maintenance       Every Lambda samples: edges longer than Q3 + 1.5 IQR of the edge lengths are
                    removed, nodes that have never won since their creation are removed except
                    the newest Lambda of them, and an active node that is isolated and has
                    stayed idle for two maintenances becomes inactive.
  Adaptation        Every Lambda samples, once there are more than two nodes: Lambda is set to
                    the largest number L of recent samples whose similarity matrix S_L is still
                    diverse, det(S_L) >= EPS_DET, and V is set to a quantile of the similarities
                    within these samples. The quantile level follows the ratio of the number of
                    clusters to the number of nodes, smoothed over the adaptations.
  Prediction        A sample gets the label of its most similar active node (of all nodes if
                    none is active). The labels are the connected components of the edge graph,
                    numbered from 1 in the order of their smallest node index.

Attributes (read only)
  num_nodes_                K, the number of nodes, active or not
  weight_                   [K x d] prototypes
  count_node_               [K] number of wins of every node
  buffer_std_               [K x 1] sigma_k
  is_weight_                [K] True for the active nodes, the nodes used by predict
  edge_                     [K x K] scipy.sparse matrix, edge_[i, j] > 0 when nodes i and j
                            are connected
  edge_candidate_counter_   [K x K] scipy.sparse matrix, co-activation counts E^cand
  lambda_                   current window length and maintenance interval
  similarity_threshold_     current similarity threshold V
  label_cluster_            [K] cluster label of every node, set by predict
  sample_counts_            number of samples learned so far

Network size (the three counts printed by the main scripts)
  all nodes      num_nodes_
  active nodes   np.count_nonzero(is_weight_)
  clusters       np.unique(label_cluster_[is_weight_]).size after predict, that is,
                 the clusters that contain at least one active node

The class constants are the values used in the paper. They are not meant to be tuned.

Requires numpy, scipy and scikit-learn.

Reference
  N. Masuyama, Y. Toda, Y. Nojima, and H. Ishibuchi, "An adaptive resonance theory-based
  topological clustering algorithm with a self-adjusting vigilance parameter,"
  arXiv:2511.17983, 2025. https://arxiv.org/abs/2511.17983
"""

import numpy as np
from scipy.linalg import lapack
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.base import BaseEstimator, ClusterMixin


class IDAT(BaseEstimator, ClusterMixin):
    LAMBDA_INIT = 2  # initial Lambda
    EPS_DET = 1.0e-6  # a window of L samples is diverse while det(S_L) >= EPS_DET
    IQR_FACTOR = 1.5  # c of the bounds Q3 + c IQR (edge pruning) and Q1 - c IQR (demotion)
    DEMOTE_TOLERANCE = 2  # idle maintenances before an isolated active node is demoted
    BUFFER_FACTOR = 2  # the buffer keeps the newest BUFFER_FACTOR * Lambda samples
    SIGMA_MIN = 1.0e-6  # lower bound of sigma_k

    # arrays with one entry per node: they have spare rows, and the first num_nodes_ are used
    _NODE_ARRAYS = (
        "_w",
        "_count",
        "_sigma",
        "_active",
        "_degree",
        "_cand_row_sum",
        "_prev_count",
        "_prev_cand_row_sum",
        "_idle",
        "_id",
    )
    _PAIR_ARRAYS = ("_pair_i", "_pair_j", "_pair_edge", "_pair_cand")

    def __init__(self):
        self.num_nodes_ = 0
        self.lambda_ = self.LAMBDA_INIT
        self.similarity_threshold_ = 0.0
        self.label_cluster_ = None
        self.sample_counts_ = 0

        self._recalc_counter = 0  # samples learned since the last adaptation
        self._ratio_nc = 0.0  # smoothed ratio of the number of clusters to the number of nodes

        # nodes (allocated at the first sample)
        self._w = None  # prototypes
        self._count = None  # number of wins
        self._sigma = None  # sigma_k
        self._active = None  # True for the active nodes
        self._degree = None  # number of edges of every node
        self._cand_row_sum = None  # row sums of E^cand
        self._prev_count = None  # _count at the last maintenance
        self._prev_cand_row_sum = None  # _cand_row_sum at the last maintenance
        self._idle = None  # consecutive maintenances as a demotion candidate
        self._id = None  # creation number of every node, kept when other nodes are removed
        self._next_id = 0

        # node pairs (i < j) that have been co-activated, with their edge count and E^cand.
        # A pair is looked up by the creation numbers of its two nodes.
        self._pair_slot = {}
        self._num_pairs = 0
        self._pair_i = np.zeros(0, dtype=np.int64)
        self._pair_j = np.zeros(0, dtype=np.int64)
        self._pair_edge = np.zeros(0)
        self._pair_cand = np.zeros(0)

        # running sums behind the two thresholds of the vigilance test (over the symmetric matrix)
        self._sum_count_above1 = 0  # sum of the counts > 1
        self._num_count_above1 = 0  # number of the counts > 1
        self._sum_cand = 0  # sum of the positive entries of E^cand
        self._num_cand = 0  # number of these entries

        self._num_components = 0  # number of connected components (valid when not stale)
        self._components_stale = True

        # running mean, second moment and count of the learned samples (Welford)
        self._run_mean = None
        self._run_m2 = None
        self._run_count = 0

        # ring buffer of the most recent samples
        self._buf = None
        self._buf_start = 0  # row of the oldest kept sample
        self._buf_count = 0  # number of kept samples
        self._buf_keep = 0  # samples to keep (0 before the first adaptation: all)

    # ------------------------------------------------------------------------------------------
    # Learning and prediction
    # ------------------------------------------------------------------------------------------
    def fit(self, X, y=None):
        """Learn the rows of X one at a time, in the given order. y is ignored."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        for x in X:
            self._append_to_buffer(x)
            self._learn_one_sample(x)
            self.sample_counts_ += 1
            self._recalc_counter += 1
            if self._recalc_counter >= self.lambda_ and self.num_nodes_ > 2:
                self._adapt_lambda_and_threshold()
                self._recalc_counter = 0
                self._update_buffer_keep_len()
        return self

    def predict(self, X):
        """Cluster label of every row of X, as an integer array of length len(X)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        K = self.num_nodes_
        if K == 0:
            self.label_cluster_ = np.zeros(0, dtype=int)
            return np.zeros(0, dtype=int)
        _, labels = connected_components(self._edge_graph(), directed=False)
        labels = labels.astype(int) + 1
        self.label_cluster_ = labels
        nodes = np.flatnonzero(self._active[:K])
        if nodes.size == 0:
            nodes = np.arange(K)
        alpha = 1.0 / self._sigma[nodes]
        W = self._w[nodes]
        out = np.empty(X.shape[0], dtype=int)
        chunk = max(1, 4_000_000 // nodes.size)  # bounds the size of the [nodes x chunk] matrices
        for start in range(0, X.shape[0], chunk):
            Y = X[start : start + chunk]
            D = cdist(W, Y)
            S = 1.0 / (1.0 + alpha[:, None] * D)
            out[start : start + Y.shape[0]] = labels[nodes[np.argmax(S, axis=0)]]
        return out

    def fit_predict(self, X, y=None):
        """fit(X) followed by predict(X)."""
        return self.fit(X).predict(X)

    # ------------------------------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------------------------------
    @property
    def weight_(self):
        return None if self._w is None else self._w[: self.num_nodes_]

    @property
    def count_node_(self):
        return None if self._count is None else self._count[: self.num_nodes_]

    @property
    def buffer_std_(self):
        return None if self._sigma is None else self._sigma[: self.num_nodes_].reshape(-1, 1)

    @property
    def is_weight_(self):
        return None if self._active is None else self._active[: self.num_nodes_]

    @property
    def edge_(self):
        return self._pair_matrix(self._pair_edge)

    @property
    def edge_candidate_counter_(self):
        return self._pair_matrix(self._pair_cand)

    def _pair_matrix(self, values):
        """Symmetric [K x K] sparse matrix of the positive values of the node pairs."""
        n = self._num_pairs
        keep = values[:n] > 0
        i, j, v = self._pair_i[:n][keep], self._pair_j[:n][keep], values[:n][keep]
        K = self.num_nodes_
        return csr_matrix(
            (np.concatenate([v, v]), (np.concatenate([i, j]), np.concatenate([j, i]))),
            shape=(K, K),
        )

    def _edge_graph(self):
        """[K x K] sparse matrix with one entry for every edge."""
        n = self._num_pairs
        has_edge = self._pair_edge[:n] > 0
        i, j = self._pair_i[:n][has_edge], self._pair_j[:n][has_edge]
        K = self.num_nodes_
        return csr_matrix((np.ones(i.size), (i, j)), shape=(K, K))

    # ------------------------------------------------------------------------------------------
    # One sample: category choice, vigilance test and maintenance
    # ------------------------------------------------------------------------------------------
    def _learn_one_sample(self, x):
        if self.num_nodes_ < 3:
            self._add_node(x)
            if self.num_nodes_ == 2:
                # the first node takes the scale of the first two samples
                self._sigma[0] = self._sigma[1]
        else:
            # category choice: the two nearest nodes (argmin returns the first of equal distances)
            distances = np.linalg.norm(self._w[: self.num_nodes_] - x, axis=1)
            s1 = int(np.argmin(distances))
            d1 = distances[s1]
            distances[s1] = np.inf
            s2 = int(np.argmin(distances))
            d2 = distances[s2]
            sim1 = 1.0 / (1.0 + (1.0 / self._sigma[s1]) * d1)
            sim2 = 1.0 / (1.0 + (1.0 / self._sigma[s2]) * d2)
            V = self.similarity_threshold_
            if sim1 < V:
                self._add_node(x)  # Case I
            else:
                self._increment_count(s1)  # Case II
                w = self._w[s1]
                w += (1.0 / self._count[s1]) * (x - w)
                self._sigma[s1] = max(self._update_running_std(x), self.SIGMA_MIN)
                if sim2 > V:  # Case III
                    self._increment_count(s2)
                    w = self._w[s2]
                    w += (1.0 / (self._count[s1] + self._count[s2])) * (x - w)
                    self._update_topology(s1, s2)
        # maintenance every Lambda samples
        # (sample_counts_ is the number of samples learned before x)
        if self.sample_counts_ % self.lambda_ == 0:
            self._maintain()

    def _update_topology(self, s1, s2):
        """Activation of s1 and s2, their co-activation count and, possibly, an edge (Case III)."""
        count_th = self._sum_count_above1 / self._num_count_above1  # mean of the counts > 1
        if self._count[s1] > count_th and self._count[s2] > count_th:
            self._active[s1] = True
            self._active[s2] = True
            slot = self._find_pair(s1, s2)
            c = self._pair_cand[slot]
            self._pair_cand[slot] = c + 1
            self._cand_row_sum[s1] += 1
            self._cand_row_sum[s2] += 1
            self._sum_cand += 2
            if c == 0:
                self._num_cand += 2
            edge_th = self._sum_cand / self._num_cand  # mean of the positive entries of E^cand
            if c + 1 > edge_th:
                if self._pair_edge[slot] == 0:
                    self._components_stale = True
                    self._degree[s1] += 1
                    self._degree[s2] += 1
                self._pair_edge[slot] += 1

    def _maintain(self):
        K = self.num_nodes_
        # 1) edge pruning: Tukey's rule on the edge lengths
        n = self._num_pairs
        e = np.flatnonzero(self._pair_edge[:n] > 0)
        if e.size > 0:
            length = np.linalg.norm(self._w[self._pair_i[e]] - self._w[self._pair_j[e]], axis=1)
            both = np.concatenate([length, length])  # every edge counts as (i, j) and (j, i)
            Q1, Q3 = _quantile(both, [0.25, 0.75])
            IQR = Q3 - Q1
            too_long = length > Q3 + self.IQR_FACTOR * IQR
            if np.any(too_long):
                self._pair_edge[e[too_long]] = 0
                self._pair_cand[e[too_long]] = 0
                self._refresh_candidate_sums()
                self._components_stale = True
                pi, pj = self._pair_i[:n], self._pair_j[:n]
                has_edge = self._pair_edge[:n] > 0
                self._degree[:K] = np.bincount(pi[has_edge], minlength=K) + np.bincount(
                    pj[has_edge], minlength=K
                )
                cand = self._pair_cand[:n]
                self._cand_row_sum[:K] = np.bincount(pi, weights=cand, minlength=K) + np.bincount(
                    pj, weights=cand, minlength=K
                )
                if n - self._num_cand // 2 > max(1024, n // 2):  # many pruned pairs: drop them
                    self._compact_pairs()
        # 2) removal of the nodes that have never won: the newest Lambda of them are kept
        num_count1 = K - self._num_count_above1
        if self._num_count_above1 > 0 and num_count1 > self.lambda_:
            r = np.flatnonzero(self._count[:K] == 1)
            self._remove_nodes(r[: r.size - self.lambda_])
        # 3) demotion of the isolated active nodes that stay idle
        self._demote_idle_nodes()

    def _remove_nodes(self, r):
        """The nodes r have count 1, so they have no edge and no co-activation. Every one of them
        is an isolated component, and the degrees and row sums of the other nodes and the sums
        behind the thresholds do not change."""
        K = self.num_nodes_
        keep = np.ones(K, dtype=bool)
        keep[r] = False
        for name in self._NODE_ARRAYS:
            a = getattr(self, name)
            a[: K - r.size] = a[:K][keep]
        self.num_nodes_ = K - r.size
        new_index = np.cumsum(keep) - 1
        n = self._num_pairs
        self._pair_i[:n] = new_index[self._pair_i[:n]]
        self._pair_j[:n] = new_index[self._pair_j[:n]]
        self._num_components -= r.size

    def _demote_idle_nodes(self):
        """The activity of a node is its number of wins and co-activations since the last
        maintenance. An active node that is isolated and whose activity is at most
        max(Q1 - 1.5 IQR, 0) of the positive activities for DEMOTE_TOLERANCE consecutive
        maintenances is demoted (is_weight_ becomes False)."""
        K = self.num_nodes_
        count, row_sum = self._count[:K], self._cand_row_sum[:K]
        activity = (count - self._prev_count[:K]) + (row_sum - self._prev_cand_row_sum[:K])
        self._prev_count[:K] = count
        self._prev_cand_row_sum[:K] = row_sum
        idle = self._idle[:K]
        has_activity = activity > 0
        if not np.any(has_activity):
            idle[:] = 0
            return
        isolated_active = self._active[:K] & (self._degree[:K] == 0)
        if np.any(isolated_active):
            a = activity[has_activity]
            Q1, Q3 = _quantile(a, [0.25, 0.75])
            IQR = Q3 - Q1
            candidate = isolated_active & (activity <= max(Q1 - self.IQR_FACTOR * IQR, 0.0))
        else:
            candidate = np.zeros(K, dtype=bool)
        idle[candidate] += 1
        idle[~candidate] = 0
        demote = idle >= self.DEMOTE_TOLERANCE
        self._active[:K][demote] = False
        idle[demote] = 0

    # ------------------------------------------------------------------------------------------
    # Adaptation of Lambda and V
    # ------------------------------------------------------------------------------------------
    def _adapt_lambda_and_threshold(self):
        """New Lambda: the largest L such that det(S_L) >= EPS_DET, where S_L is the similarity
        matrix of the L most recent samples. One Cholesky factorisation of the similarity matrix
        of the whole window gives every det(S_L) as a product of squared pivots.
        New V: the (1 - r) quantile of the largest similarity of every one of the Lambda most
        recent samples to another one, where r is the smoothed ratio of the number of clusters
        to the number of nodes."""
        if self._components_stale:
            self._num_components = self._count_components()
            self._components_stale = False
        K = self.num_nodes_
        lam = self.lambda_
        alpha = 1.0 / float(np.max(self._sigma[:K]))
        L = min(self.BUFFER_FACTOR * lam, self._buf_count)
        # the rows and columns start from the newest sample
        S = 1.0 / (1.0 + alpha * squareform(pdist(self._newest_rows(L))))
        # p > 0: only the leading p - 1 rows were factorised
        R, p = lapack.dpotrf(S, lower=1, clean=0)
        m = L if p == 0 else p - 1
        cp = np.cumprod(np.diagonal(R)[:m])
        det_prefix = cp * cp  # det(S_L) for L = 1..m
        below = np.flatnonzero(det_prefix < self.EPS_DET)
        new_lambda = m if below.size == 0 else int(below[0])
        if new_lambda < 2:
            return  # not even two samples are diverse: keep Lambda, V and r
        if new_lambda < lam:
            beta = 1.0 - (1.0 - 1.0 / lam) ** (new_lambda - 1)
        else:
            beta = 1.0 / lam
        self._ratio_nc = beta * (self._num_components / K) + (1.0 - beta) * self._ratio_nc
        sub = S[:new_lambda, :new_lambda].copy()
        np.fill_diagonal(sub, -np.inf)
        u = np.max(sub, axis=1)
        self.lambda_ = new_lambda
        self.similarity_threshold_ = float(_quantile(u, 1.0 - self._ratio_nc))

    def _count_components(self):
        """Number of connected components of the edge graph."""
        return int(connected_components(self._edge_graph(), directed=False)[0])

    # ------------------------------------------------------------------------------------------
    # Nodes, node pairs and running sums
    # ------------------------------------------------------------------------------------------
    def _add_node(self, x):
        k = self.num_nodes_
        if self._w is None or k == len(self._w):
            self._grow_node_arrays(x.size)
        self.num_nodes_ = k + 1
        self._w[k] = x
        self._count[k] = 1
        self._sigma[k] = max(self._update_running_std(x), self.SIGMA_MIN)
        self._active[k] = False
        self._degree[k] = 0
        self._cand_row_sum[k] = 0
        self._prev_count[k] = 0
        self._prev_cand_row_sum[k] = 0
        self._idle[k] = 0
        self._id[k] = self._next_id
        self._next_id += 1
        self._num_components += 1  # a new node is an isolated component

    def _grow_node_arrays(self, d):
        capacity = 64 if self._w is None else 2 * len(self._w)
        for name in self._NODE_ARRAYS:
            if name == "_w":
                new = np.zeros((capacity, d))
            elif name == "_active":
                new = np.zeros(capacity, dtype=bool)
            elif name == "_id":
                new = np.zeros(capacity, dtype=np.int64)
            else:
                new = np.zeros(capacity)
            old = getattr(self, name)
            if old is not None:
                new[: self.num_nodes_] = old[: self.num_nodes_]
            setattr(self, name, new)

    def _increment_count(self, k):
        c = self._count[k]
        self._count[k] = c + 1
        if c == 1:  # the node now has a count > 1
            self._num_count_above1 += 1
            self._sum_count_above1 += 2
        else:
            self._sum_count_above1 += 1

    def _find_pair(self, a, b):
        """Slot of the node pair (a, b), created when the pair is new."""
        i, j = (a, b) if a < b else (b, a)
        key = (int(self._id[i]), int(self._id[j]))
        slot = self._pair_slot.get(key)
        if slot is None:
            slot = self._num_pairs
            if slot == len(self._pair_i):
                for name in self._PAIR_ARRAYS:
                    old = getattr(self, name)
                    new = np.zeros(max(256, 2 * slot), dtype=old.dtype)
                    new[:slot] = old
                    setattr(self, name, new)
            self._pair_i[slot] = i
            self._pair_j[slot] = j
            self._pair_edge[slot] = 0
            self._pair_cand[slot] = 0
            self._pair_slot[key] = slot
            self._num_pairs += 1
        return slot

    def _compact_pairs(self):
        """Keep only the pairs with a positive co-activation count (a pruned pair has none)."""
        n = self._num_pairs
        keep = self._pair_cand[:n] > 0
        m = int(np.count_nonzero(keep))
        for name in self._PAIR_ARRAYS:
            a = getattr(self, name)
            a[:m] = a[:n][keep]
        self._num_pairs = m
        ids_i = self._id[self._pair_i[:m]].tolist()
        ids_j = self._id[self._pair_j[:m]].tolist()
        self._pair_slot = dict(zip(zip(ids_i, ids_j), range(m)))

    def _refresh_candidate_sums(self):
        v = self._pair_cand[: self._num_pairs]
        v = v[v > 0]
        self._sum_cand = 2 * int(v.sum())  # every pair is two entries of the symmetric E^cand
        self._num_cand = 2 * v.size

    def _update_running_std(self, x):
        """Welford's update of the running mean and variance of the learned samples. Returns the
        largest standard deviation over the features."""
        if self._run_count == 0:
            self._run_mean = x.copy()
            self._run_m2 = np.zeros_like(x)
            self._run_count = 1
            return 0.0
        self._run_count += 1
        delta = x - self._run_mean
        self._run_mean += delta / self._run_count
        self._run_m2 += delta * (x - self._run_mean)
        # the division by a positive number and the square root keep the order of the features
        return float(np.sqrt(self._run_m2.max() / (self._run_count - 1)))

    # ------------------------------------------------------------------------------------------
    # Ring buffer of the most recent samples
    # ------------------------------------------------------------------------------------------
    def _append_to_buffer(self, x):
        if self._buf is None:
            self._buf = np.zeros((500, x.size))
        elif self._buf_count == len(self._buf):
            self._grow_buffer(2 * len(self._buf))
        self._buf[(self._buf_start + self._buf_count) % len(self._buf)] = x
        self._buf_count += 1
        if 0 < self._buf_keep < self._buf_count:
            self._drop_oldest(self._buf_count - self._buf_keep)

    def _update_buffer_keep_len(self):
        self._buf_keep = self.BUFFER_FACTOR * self.lambda_
        if self._buf_count > self._buf_keep:
            self._drop_oldest(self._buf_count - self._buf_keep)

    def _drop_oldest(self, n):
        self._buf_start = (self._buf_start + n) % len(self._buf)
        self._buf_count -= n

    def _grow_buffer(self, capacity):
        rows = self._newest_rows(self._buf_count)[::-1]
        self._buf = np.zeros((capacity, self._buf.shape[1]))
        self._buf[: len(rows)] = rows
        self._buf_start = 0

    def _newest_rows(self, L):
        """The L most recent samples, the newest first."""
        rows = (self._buf_start + self._buf_count - 1 - np.arange(L)) % len(self._buf)
        return self._buf[rows]


def _quantile(x, p):
    """The p quantile of x: the k-th smallest value is the (k - 0.5) / n quantile, with linear
    interpolation in between."""
    x = np.sort(x)
    pos = np.clip(np.asarray(p, dtype=float) * x.size - 0.5, 0.0, x.size - 1.0)
    k = pos.astype(int)
    return x[k] + (pos - k) * (x[np.minimum(k + 1, x.size - 1)] - x[k])

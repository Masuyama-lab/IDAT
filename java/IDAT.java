// Copyright (c) 2025-2026 Naoki Masuyama
// SPDX-License-Identifier: MIT

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

/**
 * IDAT: Inverse Distance-based ART with Topology.
 *
 * <pre>
 *     IDAT model = new IDAT();             // a new model, with no hyperparameter to set
 *     model.fit(X);                        // learn the rows of X one at a time, in order
 *     int[] labels = model.predict(Y);     // cluster label of every row of Y
 * </pre>
 *
 * <p>X and Y are {@code double[][]}, one row per sample. fit can be called again with new samples.
 * The model keeps learning from where it stopped, so a data stream can be given chunk by chunk.
 * fitPredict(X) is fit(X) followed by predict(X).
 *
 * <pre>
 * How it works (one sample x at a time)
 *   Category choice   Every node k has a prototype w_k and a scale sigma_k. The similarity of x to
 *                     node k is s_k = 1 / (1 + ||x - w_k|| / sigma_k). The two most similar nodes
 *                     are s1 and s2. The first three samples become nodes.
 *   Vigilance test    V is the similarity threshold (the vigilance parameter).
 *                       s_1 &lt;  V           : x becomes a new node.
 *                       s_1 &gt;= V, s_2 &lt;= V : w_s1 moves towards x.
 *                       s_1 &gt;= V, s_2 &gt;  V : w_s1 and w_s2 move towards x. When both nodes have
 *                     won more often than the average of the nodes that have won at least twice,
 *                     both become active (getIsWeight()) and their co-activation count
 *                     E^cand(s1, s2) grows. An edge is created once that count exceeds the average
 *                     of the positive co-activation counts.
 *   Maintenance       Every Lambda samples: edges longer than Q3 + 1.5 IQR of the edge lengths are
 *                     removed, nodes that have never won since their creation are removed except
 *                     the newest Lambda of them, and an active node that is isolated and has
 *                     stayed idle for two maintenances becomes inactive.
 *   Adaptation        Every Lambda samples, once there are more than two nodes: Lambda is set to
 *                     the largest number L of recent samples whose similarity matrix S_L is still
 *                     diverse, det(S_L) &gt;= EPS_DET, and V is set to a quantile of the similarities
 *                     within these samples. The quantile level follows the ratio of the number of
 *                     clusters to the number of nodes, smoothed over the adaptations.
 *   Prediction        A sample gets the label of its most similar active node (of all nodes if
 *                     none is active). The labels are the connected components of the edge graph,
 *                     numbered from 1 in the order of their smallest node index.
 *
 * Accessors (read only)
 *   getNumNodes()              K, the number of nodes, active or not
 *   getWeight()                [K x d] prototypes
 *   getCountNode()             [K] number of wins of every node
 *   getBufferStd()             [K] sigma_k
 *   getIsWeight()              [K] true for the active nodes, the nodes used by predict
 *   getEdges()                 the edges, as pairs (i, j) of node indices with i &lt; j
 *   getLambda()                current window length and maintenance interval
 *   getSimilarityThreshold()   current similarity threshold V
 *   getLabelCluster()          [K] cluster label of every node, set by predict
 *   getSampleCounts()          number of samples learned so far
 *
 * Network size (the three counts printed by the main programs)
 *   all nodes      getNumNodes()
 *   active nodes   the number of true entries of getIsWeight()
 *   clusters       the number of distinct values of getLabelCluster() over the active nodes after
 *                  predict, that is, the clusters that contain at least one active node
 * </pre>
 *
 * <p>The class constants are the values used in the paper. They are not meant to be tuned.
 *
 * <p>Requires JDK 11 or later and no other library.
 *
 * <pre>
 * Reference
 *   N. Masuyama, Y. Toda, Y. Nojima, and H. Ishibuchi, "An adaptive resonance theory-based
 *   topological clustering algorithm with a self-adjusting vigilance parameter,"
 *   arXiv:2511.17983, 2025. https://arxiv.org/abs/2511.17983
 * </pre>
 */
public class IDAT {
    /** Initial Lambda. */
    public static final int LAMBDA_INIT = 2;
    /** A window of L samples is diverse while det(S_L) &gt;= EPS_DET. */
    public static final double EPS_DET = 1.0e-6;
    /** c of the bounds Q3 + c IQR (edge pruning) and Q1 - c IQR (demotion). */
    public static final double IQR_FACTOR = 1.5;
    /** Idle maintenances before an isolated active node is demoted. */
    public static final int DEMOTE_TOLERANCE = 2;
    /** The buffer keeps the newest BUFFER_FACTOR * Lambda samples. */
    public static final int BUFFER_FACTOR = 2;
    /** Lower bound of sigma_k. */
    public static final double SIGMA_MIN = 1.0e-6;

    // ---------------------------------------------------------------------------------------------
    // Learning and prediction
    // ---------------------------------------------------------------------------------------------

    /** Learn the rows of X one at a time, in the given order. */
    public IDAT fit(double[][] X) {
        for (double[] x : X) {
            if (sampleCounts == 0) {
                d = x.length;
            } else if (x.length != d) {
                throw new IllegalArgumentException("IDAT: all samples must have the same length");
            }
            appendToBuffer(x);
            learnOneSample(x);
            sampleCounts += 1;
            recalcCounter += 1;
            if (recalcCounter >= lambda && numNodes > 2) {
                adaptLambdaAndThreshold();
                recalcCounter = 0;
                updateBufferKeepLen();
            }
        }
        return this;
    }

    /** Cluster label of every row of X. */
    public int[] predict(double[][] X) {
        final int K = numNodes;
        if (K == 0) {
            labelCluster = new int[0];
            return new int[0];
        }
        final int[] labels = new int[K];
        connectedComponents(labels);
        labelCluster = labels.clone();
        int numUsed = 0;
        for (int k = 0; k < K; ++k) {
            if (active[k]) {
                numUsed += 1;
            }
        }
        int[] nodes;
        if (numUsed > 0) {
            nodes = new int[numUsed];
            int q = 0;
            for (int k = 0; k < K; ++k) {
                if (active[k]) {
                    nodes[q++] = k;
                }
            }
        } else {
            nodes = new int[K];
            for (int k = 0; k < K; ++k) {
                nodes[k] = k;
            }
        }
        final double[] alpha = new double[nodes.length];
        for (int q = 0; q < nodes.length; ++q) {
            alpha[q] = 1.0 / sigma[nodes[q]];
        }
        final int[] out = new int[X.length];
        for (int n = 0; n < X.length; ++n) {
            if (X[n].length != d) {
                throw new IllegalArgumentException("IDAT: all samples must have the same length");
            }
            // the most similar node (the first of equal similarities)
            int best = 0;
            double bestS = Double.NEGATIVE_INFINITY;
            for (int q = 0; q < nodes.length; ++q) {
                final double s = similarity(alpha[q], distance(w, nodes[q] * d, X[n], 0));
                if (s > bestS) {
                    bestS = s;
                    best = q;
                }
            }
            out[n] = labels[nodes[best]];
        }
        return out;
    }

    /** fit(X) followed by predict(X). */
    public int[] fitPredict(double[][] X) {
        fit(X);
        return predict(X);
    }

    // ---------------------------------------------------------------------------------------------
    // Accessors
    // ---------------------------------------------------------------------------------------------

    /** K, the number of nodes, active or not. */
    public int getNumNodes() {
        return numNodes;
    }

    /** [K x d] prototypes. */
    public double[][] getWeight() {
        final double[][] W = new double[numNodes][];
        for (int k = 0; k < numNodes; ++k) {
            W[k] = Arrays.copyOfRange(w, k * d, (k + 1) * d);
        }
        return W;
    }

    /** [K] number of wins of every node. */
    public double[] getCountNode() {
        return Arrays.copyOf(count, numNodes);
    }

    /** [K] sigma_k. */
    public double[] getBufferStd() {
        return Arrays.copyOf(sigma, numNodes);
    }

    /** [K] true for the active nodes, the nodes used by predict. */
    public boolean[] getIsWeight() {
        return Arrays.copyOf(active, numNodes);
    }

    /** The edges, as rows (i, j) of node indices with i &lt; j, in lexicographic order. */
    public int[][] getEdges() {
        final List<int[]> E = new ArrayList<>();
        for (int p = 0; p < numPairs; ++p) {
            if (pairEdge[p] > 0) {
                E.add(new int[] {pairI[p], pairJ[p]});
            }
        }
        E.sort((a, b) -> a[0] != b[0] ? Integer.compare(a[0], b[0]) : Integer.compare(a[1], b[1]));
        return E.toArray(new int[0][]);
    }

    /** Current window length and maintenance interval. */
    public int getLambda() {
        return lambda;
    }

    /** Current similarity threshold V. */
    public double getSimilarityThreshold() {
        return similarityThreshold;
    }

    /** [K] cluster label of every node, set by predict. */
    public int[] getLabelCluster() {
        return labelCluster.clone();
    }

    /** Number of samples learned so far. */
    public long getSampleCounts() {
        return sampleCounts;
    }

    // In the arithmetic below, every product and every sum is rounded to double on its own. Java
    // does not merge a product and a sum into one fused multiply add.

    private int d = 0; // number of features
    private int numNodes = 0;
    private int lambda = LAMBDA_INIT;
    private double similarityThreshold = 0.0;
    private int[] labelCluster = new int[0];
    private long sampleCounts = 0;

    private int recalcCounter = 0; // samples learned since the last adaptation
    private double ratioNc = 0.0; // smoothed ratio of the number of clusters to the number of nodes

    // nodes: arrays with one entry per node. They have spare rows, and the first numNodes are used.
    private int capacity = 0;
    private double[] w = new double[0]; // prototypes, [capacity x d] row by row
    private double[] count = new double[0]; // number of wins
    private double[] sigma = new double[0]; // sigma_k
    private boolean[] active = new boolean[0]; // true for the active nodes
    private double[] degree = new double[0]; // number of edges of every node
    private double[] candRowSum = new double[0]; // row sums of E^cand
    private double[] prevCount = new double[0]; // count at the last maintenance
    private double[] prevCandRowSum = new double[0]; // candRowSum at the last maintenance
    private double[] idle = new double[0]; // consecutive maintenances as a demotion candidate
    // creation number of every node, kept when other nodes are removed
    private long[] id = new long[0];
    private long nextId = 0;

    // Key of a node pair: the creation numbers of its two nodes.
    private static final class PairKey {
        private final long first;
        private final long second;

        PairKey(long first, long second) {
            this.first = first;
            this.second = second;
        }

        @Override
        public boolean equals(Object other) {
            if (!(other instanceof PairKey)) {
                return false;
            }
            final PairKey o = (PairKey) other;
            return first == o.first && second == o.second;
        }

        @Override
        public int hashCode() {
            return Long.hashCode(first * 0x9E3779B97F4A7C15L ^ second);
        }
    }

    // node pairs (i < j) that have been co-activated, with their edge count and E^cand.
    // A pair is looked up by the creation numbers of its two nodes.
    private final HashMap<PairKey, Integer> pairSlot = new HashMap<>();
    private int numPairs = 0;
    private int[] pairI = new int[0];
    private int[] pairJ = new int[0];
    private double[] pairEdge = new double[0];
    private double[] pairCand = new double[0];

    // running sums behind the two thresholds of the vigilance test (over the symmetric matrix)
    private long sumCountAbove1 = 0; // sum of the counts > 1
    private long numCountAbove1 = 0; // number of the counts > 1
    private long sumCand = 0; // sum of the positive entries of E^cand
    private long numCand = 0; // number of these entries

    private long numComponents = 0; // number of connected components (valid when not stale)
    private boolean componentsStale = true;

    // running mean, second moment and count of the learned samples (Welford)
    private double[] runMean = new double[0];
    private double[] runM2 = new double[0];
    private long runCount = 0;

    // ring buffer of the most recent samples, [bufRows x d] row by row
    private double[] buf = new double[0];
    private int bufRows = 0;
    private int bufStart = 0; // row of the oldest kept sample
    private int bufCount = 0; // number of kept samples
    private int bufKeep = 0; // samples to keep (0 before the first adaptation: all)

    // The larger of a and b, and a when they are equal.
    private static double max(double a, double b) {
        return a < b ? b : a;
    }

    // The smaller of a and b, and a when they are equal.
    private static double min(double a, double b) {
        return b < a ? b : a;
    }

    // Euclidean distance between a[aOff ..] and b[bOff ..], with the features added from the first
    // to the last.
    private double distance(double[] a, int aOff, double[] b, int bOff) {
        double s = 0.0;
        for (int f = 0; f < d; ++f) {
            final double diff = a[aOff + f] - b[bOff + f];
            s += diff * diff;
        }
        return Math.sqrt(s);
    }

    // 1 / (1 + alpha * distance)
    private static double similarity(double alpha, double dist) {
        return 1.0 / (1.0 + alpha * dist);
    }

    // w_k += rate * (x - w_k)
    private void moveTowards(int k, double[] x, double rate) {
        final int off = k * d;
        for (int f = 0; f < d; ++f) {
            w[off + f] += rate * (x[f] - w[off + f]);
        }
    }

    // ---------------------------------------------------------------------------------------------
    // One sample: category choice, vigilance test and maintenance
    // ---------------------------------------------------------------------------------------------
    private void learnOneSample(double[] x) {
        if (numNodes < 3) {
            addNode(x);
            if (numNodes == 2) {
                // the first node takes the scale of the first two samples
                sigma[0] = sigma[1];
            }
        } else {
            // category choice: the two nearest nodes (the first of equal distances is taken)
            final int K = numNodes;
            final double[] distances = new double[K];
            for (int k = 0; k < K; ++k) {
                distances[k] = distance(w, k * d, x, 0);
            }
            final int s1 = argmin(distances);
            final double d1 = distances[s1];
            distances[s1] = Double.POSITIVE_INFINITY;
            final int s2 = argmin(distances);
            final double d2 = distances[s2];
            final double sim1 = similarity(1.0 / sigma[s1], d1);
            final double sim2 = similarity(1.0 / sigma[s2], d2);
            final double V = similarityThreshold;
            if (sim1 < V) {
                addNode(x); // Case I
            } else {
                incrementCount(s1); // Case II
                moveTowards(s1, x, 1.0 / count[s1]);
                sigma[s1] = max(updateRunningStd(x), SIGMA_MIN);
                if (sim2 > V) { // Case III
                    incrementCount(s2);
                    moveTowards(s2, x, 1.0 / (count[s1] + count[s2]));
                    updateTopology(s1, s2);
                }
            }
        }
        // maintenance every Lambda samples
        // (sampleCounts is the number of samples learned before x)
        if (sampleCounts % lambda == 0) {
            maintain();
        }
    }

    private static int argmin(double[] v) {
        int best = 0;
        for (int k = 1; k < v.length; ++k) {
            if (v[k] < v[best]) {
                best = k;
            }
        }
        return best;
    }

    // Activation of s1 and s2, their co-activation count and, possibly, an edge (Case III).
    private void updateTopology(int s1, int s2) {
        // mean of the counts > 1
        final double countTh = (double) sumCountAbove1 / (double) numCountAbove1;
        if (count[s1] > countTh && count[s2] > countTh) {
            active[s1] = true;
            active[s2] = true;
            final int slot = findPair(s1, s2);
            final double c = pairCand[slot];
            pairCand[slot] = c + 1;
            candRowSum[s1] += 1;
            candRowSum[s2] += 1;
            sumCand += 2;
            if (c == 0) {
                numCand += 2;
            }
            // mean of the positive entries of E^cand
            final double edgeTh = (double) sumCand / (double) numCand;
            if (c + 1 > edgeTh) {
                if (pairEdge[slot] == 0) {
                    componentsStale = true;
                    degree[s1] += 1;
                    degree[s2] += 1;
                }
                pairEdge[slot] += 1;
            }
        }
    }

    private void maintain() {
        final int K = numNodes;
        // 1) edge pruning: Tukey's rule on the edge lengths
        final int n = numPairs;
        int numEdges = 0;
        for (int p = 0; p < n; ++p) {
            if (pairEdge[p] > 0) {
                numEdges += 1;
            }
        }
        if (numEdges > 0) {
            final int[] e = new int[numEdges];
            int q0 = 0;
            for (int p = 0; p < n; ++p) {
                if (pairEdge[p] > 0) {
                    e[q0++] = p;
                }
            }
            final double[] length = new double[numEdges];
            for (int q = 0; q < numEdges; ++q) {
                length[q] = distance(w, pairI[e[q]] * d, w, pairJ[e[q]] * d);
            }
            // every edge counts as (i, j) and (j, i)
            final double[] both = new double[2 * numEdges];
            System.arraycopy(length, 0, both, 0, numEdges);
            System.arraycopy(length, 0, both, numEdges, numEdges);
            Arrays.sort(both);
            final double Q1 = quantile(both, 0.25);
            final double Q3 = quantile(both, 0.75);
            final double IQR = Q3 - Q1;
            final double upper = Q3 + IQR_FACTOR * IQR;
            boolean anyTooLong = false;
            for (int q = 0; q < numEdges; ++q) {
                if (length[q] > upper) {
                    pairEdge[e[q]] = 0;
                    pairCand[e[q]] = 0;
                    anyTooLong = true;
                }
            }
            if (anyTooLong) {
                refreshCandidateSums();
                componentsStale = true;
                Arrays.fill(degree, 0, K, 0.0);
                Arrays.fill(candRowSum, 0, K, 0.0);
                for (int p = 0; p < n; ++p) {
                    if (pairEdge[p] > 0) {
                        degree[pairI[p]] += 1;
                        degree[pairJ[p]] += 1;
                    }
                    candRowSum[pairI[p]] += pairCand[p];
                    candRowSum[pairJ[p]] += pairCand[p];
                }
                // many pruned pairs: drop them
                if (n - numCand / 2 > Math.max(1024, n / 2)) {
                    compactPairs();
                }
            }
        }
        // 2) removal of the nodes that have never won: the newest Lambda of them are kept
        final long numCount1 = K - numCountAbove1;
        if (numCountAbove1 > 0 && numCount1 > lambda) {
            final int[] r = new int[K];
            int numR = 0;
            for (int k = 0; k < K; ++k) {
                if (count[k] == 1) {
                    r[numR++] = k;
                }
            }
            removeNodes(Arrays.copyOf(r, numR - lambda));
        }
        // 3) demotion of the isolated active nodes that stay idle
        demoteIdleNodes();
    }

    // The nodes r have count 1, so they have no edge and no co-activation. Every one of them is an
    // isolated component, and the degrees and row sums of the other nodes and the sums behind the
    // thresholds do not change.
    private void removeNodes(int[] r) {
        final int K = numNodes;
        final boolean[] keep = new boolean[K];
        Arrays.fill(keep, true);
        for (int k : r) {
            keep[k] = false;
        }
        final int[] newIndex = new int[K];
        int m = 0;
        for (int k = 0; k < K; ++k) {
            newIndex[k] = m;
            if (keep[k]) {
                System.arraycopy(w, k * d, w, m * d, d);
                count[m] = count[k];
                sigma[m] = sigma[k];
                active[m] = active[k];
                degree[m] = degree[k];
                candRowSum[m] = candRowSum[k];
                prevCount[m] = prevCount[k];
                prevCandRowSum[m] = prevCandRowSum[k];
                idle[m] = idle[k];
                id[m] = id[k];
                m += 1;
            }
        }
        numNodes = m;
        for (int p = 0; p < numPairs; ++p) {
            pairI[p] = newIndex[pairI[p]];
            pairJ[p] = newIndex[pairJ[p]];
        }
        numComponents -= r.length;
    }

    // The activity of a node is its number of wins and co-activations since the last maintenance.
    // An active node that is isolated and whose activity is at most max(Q1 - 1.5 IQR, 0) of the
    // positive activities for DEMOTE_TOLERANCE consecutive maintenances is demoted (getIsWeight()
    // becomes false).
    private void demoteIdleNodes() {
        final int K = numNodes;
        final double[] activity = new double[K];
        boolean anyActivity = false;
        for (int k = 0; k < K; ++k) {
            activity[k] = (count[k] - prevCount[k]) + (candRowSum[k] - prevCandRowSum[k]);
            prevCount[k] = count[k];
            prevCandRowSum[k] = candRowSum[k];
            anyActivity = anyActivity || activity[k] > 0;
        }
        if (!anyActivity) {
            Arrays.fill(idle, 0, K, 0.0);
            return;
        }
        final boolean[] isolatedActive = new boolean[K];
        boolean anyIsolatedActive = false;
        for (int k = 0; k < K; ++k) {
            isolatedActive[k] = active[k] && degree[k] == 0;
            anyIsolatedActive = anyIsolatedActive || isolatedActive[k];
        }
        final boolean[] candidate = new boolean[K];
        if (anyIsolatedActive) {
            int numPositive = 0;
            for (int k = 0; k < K; ++k) {
                if (activity[k] > 0) {
                    numPositive += 1;
                }
            }
            final double[] a = new double[numPositive];
            int q = 0;
            for (int k = 0; k < K; ++k) {
                if (activity[k] > 0) {
                    a[q++] = activity[k];
                }
            }
            Arrays.sort(a);
            final double Q1 = quantile(a, 0.25);
            final double Q3 = quantile(a, 0.75);
            final double IQR = Q3 - Q1;
            final double bound = max(Q1 - IQR_FACTOR * IQR, 0.0);
            for (int k = 0; k < K; ++k) {
                candidate[k] = isolatedActive[k] && activity[k] <= bound;
            }
        }
        for (int k = 0; k < K; ++k) {
            if (candidate[k]) {
                idle[k] += 1;
            } else {
                idle[k] = 0;
            }
            if (idle[k] >= DEMOTE_TOLERANCE) {
                active[k] = false;
                idle[k] = 0;
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
    private void adaptLambdaAndThreshold() {
        if (componentsStale) {
            numComponents = countComponents();
            componentsStale = false;
        }
        final int K = numNodes;
        final int lam = lambda;
        double sigmaMax = sigma[0];
        for (int k = 1; k < K; ++k) {
            if (sigmaMax < sigma[k]) {
                sigmaMax = sigma[k];
            }
        }
        final double alpha = 1.0 / sigmaMax;
        final int L = Math.min(BUFFER_FACTOR * lam, bufCount);
        // the rows and columns start from the newest sample
        final double[] rows = newestRows(L);
        final double[] S = new double[L * L];
        for (int i = 0; i < L; ++i) {
            S[i * L + i] = similarity(alpha, 0.0);
            for (int j = i + 1; j < L; ++j) {
                final double s = similarity(alpha, distance(rows, i * d, rows, j * d));
                S[i * L + j] = s;
                S[j * L + i] = s;
            }
        }
        // Cholesky factorisation S = R R^T, row by row. It stops at the first pivot that is not
        // positive, so that only the leading m rows are factorised.
        final double[] R = new double[L * L];
        int m = L;
        for (int i = 0; i < L && m == L; ++i) {
            for (int j = 0; j < i; ++j) {
                double dot = 0.0;
                for (int k = 0; k < j; ++k) {
                    dot += R[i * L + k] * R[j * L + k];
                }
                R[i * L + j] = (S[i * L + j] - dot) / R[j * L + j];
            }
            double dot = 0.0;
            for (int k = 0; k < i; ++k) {
                dot += R[i * L + k] * R[i * L + k];
            }
            final double pivot = S[i * L + i] - dot;
            if (pivot > 0.0) {
                R[i * L + i] = Math.sqrt(pivot);
            } else {
                m = i;
            }
        }
        // det(S_L) for L = 1..m is the square of the product of the first L diagonal entries
        int newLambda = m;
        double cp = 1.0;
        for (int q = 0; q < m; ++q) {
            cp *= R[q * L + q];
            if (cp * cp < EPS_DET) {
                newLambda = q;
                break;
            }
        }
        if (newLambda < 2) {
            return; // not even two samples are diverse: keep Lambda, V and r
        }
        double beta;
        if (newLambda < lam) {
            final double base = 1.0 - 1.0 / (double) lam;
            beta = 1.0 - Math.pow(base, (double) (newLambda - 1));
        } else {
            beta = 1.0 / (double) lam;
        }
        final double ratioNow = (double) numComponents / (double) K;
        ratioNc = beta * ratioNow + (1.0 - beta) * ratioNc;
        final double[] u = new double[newLambda];
        Arrays.fill(u, Double.NEGATIVE_INFINITY);
        for (int i = 0; i < newLambda; ++i) {
            for (int j = 0; j < newLambda; ++j) {
                if (j != i) {
                    u[i] = max(u[i], S[i * L + j]);
                }
            }
        }
        Arrays.sort(u);
        lambda = newLambda;
        similarityThreshold = quantile(u, 1.0 - ratioNc);
    }

    // Labels (written to labels, of length numNodes) of the connected components of the edge graph,
    // numbered from 1 in the order of their smallest node index. Returns the number of components.
    private int connectedComponents(int[] labels) {
        final int K = numNodes;
        // adjacency lists of the edge graph
        final int[] start = new int[K + 1];
        for (int p = 0; p < numPairs; ++p) {
            if (pairEdge[p] > 0) {
                start[pairI[p] + 1] += 1;
                start[pairJ[p] + 1] += 1;
            }
        }
        for (int k = 0; k < K; ++k) {
            start[k + 1] += start[k];
        }
        final int[] neighbor = new int[start[K]];
        final int[] next = Arrays.copyOf(start, K);
        for (int p = 0; p < numPairs; ++p) {
            if (pairEdge[p] > 0) {
                neighbor[next[pairI[p]]++] = pairJ[p];
                neighbor[next[pairJ[p]]++] = pairI[p];
            }
        }
        // breadth first search from every node that has no label yet
        Arrays.fill(labels, 0, K, 0);
        int num = 0;
        final int[] queue = new int[K];
        for (int v = 0; v < K; ++v) {
            if (labels[v] != 0) {
                continue;
            }
            num += 1;
            labels[v] = num;
            queue[0] = v;
            int tail = 1;
            for (int h = 0; h < tail; ++h) {
                final int a = queue[h];
                for (int q = start[a]; q < start[a + 1]; ++q) {
                    final int b = neighbor[q];
                    if (labels[b] == 0) {
                        labels[b] = num;
                        queue[tail++] = b;
                    }
                }
            }
        }
        return num;
    }

    // Number of connected components of the edge graph.
    private int countComponents() {
        return connectedComponents(new int[numNodes]);
    }

    // ---------------------------------------------------------------------------------------------
    // Nodes, node pairs and running sums
    // ---------------------------------------------------------------------------------------------
    private void addNode(double[] x) {
        final int k = numNodes;
        if (k == capacity) {
            growNodeArrays();
        }
        numNodes = k + 1;
        System.arraycopy(x, 0, w, k * d, d);
        count[k] = 1;
        sigma[k] = max(updateRunningStd(x), SIGMA_MIN);
        active[k] = false;
        degree[k] = 0;
        candRowSum[k] = 0;
        prevCount[k] = 0;
        prevCandRowSum[k] = 0;
        idle[k] = 0;
        id[k] = nextId;
        nextId += 1;
        numComponents += 1; // a new node is an isolated component
    }

    private void growNodeArrays() {
        capacity = capacity == 0 ? 64 : 2 * capacity;
        w = Arrays.copyOf(w, capacity * d);
        count = Arrays.copyOf(count, capacity);
        sigma = Arrays.copyOf(sigma, capacity);
        active = Arrays.copyOf(active, capacity);
        degree = Arrays.copyOf(degree, capacity);
        candRowSum = Arrays.copyOf(candRowSum, capacity);
        prevCount = Arrays.copyOf(prevCount, capacity);
        prevCandRowSum = Arrays.copyOf(prevCandRowSum, capacity);
        idle = Arrays.copyOf(idle, capacity);
        id = Arrays.copyOf(id, capacity);
    }

    private void incrementCount(int k) {
        final double c = count[k];
        count[k] = c + 1;
        if (c == 1) { // the node now has a count > 1
            numCountAbove1 += 1;
            sumCountAbove1 += 2;
        } else {
            sumCountAbove1 += 1;
        }
    }

    // Slot of the node pair (a, b), created when the pair is new.
    private int findPair(int a, int b) {
        final int i = a < b ? a : b;
        final int j = a < b ? b : a;
        final PairKey key = new PairKey(id[i], id[j]);
        final Integer found = pairSlot.get(key);
        if (found != null) {
            return found;
        }
        final int slot = numPairs;
        if (slot == pairI.length) {
            final int newCapacity = Math.max(256, 2 * slot);
            pairI = Arrays.copyOf(pairI, newCapacity);
            pairJ = Arrays.copyOf(pairJ, newCapacity);
            pairEdge = Arrays.copyOf(pairEdge, newCapacity);
            pairCand = Arrays.copyOf(pairCand, newCapacity);
        }
        pairI[slot] = i;
        pairJ[slot] = j;
        pairEdge[slot] = 0;
        pairCand[slot] = 0;
        pairSlot.put(key, slot);
        numPairs += 1;
        return slot;
    }

    // Keep only the pairs with a positive co-activation count (a pruned pair has none).
    private void compactPairs() {
        int m = 0;
        for (int p = 0; p < numPairs; ++p) {
            if (pairCand[p] > 0) {
                pairI[m] = pairI[p];
                pairJ[m] = pairJ[p];
                pairEdge[m] = pairEdge[p];
                pairCand[m] = pairCand[p];
                m += 1;
            }
        }
        numPairs = m;
        pairSlot.clear();
        for (int p = 0; p < m; ++p) {
            pairSlot.put(new PairKey(id[pairI[p]], id[pairJ[p]]), p);
        }
    }

    private void refreshCandidateSums() {
        double total = 0.0;
        long num = 0;
        for (int p = 0; p < numPairs; ++p) {
            if (pairCand[p] > 0) {
                total += pairCand[p];
                num += 1;
            }
        }
        // every pair is two entries of the symmetric E^cand
        sumCand = 2 * (long) total;
        numCand = 2 * num;
    }

    // Welford's update of the running mean and variance of the learned samples. Returns the largest
    // standard deviation over the features.
    private double updateRunningStd(double[] x) {
        if (runCount == 0) {
            runMean = x.clone();
            runM2 = new double[x.length];
            runCount = 1;
            return 0.0;
        }
        runCount += 1;
        final double n = (double) runCount;
        double m2Max = Double.NEGATIVE_INFINITY;
        for (int f = 0; f < d; ++f) {
            final double delta = x[f] - runMean[f];
            runMean[f] += delta / n;
            runM2[f] += delta * (x[f] - runMean[f]);
            m2Max = max(m2Max, runM2[f]);
        }
        // the division by a positive number and the square root keep the order of the features
        return Math.sqrt(m2Max / (double) (runCount - 1));
    }

    // ---------------------------------------------------------------------------------------------
    // Ring buffer of the most recent samples
    // ---------------------------------------------------------------------------------------------
    private void appendToBuffer(double[] x) {
        if (bufRows == 0) {
            bufRows = 500;
            buf = new double[bufRows * d];
        } else if (bufCount == bufRows) {
            growBuffer(2 * bufRows);
        }
        final int row = (bufStart + bufCount) % bufRows;
        System.arraycopy(x, 0, buf, row * d, d);
        bufCount += 1;
        if (0 < bufKeep && bufKeep < bufCount) {
            dropOldest(bufCount - bufKeep);
        }
    }

    private void updateBufferKeepLen() {
        bufKeep = BUFFER_FACTOR * lambda;
        if (bufCount > bufKeep) {
            dropOldest(bufCount - bufKeep);
        }
    }

    private void dropOldest(int n) {
        bufStart = (bufStart + n) % bufRows;
        bufCount -= n;
    }

    private void growBuffer(int newCapacity) {
        final double[] rows = newestRows(bufCount);
        buf = new double[newCapacity * d];
        for (int q = 0; q < bufCount; ++q) { // the oldest sample first
            final int src = bufCount - 1 - q;
            System.arraycopy(rows, src * d, buf, q * d, d);
        }
        bufRows = newCapacity;
        bufStart = 0;
    }

    // The L most recent samples, the newest first, [L x d] row by row.
    private double[] newestRows(int L) {
        final double[] rows = new double[L * d];
        for (int q = 0; q < L; ++q) {
            final int row = (bufStart + bufCount - 1 - q) % bufRows;
            System.arraycopy(buf, row * d, rows, q * d, d);
        }
        return rows;
    }

    // The p quantile of x, sorted in ascending order: the k-th smallest value is the (k - 0.5) / n
    // quantile, with linear interpolation in between.
    private static double quantile(double[] x, double p) {
        final double n = (double) x.length;
        final double pos = min(max(p * n - 0.5, 0.0), n - 1.0);
        final int k = (int) pos;
        final int k1 = Math.min(k + 1, x.length - 1);
        return x[k] + (pos - (double) k) * (x[k1] - x[k]);
    }
}

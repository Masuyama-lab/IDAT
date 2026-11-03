// Copyright (c) 2025-2026 Naoki Masuyama
// SPDX-License-Identifier: MIT

import java.util.Arrays;

/**
 * Adjusted Mutual Information (AMI) and Adjusted Rand Index (ARI) of a predicted clustering.
 *
 * <pre>
 *     double[] s = ClusteringEvaluationMetrics.clusteringEvaluationMetrics(groundTruth, predicted);
 *     double ami = s[0];
 *     double ari = s[1];
 * </pre>
 *
 * <p>AMI follows the default of scikit-learn: the arithmetic mean of the two entropies as the
 * normalizer and the natural logarithm.
 */
public final class ClusteringEvaluationMetrics {
    private ClusteringEvaluationMetrics() {
    }

    /** {AMI, ARI} of the predicted labels against the ground truth labels. */
    public static double[] clusteringEvaluationMetrics(int[] groundTruth, int[] predicted) {
        return new double[] {adjustedMutualInformation(groundTruth, predicted),
            adjustedRandIndex(groundTruth, predicted)};
    }

    // Sorted distinct values of v.
    private static int[] uniqueSorted(int[] v) {
        final int[] s = v.clone();
        Arrays.sort(s);
        int m = 0;
        for (int n = 0; n < s.length; ++n) {
            if (n == 0 || s[n] != s[m - 1]) {
                s[m++] = s[n];
            }
        }
        return Arrays.copyOf(s, m);
    }

    // Contingency table of two labelings: entry (i, j) is the number of samples with the i-th
    // smallest label of gt and the j-th smallest label of pred.
    private static double[][] contingencyTable(int[] gt, int[] pred) {
        final int[] classesGt = uniqueSorted(gt);
        final int[] classesPred = uniqueSorted(pred);
        final double[][] contingency = new double[classesGt.length][classesPred.length];
        for (int n = 0; n < gt.length; ++n) {
            final int i = Arrays.binarySearch(classesGt, gt[n]);
            final int j = Arrays.binarySearch(classesPred, pred[n]);
            contingency[i][j] += 1;
        }
        return contingency;
    }

    // Expected mutual information E[MI] for the row sums a and the column sums b of a contingency
    // table of n samples (hypergeometric model).
    private static double expectedMutualInformation(double[] a, double[] b, int n) {
        // log(k!) for k = 0..n
        final double[] logFactorial = new double[n + 1];
        for (int k = 1; k <= n; ++k) {
            logFactorial[k] = logFactorial[k - 1] + Math.log((double) k);
        }

        final double N = (double) n;
        double emi = 0.0;
        for (int i = 0; i < a.length; ++i) {
            final int ai = (int) a[i];
            for (int j = 0; j < b.length; ++j) {
                final int bj = (int) b[j];
                final int nijMin = ai + bj > n ? ai + bj - n : 0;
                final int nijMax = Math.min(ai, bj);
                final double logNChooseBj = logNChooseK(logFactorial, n, bj);
                for (int nij = Math.max(nijMin, 1); nij <= nijMax; ++nij) {
                    // P(nij) = C(ai, nij) * C(n - ai, bj - nij) / C(n, bj)
                    final double logP = logNChooseK(logFactorial, ai, nij)
                        + logNChooseK(logFactorial, n - ai, bj - nij) - logNChooseBj;
                    final double p = Math.exp(logP);
                    final double x = (double) nij;
                    emi += (x / N) * Math.log((N * x) / (a[i] * b[j])) * p;
                }
            }
        }
        return emi;
    }

    private static double logNChooseK(double[] logFactorial, int n, int k) {
        return logFactorial[n] - logFactorial[k] - logFactorial[n - k];
    }

    private static double adjustedMutualInformation(int[] gt, int[] pred) {
        final int n = gt.length;
        if (n == 0) {
            return Double.NaN;
        }
        final double[][] contingency = contingencyTable(gt, pred);
        final int numGt = contingency.length;
        final int numPred = contingency[0].length;
        final double[] a = new double[numGt];
        final double[] b = new double[numPred];
        for (int i = 0; i < numGt; ++i) {
            for (int j = 0; j < numPred; ++j) {
                a[i] += contingency[i][j];
                b[j] += contingency[i][j];
            }
        }
        final double N = (double) n;

        // mutual information (natural logarithm)
        double mi = 0.0;
        for (int i = 0; i < numGt; ++i) {
            for (int j = 0; j < numPred; ++j) {
                final double nij = contingency[i][j];
                if (nij > 0) {
                    mi += (nij / N) * Math.log((N * nij) / (a[i] * b[j]));
                }
            }
        }

        // entropies (natural logarithm)
        double entropyGt = 0.0;
        for (double ai : a) {
            entropyGt -= (ai / N) * Math.log(ai / N);
        }
        double entropyPred = 0.0;
        for (double bj : b) {
            entropyPred -= (bj / N) * Math.log(bj / N);
        }

        final double emi = expectedMutualInformation(a, b, n);

        // arithmetic mean of the two entropies
        final double normalizer = 0.5 * (entropyGt + entropyPred);
        if (Math.abs(normalizer - emi) < 1.0e-15) {
            // both clusterings have zero entropy
            return Math.abs(mi - emi) < 1.0e-15 ? 1.0 : 0.0;
        }
        final double ami = (mi - emi) / (normalizer - emi);
        return Math.min(Math.max(ami, -1.0), 1.0);
    }

    private static double choose2(double x) {
        return x * (x - 1) / 2;
    }

    private static double adjustedRandIndex(int[] gt, int[] pred) {
        final int n = gt.length;
        if (n < 2) {
            return 1.0;
        }
        final double[][] contingency = contingencyTable(gt, pred);
        final int numGt = contingency.length;
        final int numPred = contingency[0].length;

        double sumComb = 0.0;
        final double[] a = new double[numGt];
        final double[] b = new double[numPred];
        for (int i = 0; i < numGt; ++i) {
            for (int j = 0; j < numPred; ++j) {
                sumComb += choose2(contingency[i][j]);
                a[i] += contingency[i][j];
                b[j] += contingency[i][j];
            }
        }
        double sumCombC = 0.0;
        for (double ai : a) {
            sumCombC += choose2(ai);
        }
        double sumCombK = 0.0;
        for (double bj : b) {
            sumCombK += choose2(bj);
        }
        final double totalComb = choose2((double) n);

        final double expectedIndex = (sumCombC * sumCombK) / totalComb;
        final double maxIndex = 0.5 * (sumCombC + sumCombK);
        final double den = maxIndex - expectedIndex;
        if (Math.abs(den) < 1.0e-15) {
            return Math.abs(sumComb - expectedIndex) < 1.0e-15 ? 1.0 : 0.0;
        }
        return (sumComb - expectedIndex) / den;
    }
}

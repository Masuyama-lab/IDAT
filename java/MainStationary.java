// Copyright (c) 2025-2026 Naoki Masuyama
// SPDX-License-Identifier: MIT

import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.TreeSet;

public class MainStationary {
    static final int NUM_TRIALS = 10; // Number of trials

    // Load dataset ============
    static final String DATA_FILE = "../datasets/iris.csv";
    // static final String DATA_FILE = "../datasets/optdigits.csv";

    // Every line of the file holds the features of one sample and its class label in the last
    // column.
    static boolean loadCsv(String path, List<double[]> data, List<Integer> target) {
        List<String> lines;
        try {
            lines = Files.readAllLines(Paths.get(path));
        } catch (IOException e) {
            return false;
        }
        for (String line : lines) {
            if (line.isEmpty()) {
                continue;
            }
            final String[] fields = line.split(",");
            final double[] row = new double[fields.length - 1];
            for (int f = 0; f < row.length; ++f) {
                row[f] = Double.parseDouble(fields[f].trim());
            }
            target.add((int) Double.parseDouble(fields[fields.length - 1].trim()));
            data.add(row);
        }
        return true;
    }

    // Fisher-Yates shuffle of v with rng
    static void shuffle(int[] v, Random rng) {
        for (int n = v.length - 1; n > 0; --n) {
            final int m = rng.nextInt(n + 1);
            final int tmp = v[n];
            v[n] = v[m];
            v[m] = tmp;
        }
    }

    // Number of all nodes, of active nodes, and of clusters, AMI, ARI, and training time
    static double[] runTrial(int t, List<double[]> data, List<Integer> target) {
        // Randomization
        final int N = data.size();
        final int[] order = new int[N];
        for (int n = 0; n < N; ++n) {
            order[n] = n;
        }
        shuffle(order, new Random(t));
        final double[][] X = new double[N][];
        final int[] y = new int[N];
        for (int n = 0; n < N; ++n) {
            X[n] = data.get(order[n]);
            y[n] = target.get(order[n]);
        }

        // Train IDAT
        final IDAT model = new IDAT();
        final long start = System.nanoTime();
        model.fit(X);
        final double trainTime = (System.nanoTime() - start) / 1.0e9;

        // Test IDAT
        final int[] predictedLabels = model.predict(X);

        // Evaluation
        final double[] scores =
            ClusteringEvaluationMetrics.clusteringEvaluationMetrics(y, predictedLabels);

        // Number of all nodes, of active nodes, and of clusters
        // (clusters with at least one active node)
        final boolean[] isWeight = model.getIsWeight();
        final int[] labelCluster = model.getLabelCluster();
        final TreeSet<Integer> clusters = new TreeSet<>();
        double numActiveNodes = 0;
        for (int k = 0; k < isWeight.length; ++k) {
            if (isWeight[k]) {
                numActiveNodes += 1;
                clusters.add(labelCluster[k]);
            }
        }
        return new double[] {model.getNumNodes(), numActiveNodes, clusters.size(), scores[0],
            scores[1], trainTime};
    }

    // One line of the results: label, mean and standard deviation
    static void printLine(PrintStream out, String label, double mean, double std) {
        out.print(String.format(Locale.ROOT, "%s%.4f \u00b1 %.4f\n", label, mean, std));
    }

    public static void main(String[] args) throws IOException {
        final PrintStream out = new PrintStream(System.out, true, "UTF-8");
        final List<double[]> data = new ArrayList<>();
        final List<Integer> target = new ArrayList<>();
        if (!loadCsv(DATA_FILE, data, target)) {
            System.err.println("Cannot open " + DATA_FILE);
            System.exit(1);
        }
        final int numClasses = new TreeSet<>(target).size();

        final List<double[]> results = new ArrayList<>();
        for (int t = 1; t <= NUM_TRIALS; ++t) {
            results.add(runTrial(t, data, target));
        }

        // Mean and standard deviation over the trials
        final int numValues = results.get(0).length;
        final double[] mean = new double[numValues];
        final double[] stdDev = new double[numValues];
        for (int v = 0; v < numValues; ++v) {
            for (double[] r : results) {
                mean[v] += r[v];
            }
            mean[v] /= NUM_TRIALS;
            for (double[] r : results) {
                stdDev[v] += (r[v] - mean[v]) * (r[v] - mean[v]);
            }
            stdDev[v] = Math.sqrt(stdDev[v] / (NUM_TRIALS - 1));
        }

        // Display average and standard deviation results
        out.print("=========== Stationary ============\n");
        out.print(String.format(Locale.ROOT, "# classes: %d\n", numClasses));
        out.print(String.format(Locale.ROOT, "Average results over %d runs\n", NUM_TRIALS));
        printLine(out, "Average # all nodes:     ", mean[0], stdDev[0]);
        printLine(out, "Average # active nodes:  ", mean[1], stdDev[1]);
        printLine(out, "Average # clusters:      ", mean[2], stdDev[2]);
        printLine(out, "         Average AMI:    ", mean[3], stdDev[3]);
        printLine(out, "         Average ARI:    ", mean[4], stdDev[4]);
        printLine(out, "        Average Time:    ", mean[5], stdDev[5]);
        out.print("\n");
    }
}

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

public class MainNonstationary {
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

    // Samples and labels of every class (shuffled within each class)
    static final List<double[][]> CLASS_DATA = new ArrayList<>();
    static final List<int[]> CLASS_TARGET = new ArrayList<>();

    // Number of all nodes, of active nodes, and of clusters, AMI, ARI, training time,
    // incremental AMI and ARI, and BWT of AMI and ARI
    static double[] runTrial(int t) {
        final int numClasses = CLASS_DATA.size();

        // Class order of this trial
        final int[] classOrder = new int[numClasses];
        for (int c = 0; c < numClasses; ++c) {
            classOrder[c] = c;
        }
        shuffle(classOrder, new Random(t));

        // Scores after learning each class
        final double[] amiSteps = new double[numClasses];
        final double[] ariSteps = new double[numClasses];

        final IDAT model = new IDAT();
        double trainTime = 0.0;
        final List<double[]> evalData = new ArrayList<>();
        final List<Integer> evalTarget = new ArrayList<>();
        for (int i = 0; i < numClasses; ++i) {
            final int c = classOrder[i];

            // Train IDAT on the current class
            final long start = System.nanoTime();
            model.fit(CLASS_DATA.get(c));
            trainTime += (System.nanoTime() - start) / 1.0e9;

            // Evaluate on all classes learned so far
            for (double[] x : CLASS_DATA.get(c)) {
                evalData.add(x);
            }
            for (int y : CLASS_TARGET.get(c)) {
                evalTarget.add(y);
            }
            if (i == 0) {
                amiSteps[i] = 1.0;
                ariSteps[i] = 1.0;
            } else {
                final int[] predictedLabels = model.predict(evalData.toArray(new double[0][]));
                final int[] y = new int[evalTarget.size()];
                for (int n = 0; n < y.length; ++n) {
                    y[n] = evalTarget.get(n);
                }
                final double[] scores =
                    ClusteringEvaluationMetrics.clusteringEvaluationMetrics(y, predictedLabels);
                amiSteps[i] = scores[0];
                ariSteps[i] = scores[1];
            }
        }

        // Number of all nodes, of active nodes, and of clusters
        // (clusters with at least one active node) after learning all classes
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

        // Average incremental AMI/ARI and backward transfer (BWT)
        double incAmi = 0.0;
        double incAri = 0.0;
        double bwtAmi = 0.0;
        double bwtAri = 0.0;
        for (int i = 0; i < numClasses; ++i) {
            incAmi += amiSteps[i];
            incAri += ariSteps[i];
        }
        incAmi /= numClasses;
        incAri /= numClasses;
        for (int i = 0; i + 1 < numClasses; ++i) {
            bwtAmi += amiSteps[numClasses - 1] - amiSteps[i];
            bwtAri += ariSteps[numClasses - 1] - ariSteps[i];
        }
        bwtAmi /= numClasses - 1;
        bwtAri /= numClasses - 1;
        return new double[] {model.getNumNodes(), numActiveNodes, clusters.size(),
            amiSteps[numClasses - 1], ariSteps[numClasses - 1], trainTime, incAmi, incAri,
            bwtAmi, bwtAri};
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

        // Split data by class (shuffled within each class)
        final Random rng = new Random(0);
        final TreeSet<Integer> classLabels = new TreeSet<>(target);
        final int numClasses = classLabels.size();
        for (int c : classLabels) {
            int num = 0;
            for (int y : target) {
                if (y == c) {
                    num += 1;
                }
            }
            final int[] idx = new int[num];
            int q = 0;
            for (int n = 0; n < target.size(); ++n) {
                if (target.get(n) == c) {
                    idx[q++] = n;
                }
            }
            shuffle(idx, rng);
            final double[][] X = new double[num][];
            final int[] y = new int[num];
            for (int m = 0; m < num; ++m) {
                X[m] = data.get(idx[m]);
                y[m] = target.get(idx[m]);
            }
            CLASS_DATA.add(X);
            CLASS_TARGET.add(y);
        }

        final List<double[]> results = new ArrayList<>();
        for (int t = 1; t <= NUM_TRIALS; ++t) {
            results.add(runTrial(t));
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
        out.print("=========== Nonstationary ============\n");
        out.print(String.format(Locale.ROOT, "# classes: %d\n", numClasses));
        out.print(String.format(Locale.ROOT, "Average results over %d runs\n", NUM_TRIALS));
        printLine(out, "Average # all nodes:     ", mean[0], stdDev[0]);
        printLine(out, "Average # active nodes:  ", mean[1], stdDev[1]);
        printLine(out, "Average # clusters:      ", mean[2], stdDev[2]);
        printLine(out, "         Average AMI:    ", mean[3], stdDev[3]);
        printLine(out, "         Average ARI:    ", mean[4], stdDev[4]);
        printLine(out, "        Average Time:    ", mean[5], stdDev[5]);
        out.print("\n");
        printLine(out, "Average Incremental AMI: ", mean[6], stdDev[6]);
        printLine(out, "Average Incremental ARI: ", mean[7], stdDev[7]);
        printLine(out, "  Backward Transfer AMI: ", mean[8], stdDev[8]);
        printLine(out, "  Backward Transfer ARI: ", mean[9], stdDev[9]);
        out.print("\n");
    }
}

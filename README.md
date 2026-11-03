# Inverse Distance-based ART with Topology (IDAT)

IDAT is a topological clustering algorithm based on Adaptive Resonance Theory (ART).
It learns the samples one at a time and builds a network of nodes and edges.
The clusters are the connected components of this network.
IDAT adjusts its vigilance parameter and its window length from the data during learning.
Therefore, IDAT does not require dataset-specific hyperparameter tuning.
The class constants in the code are the values used in the paper. They are not meant to be tuned.

## Folder structure

```
datasets/
  iris.mat, optdigits.mat            Iris and OptDigits
  iris.csv, optdigits.csv            the same data as CSV files (class label in the last column)
matlab/
  IDAT.m                             IDAT class
  mainStationary.m                   repeated trials in the stationary setting
  mainNonstationary.m                repeated trials in the nonstationary setting
  clusteringEvaluationMetrics.m      AMI and ARI
python/
  idat.py                            IDAT class
  main_stationary.py                 repeated trials in the stationary setting
  main_nonstationary.py              repeated trials in the nonstationary setting
  requirements.txt                   required packages
cpp/
  idat.hpp                           IDAT class
  main_stationary.cpp                repeated trials in the stationary setting
  main_nonstationary.cpp             repeated trials in the nonstationary setting
  clustering_evaluation_metrics.hpp  AMI and ARI
  Makefile                           builds the two main programs
java/
  IDAT.java                          IDAT class
  MainStationary.java                repeated trials in the stationary setting
  MainNonstationary.java             repeated trials in the nonstationary setting
  ClusteringEvaluationMetrics.java   AMI and ARI
```

## Requirements

- MATLAB: tested with MATLAB R2025b. The Statistics and Machine Learning Toolbox is required. Without the Parallel Computing Toolbox, `parfor` runs as an ordinary `for` loop.
- Python: Python 3.10 or later and the packages in `python/requirements.txt` (numpy, scipy, scikit-learn, and joblib).
- C++: a compiler that supports C++17, and make. No other library is required.
- Java: JDK 11 or later. No other library is required.

```
pip install -r python/requirements.txt
```

## Usage

MATLAB

```matlab
net = IDAT();              % a new model
net.fit(X);                % learn the rows of X in the given order
labels = net.predict(Y);   % cluster label of every row of Y
```

Python

```python
from idat import IDAT

model = IDAT()             # a new model
model.fit(X)               # learn the rows of X in the given order
labels = model.predict(Y)  # cluster label of every row of Y
```

C++ (`X` and `Y` are `std::vector<std::vector<double>>`, with one inner vector per sample)

```cpp
#include "idat.hpp"

IDAT net;                                  // a new model
net.fit(X);                                // learn the rows of X in the given order
std::vector<int> labels = net.predict(Y);  // cluster label of every row of Y
```

Java (`X` and `Y` are `double[][]`, with one row per sample)

```java
IDAT model = new IDAT();          // a new model
model.fit(X);                     // learn the rows of X in the given order
int[] labels = model.predict(Y);  // cluster label of every row of Y
```

`fit` can be called again with new samples. The model then continues learning from its current state.

### Main scripts

Each main script runs 10 trials on Iris and prints the mean ± standard deviation over the trials.
The number of trials is set at the top of each script (`numTrials` in MATLAB, `N_TRIALS` in Python, and `NUM_TRIALS` in C++ and in Java).
To use OptDigits, comment out the line that loads Iris and uncomment the line that loads OptDigits.

- `mainStationary.m`, `main_stationary.py`, `main_stationary.cpp`, and `MainStationary.java`: In each trial, all samples are shuffled and learned once. The labels of the same samples are then predicted. The script prints the number of all nodes, the number of active nodes, the number of clusters, the Adjusted Mutual Information (AMI), the Adjusted Rand Index (ARI), and the training time in seconds.
- `mainNonstationary.m`, `main_nonstationary.py`, `main_nonstationary.cpp`, and `MainNonstationary.java`: The classes are given one at a time in a random order (class-incremental setting). After each class, the model is evaluated on all classes learned so far. In addition to the values above, the script prints the average incremental AMI and ARI and the backward transfer (BWT).

The number of all nodes counts every node in the network. The number of active nodes counts the nodes used by `predict`. The number of clusters counts the clusters that contain at least one active node.

```
>> cd matlab
>> mainStationary
```

```
cd python
python main_stationary.py
```

For C++, run `make` in `cpp/` and then run `./main_stationary` and `./main_nonstationary` in the same folder. The programs read the data from `../datasets/`.

```
cd cpp
make
./main_stationary
```

For Java, run `javac *.java` in `java/` and then run `java MainStationary` and `java MainNonstationary` in the same folder. The programs read the data from `../datasets/`.

```
cd java
javac *.java
java MainStationary
```

## Example output

The results below were obtained by giving the four versions the samples in the same order, namely the order that the Python scripts produce with the trial number as the seed. All four versions gave the same numbers, so each result is shown once. The main scripts of the other languages shuffle the samples with their own random number generators, so their runs give different numbers.

Iris, stationary setting

```
=========== Stationary ============
# classes: 3
Average results over 10 runs
Average # all nodes:     25.9000 ± 3.9567
Average # active nodes:  9.4000 ± 1.3499
Average # clusters:      6.2000 ± 1.3166
         Average AMI:    0.6920 ± 0.0317
         Average ARI:    0.6135 ± 0.0762
```

Iris, nonstationary setting

```
=========== Nonstationary ============
# classes: 3
Average results over 10 runs
Average # all nodes:     37.0000 ± 2.6667
Average # active nodes:  6.9000 ± 1.5951
Average # clusters:      3.3000 ± 0.4830
         Average AMI:    0.5688 ± 0.1044
         Average ARI:    0.4911 ± 0.1410

Average Incremental AMI: 0.6657 ± 0.0535
Average Incremental ARI: 0.6301 ± 0.0688
  Backward Transfer AMI: -0.1453 ± 0.1197
  Backward Transfer ARI: -0.2086 ± 0.1622
```

The following table shows the mean values over 10 trials for OptDigits (10 classes).

| Setting | # all nodes | # active nodes | # clusters | AMI | ARI | Incremental AMI | Incremental ARI | BWT AMI | BWT ARI |
|---|---|---|---|---|---|---|---|---|---|
| Stationary | 227.9 | 69.9 | 26.2 | 0.774 | 0.684 | | | | |
| Nonstationary | 668.2 | 122.4 | 30.5 | 0.768 | 0.684 | 0.737 | 0.675 | 0.035 | 0.011 |

The line with the training time is omitted above, because the training time depends on the language and on the machine. For OptDigits, the mean training time per trial in seconds on one machine was as follows.

| Setting | MATLAB | Python | C++ | Java |
|---|---|---|---|---|
| Stationary | 0.56 | 0.18 | 0.09 | 0.12 |
| Nonstationary | 0.38 | 0.33 | 0.10 | 0.13 |

## Notes

- `predict` returns cluster labels that start at 1. MATLAB returns a 1 × N row vector. Python returns an array of length N. C++ returns a `std::vector<int>` of length N. Java returns an `int[]` of length N.
- When they are given the samples in the same order, the MATLAB, Python, C++, and Java versions give the same results up to floating point rounding.
- IDAT learns the samples in the given order in a single pass. Therefore, the result depends on the order of the samples.

## Citation

N. Masuyama, Y. Toda, Y. Nojima, and H. Ishibuchi, <br>
"An adaptive resonance theory-based topological clustering algorithm with a self-adjusting vigilance parameter,"  <br>
arXiv preprint arXiv:2511.17983, November 2025. <br>
[https://arxiv.org/abs/2511.17983](https://arxiv.org/abs/2511.17983)

## License

MIT License. Copyright (c) 2025-2026 Naoki Masuyama. See [LICENSE](LICENSE).

## Contact

Please contact masuyama@omu.ac.jp if you have any problems.

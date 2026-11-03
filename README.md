# Inverse Distance-based ART with Topology (IDAT)

IDAT is a topological clustering algorithm based on Adaptive Resonance Theory (ART).
It learns the samples one at a time and builds a network of nodes and edges.
The clusters are the connected components of this network.
IDAT adjusts its vigilance parameter and its window length from the data during learning.
Therefore, IDAT does not require dataset-specific hyperparameter tuning.
The class constants in the code are the values used in the paper. They are not meant to be tuned.

## Folder structure

```
datasets/   Iris and OptDigits
matlab/
  IDAT.m                           IDAT class
  mainStationary.m                 repeated trials in the stationary setting
  mainNonstationary.m              repeated trials in the nonstationary setting
  clusteringEvaluationMetrics.m    AMI and ARI
python/
  idat.py                          IDAT class
  main_stationary.py               repeated trials in the stationary setting
  main_nonstationary.py            repeated trials in the nonstationary setting
  requirements.txt                 required packages
```

## Requirements

- MATLAB: tested with MATLAB R2025b. The Statistics and Machine Learning Toolbox is required. Without the Parallel Computing Toolbox, `parfor` runs as an ordinary `for` loop.
- Python: Python 3.10 or later and the packages in `python/requirements.txt` (numpy, scipy, scikit-learn, and joblib).

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

`fit` can be called again with new samples. The model then continues learning from its current state.

### Main scripts

Each main script runs 10 trials on Iris and prints the mean ± standard deviation over the trials.
The number of trials is set at the top of each script (`numTrials` in MATLAB and `N_TRIALS` in Python).
To use OptDigits, comment out the line that loads Iris and uncomment the line that loads OptDigits.

- `mainStationary.m` and `main_stationary.py`: In each trial, all samples are shuffled and learned once. The labels of the same samples are then predicted. The script prints the number of all nodes, the number of active nodes, the number of clusters, the Adjusted Mutual Information (AMI), the Adjusted Rand Index (ARI), and the training time in seconds.
- `mainNonstationary.m` and `main_nonstationary.py`: The classes are given one at a time in a random order (class-incremental setting). After each class, the model is evaluated on all classes learned so far. In addition to the values above, the script prints the average incremental AMI and ARI and the backward transfer (BWT).

The number of all nodes counts every node in the network. The number of active nodes counts the nodes used by `predict`. The number of clusters counts the clusters that contain at least one active node.

```
>> cd matlab
>> mainStationary
```

```
cd python
python main_stationary.py
```

## Example output

The following outputs are for Iris. The training time depends on the machine.
MATLAB and Python use different random number generators. Therefore, the shuffled orders differ between the two versions, and so do the numbers.

MATLAB, `mainStationary.m`

```
=========== Stationary ============
# classes: 3
Average results over 10 runs
Average # all nodes:     22.4 ± 3.893
Average # active nodes:  9.2 ± 2.201
Average # clusters:      6.5 ± 1.2693
         Average AMI:    0.67141 ± 0.035366
         Average ARI:    0.60926 ± 0.050334
        Average Time:    0.27162 ± 0.039101
```

MATLAB, `mainNonstationary.m`

```
=========== Nonstationary ============
# classes: 3
Average results over 10 runs
Average # all nodes:     38 ± 5.1208
Average # active nodes:  6.1 ± 1.4491
Average # clusters:      4 ± 0.8165
         Average AMI:    0.55078 ± 0.074403
         Average ARI:    0.49179 ± 0.072298
        Average Time:    0.26621 ± 0.0307

Average Incremental AMI: 0.58846 ± 0.10132
Average Incremental ARI: 0.56536 ± 0.094575
  Backward Transfer AMI: -0.056513 ± 0.09397
  Backward Transfer ARI: -0.11035 ± 0.14141
```

Python, `main_stationary.py`

```
=========== Stationary ============
# classes: 3
Average results over 10 runs
Average # all nodes:     25.9000 ± 3.9567
Average # active nodes:  9.4000 ± 1.3499
Average # clusters:      6.2000 ± 1.3166
         Average AMI:    0.6920 ± 0.0317
         Average ARI:    0.6135 ± 0.0762
        Average Time:    0.0036 ± 0.0001
```

Python, `main_nonstationary.py`

```
=========== Nonstationary ============
# classes: 3
Average results over 10 runs
Average # all nodes:     37.0000 ± 2.6667
Average # active nodes:  6.9000 ± 1.5951
Average # clusters:      3.3000 ± 0.4830
         Average AMI:    0.5688 ± 0.1044
         Average ARI:    0.4911 ± 0.1410
        Average Time:    0.0040 ± 0.0008

Average Incremental AMI: 0.6657 ± 0.0535
Average Incremental ARI: 0.6301 ± 0.0688
  Backward Transfer AMI: -0.1453 ± 0.1197
  Backward Transfer ARI: -0.2086 ± 0.1622
```

The following table shows the mean values over 10 trials for OptDigits (10 classes).

| Setting | Version | # all nodes | # active nodes | # clusters | AMI | ARI | Incremental AMI | Incremental ARI | BWT AMI | BWT ARI |
|---|---|---|---|---|---|---|---|---|---|---|
| Stationary | MATLAB | 244.1 | 73.3 | 26.8 | 0.765 | 0.660 | | | | |
| Stationary | Python | 227.9 | 69.9 | 26.2 | 0.774 | 0.684 | | | | |
| Nonstationary | MATLAB | 693.7 | 127.4 | 30.1 | 0.776 | 0.702 | 0.757 | 0.716 | 0.021 | −0.015 |
| Nonstationary | Python | 668.2 | 122.4 | 30.5 | 0.768 | 0.684 | 0.737 | 0.675 | 0.035 | 0.011 |

## Notes

- `predict` returns cluster labels that start at 1. MATLAB returns a 1 × N row vector. Python returns an array of length N.
- When they are given the samples in the same order, the MATLAB and Python versions give the same results up to floating point rounding.
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

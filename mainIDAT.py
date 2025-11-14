import numpy as np
import pandas as pd
import time

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from idat import IDAT
from eval_metrics import clustering_evaluation_metrics  # use eval_metrics.py


def load_openml_dataset(data_name):
    """
    Load dataset from OpenML by name.
    Then label-encode the target column if needed.
    """
    data, tmp_target = fetch_openml(name=data_name, version=1, return_X_y=True, as_frame=False)
    df = pd.DataFrame(tmp_target, columns=['label'])
    le = LabelEncoder()
    df['encoded'] = le.fit_transform(df['label'])
    tmp_target = np.array(df['encoded'])
    target = np.array(tmp_target, dtype='int')
    num_classes = np.unique(target).shape[0]
    return data, target, num_classes


rng = np.random.RandomState(1)

data_name = 'OptDigits'  # Example dataset name from OpenML
data, target, num_classes = load_openml_dataset(data_name)

# Shuffle data and target together using the same RandomState
data, target = shuffle(data, target, random_state=rng)

# Initialize IDAT model
idat = IDAT()

# Choose mode: 'stationary' for full dataset training, 'nonstationary' for incremental class-wise training
mode = 'stationary'

start_time = time.time()

if mode == 'stationary':
    # Train on the entire dataset at once
    idat.fit(data)
elif mode == 'nonstationary':
    # Incrementally update the model class by class
    unique_labels = np.unique(target)
    for current_class in unique_labels:
        # Filter the data for the current class
        indices = np.where(target == current_class)[0]
        class_data = data[indices]
        idat.fit(class_data)
else:
    raise ValueError("Invalid mode. Choose either 'stationary' or 'nonstationary'.")

predicted_labels = idat.predict(data)
elapsed_time = time.time() - start_time

# --- Compute AMI, ARI, NVI via eval_metrics.py ---
ami_val, ari_val, nvi_val = clustering_evaluation_metrics(target, predicted_labels)

center_indices = np.where(idat.is_weight_)[0]  # Extract indices where is_weight_ is True.
num_nodes = len(center_indices)
num_clusters = np.unique(idat.labels_[center_indices]).size

# Display results
print("[IDAT]")
print(f"       # nodes: {num_nodes}")
print(f"       # clusters: {num_clusters}")
print(f"       AMI: {ami_val:.4f}")
print(f"       ARI: {ari_val:.4f}")
print(f"       NVI: {nvi_val:.4f}")
print(f"       time: {elapsed_time:.4f} sec")
print("--------------------------------------------")

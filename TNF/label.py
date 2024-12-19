import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import argparse
from sklearn.metrics import accuracy_score
from scipy.stats import mode

def calculate_tnf(dna_sequences, kernel=False):
    # Define all possible tetra-nucleotides
    nucleotides = ['A', 'T', 'C', 'G']
    tetra_nucleotides = [a+b+c+d for a in nucleotides for b in nucleotides for c in nucleotides for d in nucleotides]
    
    # Build mapping from tetra-nucleotide to index
    tnf_index = {tn: i for i, tn in enumerate(tetra_nucleotides)}        

    # Iterate over each sequence and update counts
    embedding = np.zeros((len(dna_sequences), len(tetra_nucleotides)))
    for j, seq in enumerate(dna_sequences):
        for i in range(len(seq) - 3):
            tetra_nuc = seq[i:i+4]
            embedding[j, tnf_index[tetra_nuc]] += 1
    
    # Convert counts to frequencies
    total_counts = np.sum(embedding, axis=1)
    embedding = embedding / total_counts[:, None]

    if kernel:
        def validate_input_array(array):
            "Returns a C-contiguous array with its own data, similar to the input array."
            if not array.flags["C_CONTIGUOUS"]:
                array = np.ascontiguousarray(array)
            if not array.flags["OWNDATA"]:
                array = array.copy()

            assert array.flags["C_CONTIGUOUS"] and array.flags["OWNDATA"]
            return array

        npz = np.load("./helper/kernel.npz")
        kernel = validate_input_array(npz["arr_0"])
        embedding += -(1 / 256)
        embedding = np.dot(embedding, kernel)
        
    return embedding


parser = argparse.ArgumentParser()
parser.add_argument("--train_data_path", type=str, default='/projects/p32013/DNABERT-meta/GUE/0/cat.csv', help="Path to the training data (CSV file with 'sequence' column)")
parser.add_argument("--test_data_path", type=str, default='/projects/p32013/DNABERT-meta/GUE/0/test.csv', help="Path to the testing data (CSV file with 'sequence' column)")
parser.add_argument("--output_path", type=str, default='/projects/p32013/DNABERT-meta/TNF/test')
parser.add_argument('--num_label', type=int, default=2)
args = parser.parse_args()

# Load the training and testing datasets
train_data = pd.read_csv(args.train_data_path)
test_data = pd.read_csv(args.test_data_path)

# Extract sequences and labels from both datasets
train_sequence = train_data['sequence']
train_labels = train_data['label']  # Assuming the training data has a 'label' column
test_sequence = test_data['sequence']
test_labels = test_data['label']  # Assuming the test data has a 'label' column

# Calculate embeddings for both training and testing data
train_embeddings = np.array(calculate_tnf(train_sequence))
test_embeddings = np.array(calculate_tnf(test_sequence))

# Fit KMeans on the training data
kmeans = KMeans(n_clusters=args.num_label, random_state=0, n_init="auto")
kmeans.fit(train_embeddings)

# Predict the labels for the test data
test_labels_pred = kmeans.predict(test_embeddings)

# Map clusters to the majority label in the training data
train_cluster_assignments = kmeans.predict(train_embeddings)
cluster_to_label = {}

for cluster in range(args.num_label):
    # Find the labels for the training samples in the current cluster
    labels_in_cluster = train_labels[train_cluster_assignments == cluster]
    
    # Find the majority class label for this cluster
    majority_label = mode(labels_in_cluster).mode[0]
    
    # Map the cluster to the majority label
    cluster_to_label[cluster] = majority_label

    # Calculate the label distribution for this cluster
    label_counts = np.bincount(labels_in_cluster)
    label_ratios = label_counts / len(labels_in_cluster)
    
    print(f"Cluster {cluster} has majority label {majority_label} with a ratio of {label_ratios[majority_label]:.2f}")
    for label in range(args.num_label):
        print(f"  Label {label} has a ratio of {label_ratios[label]:.2f}")

# Map the test data's cluster labels to the corresponding majority labels
test_labels_mapped = [cluster_to_label[cluster] for cluster in test_labels_pred]

# Calculate accuracy score
accuracy = accuracy_score(test_labels, test_labels_mapped)
print(f"KMeans clustering accuracy: {accuracy:.4f}")

# If an output path is specified, save the results
if args.output_path:
    test_data['predicted_labels'] = test_labels_mapped
    test_data.to_csv(args.output_path, index=False)

# Optionally, print the mapped labels for the test data
print(test_labels_mapped)

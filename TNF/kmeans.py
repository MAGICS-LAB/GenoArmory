import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import argparse

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
            "Returns array similar to input array but C-contiguous and with own data."
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
parser.add_argument("--train_data_path", type=str, default='/projects/p32013/DNABERT-meta/GUE/0/cat.csv', help="path to the training data (CSV file with 'sequence' column)")
parser.add_argument("--test_data_path", type=str, default='/projects/p32013/DNABERT-meta/GUE/0/test.csv', help="path to the testing data (CSV file with 'sequence' column)")
parser.add_argument("--output_path", type=str, default='/projects/p32013/DNABERT-meta/TNF/test')
parser.add_argument('--num_label', type=int, default=2)
args = parser.parse_args()

# Load the training and testing datasets
train_data = pd.read_csv(args.train_data_path)
test_data = pd.read_csv(args.test_data_path)

# Extract sequences from both datasets
train_sequence = train_data['sequence']
test_sequence = test_data['sequence']

# Calculate embeddings for both train and test data
train_embeddings = np.array(calculate_tnf(train_sequence))
test_embeddings = np.array(calculate_tnf(test_sequence))

# Fit KMeans on the training data
kmeans = KMeans(n_clusters=args.num_label, random_state=0, n_init="auto")
kmeans.fit(train_embeddings)

# Predict the labels for the test data
test_labels = kmeans.predict(test_embeddings)

# If output path is specified, save the results
if args.output_path:
    test_data['predicted_labels'] = test_labels
    test_data.to_csv(args.output_path, index=False)

# Optionally, print the test labels
print(test_labels)

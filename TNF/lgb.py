from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import accuracy_score
import os
import numpy as np
import pandas as pd
import argparse
import json

def calculate_tnf(dna_sequences, kernel=False):
    # Define all possible tetra-nucleotides
    nucleotides = ['A', 'T', 'C', 'G']
    tetra_nucleotides = [a+b+c+d for a in nucleotides for b in nucleotides for c in nucleotides for d in nucleotides]
    tnf_index = {tn: i for i, tn in enumerate(tetra_nucleotides)}        

    # Calculate tetra-nucleotide frequencies for each sequence
    embedding = np.zeros((len(dna_sequences), len(tetra_nucleotides)))
    for j, seq in enumerate(dna_sequences):
        for i in range(len(seq) - 3):
            tetra_nuc = seq[i:i+4]
            if 'N' in tetra_nuc:
                continue
            embedding[j, tnf_index[tetra_nuc]] += 1

    # Convert counts to frequencies
    total_counts = np.sum(embedding, axis=1)
    embedding = embedding / total_counts[:, None]

    if kernel:
        # Validate input array to ensure it's C-contiguous and has its own data
        def validate_input_array(array):
            if not array.flags["C_CONTIGUOUS"]:
                array = np.ascontiguousarray(array)
            if not array.flags["OWNDATA"]:
                array = array.copy()
            return array

        # Load the kernel for adjustment
        npz = np.load("./helper/kernel.npz")
        kernel = validate_input_array(npz["arr_0"])
        embedding += -(1 / 256)
        embedding = np.dot(embedding, kernel)
        
    return embedding

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_data_path", type=str, default='train.csv', help="Path to the training DNA sequences. Expect a CSV file with 'sequence' and 'label' columns.")
parser.add_argument("--test_data_path", type=str, default='test.csv', help="Path to the testing DNA sequences. Expect a CSV file with 'sequence' and 'label' columns.")
parser.add_argument("--output_path", type=str, default='predictions.csv', help="Path to save the prediction results on test data.")
parser.add_argument("--log_path", type=str, default='/projects/p32013/DNABERT-meta/TNF/log', help="Path to save the accuracy log on test data.")
args = parser.parse_args()

# Load training data
train_data = pd.read_csv(args.train_data_path)
train_sequences = train_data['sequence']
train_labels = train_data['label']
train_embeddings = calculate_tnf(train_sequences)

# Load testing data
test_data = pd.read_csv(args.test_data_path)
test_sequences = test_data['sequence']
test_labels = test_data['label']
test_embeddings = calculate_tnf(test_sequences)

# Set LGBMClassifier parameters
lgb_params = {
    "objective": "binary",  # For binary classification
    "learning_rate": 0.1,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "n_estimators": 5000  # Maximum number of boosting iterations
}

# Initialize LGBMClassifier
model = LGBMClassifier(**lgb_params)

# Define callbacks
callbacks = [
    early_stopping(stopping_rounds=10),
    log_evaluation(period=10)  # Log progress every 10 iterations
]

# Train the model with early stopping
print("Training LGBMClassifier...")
model.fit(
    train_embeddings,
    train_labels,
    eval_set=[(train_embeddings, train_labels), (test_embeddings, test_labels)],
    eval_metric="error",  # Metric for evaluation
    callbacks=callbacks  # Pass early stopping and logging as callbacks
)

# Predict on the test data
print("Making predictions on test data...")
test_predictions = model.predict(test_embeddings)

# Calculate accuracy
accuracy = accuracy_score(test_labels, test_predictions)
print(f"{args.test_data_path} Test Accuracy: {accuracy:.4f}")

# Save prediction results
test_data['lgb'] = test_predictions
test_data.to_csv(args.output_path, index=False)
print(f"Predictions saved to {args.output_path}")

accuracy_json_path = args.log_path

# Check if the file already exists
if os.path.exists(accuracy_json_path):
    # If the file exists, try to read its contents
    try:
        with open(accuracy_json_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # If the file format is incorrect or reading fails, initialize as an empty dictionary
        data = {}
else:
    # If the file does not exist, initialize as an empty dictionary
    data = {}

# Check if the 'lgb' key already exists
if "lgb" not in data:
    data["lgb"] = accuracy  # Add the new accuracy value
    with open(accuracy_json_path, "w") as f:
        json.dump(data, f, indent=4)  # Save the updated data
    print(f"Accuracy saved to {accuracy_json_path}")
else:
    print(f"Key 'lgb' already exists in {accuracy_json_path}. Accuracy not updated.")



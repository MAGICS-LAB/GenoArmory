import os
import numpy as np
import pandas as pd
import xgboost as xgb
import argparse
from sklearn.metrics import accuracy_score
import json

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
            if 'N' in tetra_nuc:
                continue
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
parser.add_argument("--train_data_path", type=str, default='/projects/p32013/DNABERT-meta/GUE/0/train.csv', help="Path to the training DNA sequences. Expect a CSV file with 'sequence' and 'label' columns.")
parser.add_argument("--test_data_path", type=str, default='/projects/p32013/DNABERT-meta/GUE/0/test.csv', help="Path to the testing DNA sequences. Expect a CSV file with 'sequence' and 'label' columns.")
parser.add_argument("--output_path", type=str, default='/projects/p32013/DNABERT-meta/TNF/test', help="Path to save the prediction results on test data.")
parser.add_argument("--log_path", type=str, default='/projects/p32013/DNABERT-meta/TNF/log', help="Path to save the accuracy log on test data.")
args = parser.parse_args()

# Load training data and extract sequences and labels
train_data = pd.read_csv(args.train_data_path)
train_sequences = train_data['sequence']
train_labels = train_data['label']
train_embeddings = np.array(calculate_tnf(train_sequences))

# Load testing data and extract sequences and labels
test_data = pd.read_csv(args.test_data_path)
test_sequences = test_data['sequence']
test_labels = test_data['label']
test_embeddings = np.array(calculate_tnf(test_sequences))

# Train XGBoost classifier
xgb_classifier = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=0,
    learning_rate=0.1,
    objective='binary:logistic',
    eval_metric='error'
)

xgb_classifier.fit(train_embeddings, train_labels)

# Make predictions on test data
test_predictions = xgb_classifier.predict(test_embeddings)

# Calculate accuracy
accuracy = accuracy_score(test_labels, test_predictions)
print(f"{args.train_data_path} Test Accuracy: {accuracy:.4f}")

# Save predictions and accuracy if output_path is provided
if args.output_path:
    # Extract the directory from the output path
    output_dir = os.path.dirname(args.output_path)
    
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Save predictions to the specified path
    test_data['xgb'] = test_predictions
    predictions_path = args.output_path
    test_data.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")

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

# Check if the 'xgb' key already exists
if "xgb" not in data:
    data["xgb"] = accuracy  # Add the new accuracy value
    with open(accuracy_json_path, "w") as f:
        json.dump(data, f, indent=4)  # Save the updated data
    print(f"Accuracy saved to {accuracy_json_path}")
else:
    print(f"Key 'xgb' already exists in {accuracy_json_path}. Accuracy not updated.")

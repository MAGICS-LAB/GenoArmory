import os
import pandas as pd
import json
from transformers import AutoTokenizer

# Initialize tokenizer at the start
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")


def generate_subword_freq(data_dir):
    """Generate subword token frequency statistics from DNA sequences"""

    # Dictionary to store token frequencies for each dataset
    all_dataset_freqs = {}

    # Get all subdirectories in data_dir
    dataset_folders = [
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    ]

    # Process each dataset folder
    for folder in sorted(dataset_folders):
        dataset_path = os.path.join(data_dir, folder, "cat.csv")

        if os.path.exists(dataset_path):
            # Dictionary to store token frequencies for current dataset
            token_freqs = {}

            # Read sequences from cat.csv
            df = pd.read_csv(dataset_path)
            sequences = df["sequence"].tolist()

            # Tokenize each sequence and count frequencies
            for seq in sequences:
                tokens = tokenizer(seq, return_tensors="pt")["input_ids"][0]

                # Convert token IDs to strings
                token_strings = tokenizer.convert_ids_to_tokens(tokens)

                # Update frequency counts
                for token in token_strings:
                    if token not in token_freqs:
                        token_freqs[token] = 0
                    token_freqs[token] += 1

            # Sort by frequency and store in all_dataset_freqs
            sorted_freqs = dict(
                sorted(token_freqs.items(), key=lambda x: x[1], reverse=True)
            )

            # Save individual dataset frequencies to JSON
            output_path = os.path.join(data_dir, folder, "subword_frequencies.json")
            with open(output_path, "w") as f:
                json.dump(sorted_freqs, f, indent=4)


# Generate frequencies for GUE datasets
gue_dir = "GUE"
generate_subword_freq(gue_dir)



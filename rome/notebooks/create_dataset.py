from transformers import AutoTokenizer
import random
import json
import numpy as np
import pandas as pd


def create_dna_dataset_with_tokenizer(dna_sequence, tokenizer, known_id = 0):
    # Tokenize the DNA sequence
    tokens = tokenizer.tokenize(dna_sequence)
    
    # Step 1: Randomly select a subword position to split into prompt and prediction
    split_index = random.randint(1, len(tokens) - 1)  # Avoid splitting at the very beginning or end
    prompt_tokens = tokens[:split_index]
    prediction_tokens = tokens[split_index:]

    # Step 2: Randomly select a subword in prompt to be the subject
    subject_index = random.randint(0, len(prompt_tokens) - 1)
    subject = prompt_tokens[subject_index]
    template_tokens = prompt_tokens[:subject_index] + ["{}"] + prompt_tokens[subject_index + 1:]

    # Step 3: First subword in prediction is the attribute
    attribute = prediction_tokens[0]

    # Convert lists back to strings
    prompt = "".join(prompt_tokens)
    template = "".join(template_tokens)
    prediction = "".join(prediction_tokens)


    # Create the dataset entry in the desired format
    dataset_entry = {
        "known_id": known_id,
        "subject": subject,
        "attribute": attribute,
        "template": template,
        "prediction": prediction,
        "prompt": prompt,
    }

    return dataset_entry

def process_sequences_from_csv(csv_path, json_output_path, tokenizer):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Assume the DNA sequences are in a column named 'sequence'
    results = []
    for idx, row in df.iterrows():
        dna_sequence = row['sequence']
        dataset_entry = create_dna_dataset_with_tokenizer(dna_sequence, tokenizer, known_id=idx)
        results.append(dataset_entry)

    # Save the results to a JSON file
    with open(json_output_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print("file saved at " + json_output_path)

# Example usage
tokenizer = AutoTokenizer.from_pretrained("/projects/p32013/DNABERT-meta/meta-100M")
csv_base_path = "/projects/p32013/DNABERT-meta/GUE/"  
json_output_base_path = "/projects/p32013/DNABERT-meta/rome/notebooks/dna_data/"

# List of checkpoints
checkpoints = [
    "H3", "H3K14ac", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K79me3", "H3K9ac", 
    "H4", "H4ac", "prom_core_all", "prom_core_notata", "prom_core_tata", "prom_300_all", 
    "prom_300_notata", "prom_300_tata", "tf0", "tf1", "tf2", "tf3", "tf4", "0", "1", "2", 
    "3", "4", "reconstructed", "covid"
]

for checkpoint in checkpoints:
    csv_path = csv_base_path + checkpoint + "/test.csv"
    json_output_path = json_output_base_path + checkpoint + "_test.json"
    process_sequences_from_csv(csv_path, json_output_path, tokenizer)


import random
import numpy as np
from transformers import AutoTokenizer, AutoModel, BertConfig, AutoModelForSequenceClassification
import argparse
from tqdm import tqdm
import pandas as pd
import torch


# Load the pretrained Hugging Face tokenizer
def load_huggingface_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

# Tokenize the DNA sequence and compute change probabilities
def tokenize_and_mutate(dna_sequence, tokenizer, change_rate=0.1):
    # Tokenize the DNA sequence and retrieve offset mappings
    encoded = tokenizer(dna_sequence, return_offsets_mapping=True)
    tokens = encoded.tokens()
    offsets = encoded['offset_mapping']
    
    # Determine the average number of mutations based on change rate
    avg_changes = int(len(tokens) * change_rate)
    mutation_positions = np.random.choice(len(tokens), avg_changes, replace=False)

    # Mutate DNA sequence using vocabulary-based sub-token replacement
    mutated_sequence = list(dna_sequence)
    vocab_tokens = list(tokenizer.get_vocab().keys())  # Retrieve vocab tokens

    for pos in mutation_positions:
        start, end = offsets[pos]  # Get the original sequence position for each token
        for i in range(start, end):  # Iterate over each nucleotide in the position range
            possible_changes = [token for token in vocab_tokens if token != mutated_sequence[i]]
            if possible_changes:
                mutated_sequence[i] = random.choice(possible_changes)  # Mutate with a random vocab token

    return "".join(mutated_sequence)


def tokenize_and_mutate2(dna_sequence, tokenizer, change_rate=0.1):
    # Tokenize and obtain offsets
    encoded = tokenizer(dna_sequence, return_offsets_mapping=True)
    tokens = encoded['input_ids']
    offsets = encoded['offset_mapping']
    
    # Determine number of subword tokens to change
    avg_changes = int(len(tokens) * change_rate)
    mutation_positions = np.random.choice(len(tokens), avg_changes, replace=False)

    # Convert DNA sequence to a mutable list
    mutated_sequence = list(dna_sequence)
    nucleotides = ['A', 'T', 'C', 'G']

    # Apply mutations based on subword token positions
    for pos in mutation_positions:
        start, end = offsets[pos]  # Start and end positions of the subword in the original sequence
        if start != end:  # Check if offset is valid for mutation
            for i in range(start, end):
                # Choose a nucleotide different from the current one for mutation
                possible_changes = [n for n in nucleotides if n != mutated_sequence[i]]
                mutated_sequence[i] = random.choice(possible_changes)

    return "".join(mutated_sequence)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default="/projects/p32013/DNA/FT/DNABERT-2-FT/finetune/output_zhihan_Full_double/zhihan_dnabert2_Full_double_0")
parser.add_argument('--input', default='/projects/p32013/DNABERT-meta/BERT-Attack/GUE/mouse/0/cat.csv')
parser.add_argument('--output', default='/projects/p32013/DNABERT-meta/BERT-Attack/GUE/mouse/0/new.csv')
parser.add_argument('--change_rate', default=0.087, type=float)
parser.add_argument('--num_label', default=2)
args = parser.parse_args()

data = pd.read_csv(args.input)

# Load the pretrained tokenizer once outside the loop
tokenizer = load_huggingface_tokenizer(args.model_name)
config = BertConfig.from_pretrained(args.model_name, num_labels=args.num_label)
model = AutoModelForSequenceClassification.from_pretrained(args.model_name, trust_remote_code=True, config=config).to('cuda')
# Apply mutations
for i in tqdm(range(len(data))): 
    dna_sequence = data.at[i, 'sequence']
    
    # Apply mutation with specified change rate
    mutated_sequence = tokenize_and_mutate(dna_sequence, tokenizer, change_rate=args.change_rate)
    
    inputs = tokenizer.encode_plus(mutated_sequence, None, add_special_tokens=True, max_length=128, )
    input_ids, token_type_ids = torch.tensor(inputs["input_ids"]), torch.tensor(inputs["token_type_ids"])
    attention_mask = torch.tensor([1] * len(input_ids))
    seq_len = input_ids.size(0)
    orig_probs = model(input_ids.unsqueeze(0).to('cuda'),
                        attention_mask.unsqueeze(0).to('cuda'),
                        token_type_ids.unsqueeze(0).to('cuda')
                        )
    orig_probs = orig_probs[0].squeeze()
    orig_probs = torch.softmax(orig_probs, -1)
    orig_label = torch.argmax(orig_probs)

    

    # Update the new sequence column
    data.at[i, 'new_sequence'] = mutated_sequence
    data.at[i, 'new_label'] = int(orig_label.item())
    
    
# Save the mutated sequences to the output file
data.to_csv(args.output, index=False)

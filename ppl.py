import lmppl
from datasets import load_dataset
from transformers import AutoTokenizer

# Define paths
clm_path = "/projects/p32013/DNABERT-meta/meta-100M"
dataset_path = "/projects/p32013/DNABERT-meta/data/validation"

# Initialize the scorer
scorer = lmppl.LM(clm_path)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(clm_path)

# Load the dataset
dataset = load_dataset("arrow", data_files={"train": f"{dataset_path}/data-00000-of-00001.arrow"})
dataset = dataset["train"]  # Extract train split

# Calculate perplexity for each example and return the average
def calculate_avg_ppl(dataset, scorer, tokenizer, limit=100):
    total_ppl = 0
    count = 0

    # Limit dataset to first `limit` entries
    subset = dataset.select(range(limit)) if len(dataset) > limit else dataset

    for example in subset:
        input_ids = example.get("input_ids", None)  # Adjust field name based on dataset
        if input_ids:
            text = tokenizer.decode(input_ids, skip_special_tokens=True).replace(" ", "")  # Convert IDs back to text
            print(text)
            try:
                ppl = scorer.get_perplexity(text)
                print(f"Perplexity: {ppl}")
                total_ppl += ppl
                count += 1
            except Exception as e:
                print(f"Error calculating perplexity for text: {text}\n{e}")
        else:
            print(f"Skipping invalid entry: {example}")

    avg_ppl = total_ppl / count if count > 0 else float('inf')
    return avg_ppl

# Compute and print average perplexity
average_ppl = calculate_avg_ppl(dataset, scorer, tokenizer, limit=100)
print(f"Average Perplexity: {average_ppl}")

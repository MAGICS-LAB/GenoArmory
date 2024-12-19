# Install necessary libraries
# !pip install torch transformers textattack datasets
# export HF_HOME="/projects/p32013/.cache/"

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from textattack.attack_recipes import BERTAttackLi2020
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import HuggingFaceDataset

# Load a pre-trained BERT model for sequence classification
model_name = "textattack/bert-base-uncased-imdb"  # You can replace this with your own model
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Wrapping the model with TextAttack's HuggingFaceModelWrapper
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

# Initialize the BERTAttack attack recipe
attack = BERTAttackLi2020.build(model_wrapper)

# Load the IMDb dataset from Hugging Face
dataset = HuggingFaceDataset("imdb", split="test")

# Perform the attack and print results
print("Running BERTAttack on the first 5 test samples...")

for i, example in enumerate(dataset):
    if i >= 1:  # Limit the number of samples to attack for demo purposes
        break

    original_text = example[0]  # The text input
    ground_truth_label = example[1]  # The actual label (0 or 1)

    print(f"\nOriginal Input (Sample {i+1}): {original_text}")
    
    # Perform the attack by passing both the input text and its ground truth label
    print("Attack")
    result = attack.attack(original_text, ground_truth_label)
    print("Attack done")

    print(f"Adversarial Input (Sample {i+1}): {result.perturbed_text()}")

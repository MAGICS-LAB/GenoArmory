import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForSequenceClassification, BertConfig
import argparse
import pandas as pd
import numpy as np


class GCGConfig:
    def __init__(self, num_steps=250, search_width=512, topk=256, n_replace=1, batch_size=None):
        self.num_steps = num_steps  # Number of optimization steps
        self.search_width = search_width  # Number of candidate sequences to consider
        self.topk = topk  # Number of top tokens to select based on gradient
        self.n_replace = n_replace  # Number of tokens to replace at each step
        self.batch_size = batch_size  # Batch size for loss computation

class GCGResult:
    def __init__(self, best_loss, best_ids, losses, pred_labels):
        self.best_loss = best_loss  # Best loss observed
        self.best_ids = best_ids  # Optimized token sequence
        self.losses = losses  # Track all losses during optimization
        self.pred_labels = pred_labels  # Track all predicted labels

class GCGClassification:
    def __init__(self, model, tokenizer, config: GCGConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config


    def sample_ids_from_grad(self, input_ids, grad):
        """Sample token replacements based on gradients to optimize classification loss."""
        topk_tokens = (-grad).topk(self.config.topk, dim=-1).indices  # Get top tokens based on gradients
        new_input_ids = input_ids.clone()

        # Sample new tokens for n_replace positions
        sampled_positions = torch.argsort(torch.rand(input_ids.size(), device=grad.device))[:, :self.config.n_replace]
        for pos in sampled_positions:
            new_input_ids[:, pos] = topk_tokens[:, pos].squeeze(1)
        
        return new_input_ids


    def compute_loss_and_grad(self, input_ids, token_type_ids, attention_mask, target_label):
        """Compute loss and gradient for classification."""
        # Get embeddings for input tokens
        embedding_layer = self.model.get_input_embeddings()  # Get the model's embedding layer
        input_embeds = embedding_layer(input_ids)  # Get the embeddings for the input tokens

        # Clone the embeddings to ensure they are leaf variables and set requires_grad=True
        input_embeds = input_embeds.clone().requires_grad_(True)
        input_embeds.retain_grad()
        # Forward pass with embeddings instead of input_ids
        outputs = self.model(
            inputs_embeds=input_embeds,  # Pass embeddings instead of input_ids
            attention_mask=attention_mask  # Use the usual attention mask
        )
        logits = outputs.logits  # Logits for classification
        
        # Compute cross-entropy loss for classification
        loss = F.cross_entropy(logits, target_label)
        
        # Backward pass to compute gradients
        loss.backward()

        return loss, input_embeds.grad  # Gradients w.r.t. the input embeddings

    def run(self, input_text: str, label: int):
        """Run the GCG optimization for classification."""
        inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs.input_ids.to(self.model.device)
        token_type_ids = inputs.token_type_ids.to(self.model.device) if "token_type_ids" in inputs else None
        attention_mask = inputs.attention_mask.to(self.model.device)
        target_label = torch.tensor([label], device=self.model.device)
        
        best_ids = input_ids.clone()
        best_loss = float("inf")
        all_losses = []
        all_labels = []

        for step in range(self.config.num_steps):
            # Compute loss and gradient for current tokens using embeddings
            loss, grad = self.compute_loss_and_grad(input_ids, token_type_ids, attention_mask, target_label)
            
            # Keep track of best sequence based on the loss
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_ids = input_ids.clone()

            all_losses.append(loss.item())

            # Sample new token sequences based on gradient
            new_input_ids = self.sample_ids_from_grad(input_ids, grad)

            # Reset gradients
            self.model.zero_grad()

            # Update current input ids with sampled ids
            input_ids = new_input_ids

            # Log progress
            with torch.no_grad():
                pred_labels = torch.argmax(self.model(input_ids, attention_mask, token_type_ids=token_type_ids).logits, dim=1).cpu().numpy()
                all_labels.append(pred_labels)
                print(f"Step {step + 1}/{self.config.num_steps}, Loss: {loss.item()}, Predicted Label: {pred_labels}")

        return GCGResult(best_loss, best_ids, all_losses, all_labels)

# Example of using the GCGClassification for an MLM classification task
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='zhihan1996/DNABERT-2-117M')
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--input_text', type=str, default='This is a sample input for testing the classification.')
    parser.add_argument('--output', type=str, default='output.txt')
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_config = BertConfig.from_pretrained(args.model,num_labels=args.num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, config= model_config, trust_remote_code = True ).cuda()  # Binary classification

    config = GCGConfig(num_steps=100, search_width=64, topk=10, n_replace=3)

    gcg = GCGClassification(model, tokenizer, config)
    data = pd.read_csv(args.input_text)
    for i in range(len(data)):
        input_text = data['sequence'][i]
        label = data['label'][i]
        target_label = 1 if label==0 else 0
        # Run GCG optimization
        result = gcg.run(input_text, target_label)

        print(f"Best Loss: {result.best_loss}")
        print(f"Optimized Sequence: {tokenizer.decode(result.best_ids[0])}")
        
    

if __name__ == "__main__":
    main()

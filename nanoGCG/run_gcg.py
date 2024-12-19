import argparse
import pandas as pd
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, BertConfig
import nanogcg
from tqdm import tqdm
from functools import partial
import torch
import lmppl
from nanogcg import GCGConfig
import numpy as np
import os
import pandas as pd

def init_worker(model_dir, mlm_path, num_label):
    global model, tokenizer, scorer
    # Load models and tokenizers inside each worker
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    scorer = lmppl.LM(model_dir)

    # tokenizer_mlm = AutoTokenizer.from_pretrained(
    #     mlm_path,
    #     model_max_length=128,
    #     padding_side="right",
    #     use_fast=True,
    #     trust_remote_code=True
    # )

    # config_mlm = BertConfig.from_pretrained(mlm_path, num_labels=num_label)
    # mlm_model = AutoModelForSequenceClassification.from_pretrained(mlm_path, config=config_mlm, trust_remote_code=True).to("cuda")
    # mlm_model.eval()

result_buffer = []

def process_row(row):
    try:
        message = row['sequence']
        label = row['label']


        config = GCGConfig(
            num_steps=10,
            search_width=8,
            topk=4,
            seed=42,
            verbosity="WARNING",
            batch_size=10,
            optim_str_init= "GATTCCATAATTGTCACTCTGAGAGAGCAAATGCAAAGAGAAATCTTCAGGTTGATGTCGTTGTTCATGGACATACCTCCAGTGCAACCAAACGAGCAATTCACTTGGGAATACGTTGACAAAGACAAGAAAATCCACACTATCAAATCGACTCCGTTAGAATTTGCCTCCAAATACGCAAAATTGGACCCTTCCACGCCAGTCTCATTGATCAATGATCCAAGACACCCATATGGTAAATTAATTAAGATCGATCGTTTAGGAAACGTCCTTGGCGGAGATGCCGTGATTTACTTAAATGTTGACAATGAAACACTATCTAAATTGGTTGTTAAGAGATTACAAAATAACAAAGCTGTCTTTTTTGGATCTCACACTCCAAAGTTCATGGACAAGAAAACTGGTGTCATGGATATTGAATTGTGGAACTATCCTGCCATCGGCTATAATTTACCTCAGCAAAAGGCATCGCGTATTAGATACCATGAAAGTTTGATGAC",
            n_replace = 1,
            buffer_size = 1,
        )


        result = nanogcg.run(model, tokenizer, message, scorer, config)
        final = message + result.best_string

        org_ppl = scorer.get_perplexity(message)
        new_ppl = scorer.get_perplexity(final)
        add_ppl = scorer.get_perplexity(result.best_string)
        sample_ppl = scorer.get_perplexity(message[:len(result.best_string)])

        org = message
        new = final
        add = result.best_string

        result_buffer.append({
                    'Original Perplexity': org_ppl,
                    'New Perplexity': new_ppl,
                    'Add Perplexity': add_ppl,
                    'Sample Perplexity': sample_ppl,
                    'Original Sequence': org,
                    'New Sequence': new,
                    'Added Sequence': add,
                })

        print(len(result_buffer))
        if len(result_buffer) >= 5:
            save_results_to_csv(args.save_path)


        return org_ppl, new_ppl, add_ppl, sample_ppl
    except Exception as e:
        print(f"Error processing row: {e}")
        return None

def save_results_to_csv(save_path):
    global result_buffer
    output_csv_path = save_path
    
    results_df = pd.DataFrame(result_buffer)
    print(results_df)
    if os.path.exists(output_csv_path):
        results_df.to_csv(output_csv_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(output_csv_path, index=False)

    result_buffer = []

def parallel_process(data, model_dir, mlm_path, num_label, num_worker=8):
    pool = mp.Pool(num_worker, initializer=init_worker, initargs=(model_dir, mlm_path, num_label))
    results = list(tqdm(pool.imap(process_row, [row for _, row in data.iterrows()]), total=len(data)))
    
    pool.close()
    pool.join()
    
    return results

if __name__ == "__main__":
    mp.set_start_method('forkserver', force=True)

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data CSV file")
    parser.add_argument("--mlm_path", type=str, required=True, help="Path to the masked language model")
    parser.add_argument("--save_path", type=str, required=True, help="Path to the save result")
    parser.add_argument("--num_label", type=int, required=True, help="Number of labels for the MLM model")
    parser.add_argument("--num_worker", type=int, required=False, default=8, help="Number of workers for multiprocessing")

    args = parser.parse_args()

    # Load data
    data = pd.read_csv(args.data_path)

    # Run parallel processing
    results = parallel_process(data, model_dir="/projects/p32013/DNABERT-meta/meta-100M", mlm_path=args.mlm_path, num_label=args.num_label, num_worker=args.num_worker)

    # Separate results into lists
    org, new, add, sample = [], [], [], []

    # Assuming 'results' is a list of tuples where each tuple contains (org_ppl, new_ppl)
    for result in results:
        if result:
            org_ppl, new_ppl, add_ppl, sample_ppl = result
            org.append(org_ppl)
            new.append(new_ppl)
            add.append(add_ppl)
            sample.append(sample_ppl)

    # Calculate the mean perplexities
    org_ppls = np.mean(org)
    new_ppls = np.mean(new)
    add_ppls = np.mean(add)
    sample_ppls = np.mean(sample)

    # Print the perplexity values (without multiplying by 100 unless necessary)
    print(f"Original Perplexity: {org_ppls:.2f}")
    print(f"New Perplexity: {new_ppls:.2f}")
    print(f"Add Perplexity: {add_ppls:.2f}")
    print(f"Sample Perplexity: {sample_ppls:.2f}")

    # def calculate_accuracy(predicted, real):
    #     correct = sum(p == r for p, r in zip(predicted, real))
    #     total = len(real)
    #     accuracy = correct / total
    #     return accuracy

    # # Calculate and print accuracy
    # accuracy = calculate_accuracy(org, real)
    # print(f"Original Accuracy: {accuracy * 100:.2f}%")

    # accuracy = calculate_accuracy(predicted, real)
    # print(f"Accuracy: {accuracy * 100:.2f}%")

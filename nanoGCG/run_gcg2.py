import argparse
import pandas as pd
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
import nanogcg
from tqdm import tqdm
import torch
import lmppl
from nanogcg import GCGConfig
import numpy as np
import os

def init_worker(model_dir):
    global model, tokenizer, scorer
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    scorer = lmppl.LM(model_dir)

def process_row(row, shared_list, lock, save_path):
    try:
        message = row['sequence']
        label = row['label']

        config = GCGConfig(
            num_steps=100,
            search_width=128,
            topk=64,
            seed=42,
            verbosity="WARNING",
            batch_size=10,
            optim_str_init="GATTCCATAATTGTCACTCTGAGAGAGCAAATGCAAAGAGAAATCTTCAGGTTGATGTCGTTGTTCATGGACATACCTCCAGTGCAACCAAACGAGCAATTCACTTGGGAATACGTTGACAAAGACAAGAAAATCCACACTATCAAATCGACTCCGTTAGAATTTGCCTCCAAATACGCAAAATTGGACCCTTCCACGCCAGTCTCATTGATCAATGATCCAAGACACCCATATGGTAAATTAATTAAGATCGATCGTTTAGGAAACGTCCTTGGCGGAGATGCCGTGATTTACTTAAATGTTGACAATGAAACACTATCTAAATTGGTTGTTAAGAGATTACAAAATAACAAAGCTGTCTTTTTTGGATCTCACACTCCAAAGTTCATGGACAAGAAAACTGGTGTCATGGATATTGAATTGTGGAACTATCCTGCCATCGGCTATAATTTACCTCAGCAAAAGGCATCGCGTATTAGATACCATGAAAGTTTGATGAC",
            n_replace=1,
            buffer_size=1,
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

        with lock:
            shared_list.append({
                'Original Perplexity': org_ppl,
                'New Perplexity': new_ppl,
                'Add Perplexity': add_ppl,
                'Sample Perplexity': sample_ppl,
                'Original Sequence': org,
                'New Sequence': new,
                'Added Sequence': add,
            })

            # Save results to CSV every five entries
            if len(shared_list) >= 5:
                save_results_to_csv(shared_list, save_path, lock)

        return org_ppl, new_ppl, add_ppl, sample_ppl

    except Exception as e:
        print(f"Error processing row: {e}")
        return None

def save_results_to_csv(shared_list, save_path, lock):
    with lock:
        output_csv_path = save_path
        shared_list_copy = list(shared_list)  # Create a copy to avoid issues while saving

        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

        results_df = pd.DataFrame(shared_list_copy)
        if os.path.exists(output_csv_path):
            results_df.to_csv(output_csv_path, mode='a', header=False, index=False)
        else:
            results_df.to_csv(output_csv_path, index=False)

        # Clear shared list after saving
        shared_list.clear()

def parallel_process(data, model_dir, save_path, num_worker=48):
    # Use Manager to create a shared list and lock
    with mp.Manager() as manager:
        shared_list = manager.list()
        lock = manager.Lock()

        # Define the pool with initializer
        pool = mp.Pool(num_worker, initializer=init_worker, initargs=(model_dir,))
        
        # Use functools.partial to pass additional arguments to process_row
        from functools import partial
        process_func = partial(process_row, shared_list=shared_list, lock=lock, save_path=save_path)
        
        # Run parallel processing
        list(tqdm(pool.imap(process_func, [row for _, row in data.iterrows()]), total=len(data)))

        pool.close()
        pool.join()

        # Final save of any remaining records in the buffer
        if len(shared_list) > 0:
            save_results_to_csv(shared_list, save_path)

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
    parallel_process(data, model_dir="/projects/p32013/DNABERT-meta/meta-100M", num_worker=args.num_worker, save_path=args.save_path)

import os
import json
import torchvision.models as models
from deepfool import deepfool_nlp
from loader import DNA
from Normalize import Normalize
import warnings
from universal_pert import universal_perturbation_nlp
warnings.filterwarnings("ignore")
import numpy as np
from torch.utils.data import DataLoader
import torch
import random
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, BertConfig
from bert_layers import BertForSequenceClassification
import argparse
import sklearn
import gc

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])
    predictions = np.argmax(logits, axis=-1)
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }


def evaluate_model(dataloader, model, device):
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[3].cpu().numpy()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.detach().cpu().numpy()
            all_logits.append(logits)
            all_labels.append(labels)
            
            # 清理批次数据
            del input_ids, attention_mask, outputs, logits

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return calculate_metric_with_sklearn(all_logits, all_labels)


def evaluate_model_with_perturbation(dataloader, model, v, device):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[3].cpu().numpy()

            embeds = model.bert.embeddings.word_embeddings(input_ids)
            perturbed_embeds = embeds + v.to(device)

            outputs = model(inputs_embeds=perturbed_embeds, attention_mask=attention_mask)
            logits = outputs.logits.detach().cpu().numpy()

            all_logits.append(logits)
            all_labels.append(labels)
            
            # 清理批次数据
            del input_ids, attention_mask, embeds, perturbed_embeds, outputs, logits

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return calculate_metric_with_sklearn(all_logits, all_labels)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list",
    )

    parser.add_argument(
        "--model_type",
        default='',
        type=str,
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="Path to save attack results",
    )

    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True
    )

    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )

    parser.add_argument(
        "--worker", default=16, type=int, help="Number of Worker",
    )

    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )

    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )

    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )

    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument('--n_gpu', type=int, default=1)

    parser.add_argument('--num_label', type=int, default=2)

    args = parser.parse_args()

    try:
        print(f"Starting task: {args.task_name}")
        set_seed(args)
        
        # 清理GPU内存
        clear_gpu_memory()
        
        if args.model_type == 'bert':
            config = BertConfig.from_pretrained(
                args.config_name if args.config_name else args.model_name_or_path,
                num_labels=args.num_label,
                cache_dir=args.cache_dir if args.cache_dir else None,
                trust_remote_code=True,
            )
        else:
            config = AutoConfig.from_pretrained(
                args.config_name if args.config_name else args.model_name_or_path,
                num_labels=args.num_label,
                cache_dir=args.cache_dir if args.cache_dir else None,
                trust_remote_code=True,
            )
        if args.model_type == 'nt':
            config.token_dropout = False

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
            trust_remote_code=True,
        )
        
        if args.model_type == 'bert':
            model = BertForSequenceClassification.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None,
                trust_remote_code=True,
            )
            
        ds = DNA(args)

        epsilon = 10.0
        net = model.cuda()
        print("Model loaded successfully")

        print('Loading data...')
        X, val_loader = ds.make_loaders(args, tokenizer, workers=args.worker, batch_size=args.batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Evaluating clean model on validation set...")
        clean_metrics = evaluate_model(val_loader, net, device)
        print("Clean validation metrics:", clean_metrics)

        print('Computing universal perturbation...')
        v = universal_perturbation_nlp(X, val_loader, net, tokenizer, xi=epsilon)

        print("Evaluating perturbed model on validation set...")
        perturbed_metrics = evaluate_model_with_perturbation(val_loader, net, v, device)
        print("Clean validation metrics:", clean_metrics)
        print("Perturbed validation metrics:", perturbed_metrics)

        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "eval_results.json")

        output_data = {
            "clean_validation_metrics": clean_metrics,
            "perturbed_validation_metrics": perturbed_metrics,
            "task_name": args.task_name
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=4)

        print(f"Metrics written to {output_path}")
        print(f"Task {args.task_name} completed successfully!")
        
    except Exception as e:
        print(f"Error in task {args.task_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # 确保清理资源
        try:
            del model, net, tokenizer, ds
        except:
            pass
        clear_gpu_memory()
        
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
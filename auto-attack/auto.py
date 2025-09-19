from loader import DNA
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from torch.utils.data import DataLoader
import torch
import random
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, BertConfig
from bert_layers import BertForSequenceClassification
import argparse
import sklearn
from autoattack import AutoAttack
import json
import os
from datetime import datetime

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

    for batch in dataloader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[3].cpu().numpy()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.detach().cpu().numpy()
        all_logits.append(logits)
        all_labels.append(labels)

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return calculate_metric_with_sklearn(all_logits, all_labels)

def evaluate_model_with_perturbation(dataloader, model, v, device):
    model.eval()
    all_logits = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[3].cpu().numpy()

        embeds = model.bert.embeddings.word_embeddings(input_ids)
        perturbed_embeds = embeds + v.to(device)

        with torch.no_grad():
            outputs = model(inputs_embeds=perturbed_embeds, attention_mask=attention_mask)
            logits = outputs.logits.detach().cpu().numpy()

        all_logits.append(logits)
        all_labels.append(labels)

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return calculate_metric_with_sklearn(all_logits, all_labels)

def save_attack_results(results, args, output_path):
    """保存攻击结果到JSON文件"""
    result_data = {
        "task_name": args.task_name,
        "model_name": args.model_name_or_path,
        "model_type": args.model_type,
        "epsilon": 10.0,  # 从代码中的epsilon值
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "attack_results": results
    }
    
    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"Results saved to: {output_path}")

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

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)

set_seed(args)
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

print('Loading data...')
train_loader, val_loader = ds.make_loaders(args, tokenizer, workers=args.worker, batch_size=args.batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Starting adversarial attack for task: {args.task_name}')
print(f'Epsilon: {epsilon}')
print(f'Model: {args.model_name_or_path}')

adversary = AutoAttack(net, norm='L1', eps=epsilon, version='custom', attacks_to_run=['apgd-ce'])

attack_results = []
total_samples = 0
successful_attacks = 0

for batch_idx, batch in enumerate(val_loader):
    print(f'Processing batch {batch_idx + 1}...')
    
    # Support both 3-tuple and 4-tuple batches
    if len(batch) == 4:
        X = batch[0].to(device)
        attention_mask = batch[1].to(device)
        token_type_ids = batch[2].to(device)
        y = batch[3].to(device)
    elif len(batch) == 3:
        X = batch[0].to(device)
        attention_mask = batch[1].to(device)
        y = batch[2].to(device)
        token_type_ids = None
    else:
        raise ValueError(f"Unexpected batch size: {len(batch)}")
    
    X_embeds = model.bert.embeddings.word_embeddings(X)
    
    # 运行攻击
    if token_type_ids is not None:
        x_adv = adversary.run_standard_evaluation(X_embeds, y, bs=args.batch_size, 
                                                attention_mask=attention_mask, 
                                                token_type_ids=token_type_ids)
    else:
        x_adv = adversary.run_standard_evaluation(X_embeds, y, bs=args.batch_size, 
                                                attention_mask=attention_mask)
    
    # 计算攻击成功率等统计信息
    batch_size = X.shape[0]
    total_samples += batch_size
    
    # 这里可以添加更多的结果分析
    batch_result = {
        "batch_idx": batch_idx,
        "batch_size": batch_size,
        "original_shape": X_embeds.shape,
        "adversarial_shape": x_adv.shape if x_adv is not None else None
    }
    attack_results.append(batch_result)
    
    print(f'Batch {batch_idx + 1} completed. Samples processed: {batch_size}')
    
    # 只处理第一个batch用于测试（根据原代码的break）
    break

# 保存详细结果
results_summary = {
    "total_samples": total_samples,
    "epsilon": epsilon,
    "attack_method": "apgd-ce",
    "norm": "L1",
    "batch_results": attack_results
}

# 保存到文件
output_file = os.path.join(args.output_dir, f"{args.task_name}_attack_results.json")
save_attack_results(results_summary, args, output_file)

print(f'\nAttack completed for task: {args.task_name}')
print(f'Total samples processed: {total_samples}')
print(f'Results saved to: {output_file}')
# -*- coding: utf-8 -*-
# @Time    : 2020/6/10
# @Author  : Linyang Li
# @Email   : linyangli19@fudan.edu.cn
# @File    : attack.py


import warnings
import os
import random
import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import BertConfig, AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoModelForCausalLM
import copy
import argparse
import numpy as np
import lmppl
from evaluate import load

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

filter_words = []
filter_words = set(filter_words)


def divide_dna_sequence(sequence, min_len=1, max_len=50):
    words = []
    start = 0
    
    # Continue splitting until the entire sequence is processed
    while start < len(sequence):
        # Randomly choose a word length between min_len and max_len
        word_length = random.randint(min_len, min_len + (max_len - min_len))
        
        # Ensure we do not exceed the sequence length
        word = sequence[start:start + word_length]
        words.append(word)
        
        # Move the starting point forward
        start += word_length
    
    # Join the words with spaces and return the result
    return ' '.join(words)

def get_tokenized_dna(sequence, tokenizer):
    tokens = tokenizer.tokenize(sequence)
    return ' '.join(tokens)

def get_sim_embed(embed_path, sim_path):
    id2word = {}
    word2id = {}

    with open(embed_path, 'r', encoding='utf-8') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in id2word:
                id2word[len(id2word)] = word
                word2id[word] = len(id2word) - 1

    cos_sim = np.load(sim_path)
    return cos_sim, word2id, id2word


def get_data_cls(data_path):
    lines = open(data_path, 'r', encoding='utf-8').readlines()[1:]
    features = []
    for i, line in enumerate(lines):
        split = line.strip('\n').split(',')
        label = int(split[-1])
        seq = split[0]

        features.append([seq, label])
    return features


class Feature(object):
    def __init__(self, seq_a, label):
        self.label = label
        self.seq = seq_a
        self.seq_down = seq_a
        self.final_adverse = seq_a
        self.query = 0
        self.change = 0
        self.success = 0
        self.sim = 0.0
        self.changes = []
        self.org_ppl = 0
        self.ppl = 0


def _tokenize(seq, tokenizer):
    seq = seq.replace('\n', '')
    words = seq.split(' ')

    sub_words = []
    keys = []
    index = 0
    for word in words:
        sub = tokenizer.tokenize(word)
        sub_words += sub
        keys.append([index, index + len(sub)])
        index += len(sub)

    return words, sub_words, keys


def _get_masked(words):
    len_text = len(words)
    masked_words = []
    for i in range(len_text - 1):
        masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
    # list of words
    return masked_words


def calculate_perplexity(sequence, clm_path):
    tokenizer = AutoTokenizer.from_pretrained(clm_path)
    model = AutoModelForCausalLM.from_pretrained(clm_path)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize the input sequence
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"].to(device)

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # Cross-entropy loss

    # Compute perplexity
    perplexity = torch.exp(loss).item()
    return perplexity

def get_important_scores(words, scorer, clm_path, tokenizer, batch_size, max_length):
    """
    Calculate the importance scores of each word in the sequence using perplexity and return a dictionary.
    """
    # Generate the original text and calculate its perplexity
    original_text = ''.join(words)
    # print(f"Original text: {original_text}")
    original_ppl = scorer.get_perplexity(original_text)

    # Generate masked sequences
    masked_words = _get_masked(words)
    texts = [''.join(words) for words in masked_words]

    # Calculate perplexity scores for each masked sequence
    perplexities = [scorer.get_perplexity(text) for text in texts]

    # Calculate importance scores
    import_scores = np.array([original_ppl - ppl for ppl in perplexities])

    # Create a dictionary with index and importance scores
    result_dict = {str(idx): score for idx, score in enumerate(import_scores)}

    # Return the dictionary
    return result_dict


def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k
    if sub_len == 0:
        return words
        
    elif sub_len == 1:
        for (i,j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    #
    # print(words)
    return words


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    # substitutes L, k

    substitutes = substitutes[0:12, 0:4] # maximum BPE candidates

    # find all possible candidates 
    #print(substitutes)
    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes) # [ N, L ]
    all_substitutes = all_substitutes[:24].to('cuda')
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0] # N L vocab-size
    ppl = c_loss(word_predictions.view(N*L, -1), all_substitutes.view(-1)) # [ N*L ] 
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1)) # N  
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    
    for word in word_list:
        tokens = [tokenizer.convert_ids_to_tokens(int(i)) for i in word]
        #text = tokenizer.convert_tokens_to_string(tokens)
        text = ''.join(tokens)
        final_words.append(text)
    return final_words


def attack(feature, clm_path, mlm_model, tokenizer, k, batch_size, max_length=512, cos_mat=None, w2i={}, i2w={}, use_bpe=1, scorer=None):
    # MLM-process
    words, sub_words, keys = _tokenize(feature.seq, tokenizer)
    #words = [i for i in words[0]]
    # original label
    
    inputs = tokenizer.encode_plus(feature.seq, None, add_special_tokens=True, max_length=max_length, )
    input_ids, token_type_ids = torch.tensor(inputs["input_ids"]), torch.tensor(inputs["token_type_ids"])
    
    attention_mask = torch.tensor([1] * len(input_ids))
    seq_len = input_ids.size(0)
    

    sub_words = ['[CLS]'] + sub_words[:max_length - 2] + ['[SEP]']
    input_ids_ = torch.tensor([tokenizer.convert_tokens_to_ids(sub_words)])
    word_predictions = mlm_model(input_ids_.to('cuda'))[0].squeeze()  # seq-len(sub) vocab
    word_pred_scores_all, word_predictions = torch.topk(word_predictions, k, -1)  # seq-len k

    word_predictions = word_predictions[1:len(sub_words) + 1, :]
    word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]
    
    # print(scorer.get_perplexity("ACGTAGCTAG"))
    # print(calculate_perplexity("ACGTAGCTAG", clm_path))
    # exit(0)
    result_dict = get_important_scores(words,  scorer, clm_path,
                                            tokenizer, batch_size, max_length)
    print(result_dict)
    return result_dict


def evaluate(features, scorer=None, clm_path=None):
    do_use = 0
    use = None
    sim_thres = 0
    # evaluate with USE

    if do_use == 1:
        cache_path = ''
        import tensorflow as tf
        import tensorflow_hub as hub
    
        class USE(object):
            def __init__(self, cache_path):
                super(USE, self).__init__()

                self.embed = hub.Module(cache_path)
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.Session()
                self.build_graph()
                self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

            def build_graph(self):
                self.sts_input1 = tf.placeholder(tf.string, shape=(None))
                self.sts_input2 = tf.placeholder(tf.string, shape=(None))

                sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
                sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
                self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
                clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
                self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

            def semantic_sim(self, sents1, sents2):
                sents1 = [s.lower() for s in sents1]
                sents2 = [s.lower() for s in sents2]
                scores = self.sess.run(
                    [self.sim_scores],
                    feed_dict={
                        self.sts_input1: sents1,
                        self.sts_input2: sents2,
                    })
                return scores[0]

            use = USE(cache_path)


    total = 0
    total_q = 0
    total_change = 0
    total_word = 0
    org_ppl = []
    ppl = []
    
    for feat in features:
    
        total_q += feat.query
        total_change += feat.change
        total_word += len(feat.seq.split(' '))
        op = scorer.get_perplexity(feat.seq.replace(' ', ''))
        org_ppl.append(op)
        feat.org_ppl = op
        fp = scorer.get_perplexity(feat.final_adverse)
        ppl.append(fp)
        feat.ppl = fp


        total += 1


    query = float(total_q / total)
    change_rate = float(total_change / total_word)

    ppl_mean = np.mean(ppl)
    org_ppl_mean = np.mean(org_ppl)

    print('origin_ppl/final_ppl {:.6f}/ {:.6f}'.format(org_ppl_mean, ppl_mean))

def dump_features(features, output):
    outputs = []

    for feature in features:
        outputs.append({'label': feature.label,
                        'change': feature.change,
                        'num_word': len(feature.seq.split(' ')),
                        'query': feature.query,
                        'changes': feature.changes,
                        'seq_a': feature.seq.replace(' ', ''),
                        'adv': feature.final_adverse.replace(' ', ''),
                        'org_ppl': feature.org_ppl,
                        'ppl': feature.ppl,
                        })
    output_json = output
    json.dump(outputs, open(output_json, 'w'), indent=2)

    print('finished dump')


def run_attack():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="./data/xxx")
    parser.add_argument("--mlm_path", type=str, help="xxx mlm")

    parser.add_argument("--output_dir", type=str, help="train file")
    parser.add_argument("--use_sim_mat", type=int, help='whether use cosine_similarity to filter out atonyms')
    parser.add_argument("--start", type=int, help="start step, for multi-thread process")
    parser.add_argument("--end", type=int, help="end step, for multi-thread process")
    parser.add_argument("--num_label", type=int, )
    parser.add_argument("--use_bpe", type=int, )
    parser.add_argument("--k", type=int, )
    parser.add_argument("--threshold_pred_score", type=float, )
    parser.add_argument('--clm_path', type=str, default='/projects/p32013/DNABERT-meta/meta-100M')


    args = parser.parse_args()
    data_path = str(args.data_path)
    mlm_path = str(args.mlm_path)
    clm_path = str(args.clm_path)
    output_dir = str(args.output_dir)
    num_label = args.num_label
    use_bpe = args.use_bpe
    k = args.k
    start = args.start
    end = args.end
    threshold_pred_score = args.threshold_pred_score
    scorer = lmppl.LM(args.clm_path)
    # scorer = load("perplexity", module_type="metric", token=True)
    print('start process')

    tokenizer_mlm = AutoTokenizer.from_pretrained(
        mlm_path,
        model_max_length=128,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )

    # tokenizer_clm = AutoTokenizer.from_pretrained(clm_path)

    config_atk = BertConfig.from_pretrained(mlm_path)
    
    mlm_model = AutoModelForMaskedLM.from_pretrained(mlm_path, config=config_atk, trust_remote_code=True)
    mlm_model.to('cuda')

    
    features = get_data_cls(data_path)
    print('loading sim-embed')
    
    if args.use_sim_mat == 1:
        cos_mat, w2i, i2w = get_sim_embed('data_defense/counter-fitted-vectors.txt', 'data_defense/cos_sim_counter_fitting.npy')
    else:        
        cos_mat, w2i, i2w = None, {}, {}

    print('finish get-sim-embed')
    result_output = []

    with torch.no_grad():
        for index, feature in enumerate(features[start:end]):
            seq_a, label = feature
            seq_a = get_tokenized_dna(seq_a, tokenizer_mlm)
            feat = Feature(seq_a, label)
            result = attack(feat, clm_path, mlm_model, tokenizer_mlm, k, batch_size=32, max_length=128,
                          cos_mat=cos_mat, w2i=w2i, i2w=i2w, use_bpe=use_bpe, scorer=scorer)

            result_output.append(result)

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    # Save the result_output to a JSON file
    with open(output_dir, "w") as f:
        json.dump(result_output, f, indent=4)




if __name__ == '__main__':
    run_attack()

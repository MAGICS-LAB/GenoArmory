import lmppl
import json
from transformers import AutoTokenizer
import numpy as np

def get_tokenized_dna(sequence, tokenizer):
    tokens = tokenizer.tokenize(sequence)
    return ' '.join(tokens)

tokenizer = AutoTokenizer.from_pretrained(
        'zhihan1996/DNABERT-2-117M',
        model_max_length=128,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )

clm_path = "/projects/p32013/DNABERT-meta/meta-100M"


scorer = lmppl.LM(clm_path)

# checkpoints = [
#     "H3", "H3K14ac", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K79me3", 
#     "H3K9ac"
# ]

checkpoints = [
     "prom_300_tata"
]

#"prom_core_all", "prom_core_notata", "prom_core_tata", "prom_300_all", "prom_300_notata", "prom_300_tata", "tf4"


for checkpoint in checkpoints:
    for i in range(1,3):
        try:
            dict_path = f'/projects/p32013/DNABERT-meta/BERT-Attack/results/meta/{checkpoint}/'
            json_file_path = dict_path + f"{checkpoint}-cat{i}.json"


            with open(json_file_path, 'r') as f:
                data = json.load(f)

            adv_sentences = [item['seq_a'] for item in data if 'seq_a' in item]
            if not adv_sentences:
                raise ValueError("no seq_a")

            dict = {}

            in_loop = True
            for item in data:
                dict_loop = {}
                if 'seq_a' in item and 'changes' in item:
                    org_ppl = scorer.get_perplexity(item['seq_a'])
                    seq_a = get_tokenized_dna(item['seq_a'], tokenizer)
                    for change in item['changes']:
                        seq_ls = seq_a.split(' ')
                        seq_ls[change[0]] = change[1]
                        new_seq = ''.join(seq_ls)
                        dict_loop[change[0]] = org_ppl - scorer.get_perplexity(new_seq.replace(' ',''))
                for key, value in dict_loop.items():
                    if key not in dict:
                        dict[key] = []
                    dict[key].append(value)

            # print(dict)
            np.save( dict_path + f"data-cat{i}.npy", dict, allow_pickle=True)
        except:
            print('There is no file called' + f"{checkpoint}-cat{i}.json")





#print(f"The Perplexity (PPL) of these sentence is: {avg_ppl}")

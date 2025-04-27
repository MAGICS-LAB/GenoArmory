# import os

# # for BERT target
# command = 'python attack_classification.py --dataset_path /projects/p32013/DNABERT-meta/BERT-Attack/GUE/mouse/0/cat.csv ' \
#           '--target_model bert ' \
#           '--target_model_path /scratch/hlv8980/GERM_ICML/output_zhihan_vanilla_Full_double/zhihan_vanilla_dnabert2_Full_double_0 ' \
#           '--max_seq_length 256 --batch_size 32 ' \
#           '--counter_fitting_embeddings_path  /projects/p32013/DNABERT-meta/TextFooler/embeddings/subword_dnabert2_embeddings.txt ' \
#           '--counter_fitting_cos_sim_path /projects/p32013/DNABERT-meta/TextFooler/cos_sim_counter_fitting/cos_sim_counter_fitting.npy ' \
#           '--USE_cache_path /projects/p32013/DNABERT-meta/TextFooler/tf_cache ' \
#           '--nclasses 2 --quantize'

# command1 = 'python attack_classification.py --dataset_path /projects/p32013/DNABERT-meta/BERT-Attack/GUE/mouse/1/cat.csv ' \
#           '--target_model bert ' \
#           '--target_model_path /scratch/hlv8980/GERM_ICML/output_zhihan_vanilla_Full_double/zhihan_vanilla_dnabert2_Full_double_1 ' \
#           '--max_seq_length 256 --batch_size 32 ' \
#           '--counter_fitting_embeddings_path  /projects/p32013/DNABERT-meta/TextFooler/embeddings/subword_dnabert2_embeddings.txt ' \
#           '--counter_fitting_cos_sim_path /projects/p32013/DNABERT-meta/TextFooler/cos_sim_counter_fitting/cos_sim_counter_fitting.npy ' \
#           '--USE_cache_path /projects/p32013/DNABERT-meta/TextFooler/tf_cache ' \
#           '--nclasses 2'


# command2 = 'python attack_classification.py --dataset_path /projects/p32013/DNABERT-meta/BERT-Attack/GUE/mouse/2/cat.csv ' \
#           '--target_model bert ' \
#           '--target_model_path /scratch/hlv8980/GERM_ICML/output_zhihan_vanilla_Full_double/zhihan_vanilla_dnabert2_Full_double_2 ' \
#           '--max_seq_length 256 --batch_size 32 ' \
#           '--counter_fitting_embeddings_path  /projects/p32013/DNABERT-meta/TextFooler/embeddings/subword_dnabert2_embeddings.txt ' \
#           '--counter_fitting_cos_sim_path /projects/p32013/DNABERT-meta/TextFooler/cos_sim_counter_fitting/cos_sim_counter_fitting.npy ' \
#           '--USE_cache_path /projects/p32013/DNABERT-meta/TextFooler/tf_cache ' \
#           '--nclasses 2'


# command3 = 'python attack_classification.py --dataset_path /projects/p32013/DNABERT-meta/BERT-Attack/GUE/mouse/3/cat.csv ' \
#           '--target_model bert ' \
#           '--target_model_path /scratch/hlv8980/GERM_ICML/output_zhihan_vanilla_Full_double/zhihan_vanilla_dnabert2_Full_double_3 ' \
#           '--max_seq_length 256 --batch_size 32 ' \
#           '--counter_fitting_embeddings_path  /projects/p32013/DNABERT-meta/TextFooler/embeddings/subword_dnabert2_embeddings.txt ' \
#           '--counter_fitting_cos_sim_path /projects/p32013/DNABERT-meta/TextFooler/cos_sim_counter_fitting/cos_sim_counter_fitting.npy ' \
#           '--USE_cache_path /projects/p32013/DNABERT-meta/TextFooler/tf_cache ' \
#           '--nclasses 2'


# command4 = 'python attack_classification.py --dataset_path /projects/p32013/DNABERT-meta/BERT-Attack/GUE/mouse/4/cat.csv ' \
#           '--target_model bert ' \
#           '--target_model_path /scratch/hlv8980/GERM_ICML/output_zhihan_vanilla_Full_double/zhihan_vanilla_dnabert2_Full_double_4 ' \
#           '--max_seq_length 256 --batch_size 32 ' \
#           '--counter_fitting_embeddings_path  /projects/p32013/DNABERT-meta/TextFooler/embeddings/subword_dnabert2_embeddings.txt ' \
#           '--counter_fitting_cos_sim_path /projects/p32013/DNABERT-meta/TextFooler/cos_sim_counter_fitting/cos_sim_counter_fitting.npy ' \
#           '--USE_cache_path /projects/p32013/DNABERT-meta/TextFooler/tf_cache ' \
#           '--nclasses 2'

# os.system(command)

import os
import subprocess

# Define the base command template
command_template = 'python attack_classification_general.py --dataset_path {dataset_path} ' \
                   '--target_model bert ' \
                   '--target_model_path {target_model_path} ' \
                   '--output_dir /projects/p32013/DNABERT-meta/TextFooler/output/quantize/{model}/{dataset_dir} ' \
                   '--max_seq_length 256 --batch_size 128 ' \
                   '--counter_fitting_embeddings_path /projects/p32013/DNABERT-meta/TextFooler/embeddings/subword_{model}_embeddings.txt ' \
                   '--counter_fitting_cos_sim_path /projects/p32013/DNABERT-meta/TextFooler/cos_sim_counter_fitting/cos_sim_counter_fitting_{model}.npy ' \
                   '--USE_cache_path /projects/p32013/DNABERT-meta/TextFooler/tf_cache ' \
                   '--nclasses 2 --quantize'

# Set the base directory for the datasets
base_dir = '/projects/p32013/DNABERT-meta/GUE'
model = 'dnabert'
dataset_dirs = ["H3", "H3K14ac", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K79me3", 
                "H3K9ac", "H4", "H4ac", "prom_core_all", "prom_core_notata", "prom_core_tata", 
                "prom_300_all", "prom_300_notata", "prom_300_tata", "tf0", "tf1", "tf2", 
                "tf3", "tf4", "0", "1", "2", "3", "4"]

# dataset_dirs = ["0"]

# Loop over each dataset directory
for dataset_dir in dataset_dirs:
    dataset_path = os.path.join(base_dir, dataset_dir, 'cat.csv')
    target_model_path = f"/scratch/hlv8980/Attack_Benchmark/models/{model}/{dataset_dir}/origin"

    # Check if the dataset file exists
    if os.path.exists(dataset_path):
        # Format the command with the current dataset and target model path
        print(f"Dataset file found:{dataset_path}")
        command = command_template.format(dataset_path=dataset_path, dataset_dir=dataset_dir, target_model_path=target_model_path, model=model)
        
        # Run the command using subprocess for better control
        subprocess.run(command, shell=True, check=True)
    else:
        print(f"Dataset file not found: {dataset_path}")

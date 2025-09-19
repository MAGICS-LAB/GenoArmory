import os
import argparse
import subprocess
parser = argparse.ArgumentParser(description='Run adversarial training pipeline for specified task.')
parser.add_argument('--task', type=str, required=True, help='Task name to process')

args = parser.parse_args()
task = args.task
 


# Set the base directory for the datasets
base_dir = 'GUE'

dataset_dirs = ["H3K9ac"]

# dataset_dirs = ["4"]

# Loop over each dataset directory
for dataset_dir in dataset_dirs:
  dataset_path = os.path.join(base_dir, dataset_dir, 'cat.csv')
  ### model ckpt
  target_model_path = f"/scratch/anonymous/Attack_Benchmark/models/{task}/{dataset_dir}/origin"

  # # Check if the dataset file exists
  # if os.path.exists(dataset_path):
  #     # Format the command with the current dataset and target model path
  #     print(f"Dataset file found:{dataset_path}")
  #     command = command_template.format(dataset_path=dataset_path, dataset_dir=dataset_dir, target_model_path=target_model_path)
      
  #     # Run the command using subprocess for better control
  #     subprocess.run(command, shell=True, check=True)
    # else:
    #     print(f"Dataset file not found: {dataset_path}")





  # 1.2 Use TextFooler as the attack method to produce adversarial examples:
  command3 = f'python attack_classification_simplified.py --dataset_path {dataset_path} ' \
            f'--target_model {task} ' \
            f'--target_model_path {target_model_path} ' \
            '--max_seq_length 256 --batch_size 32 ' \
            f'--counter_fitting_embeddings_path  TextFooler/embeddings/subword_{task}_embeddings.txt ' \
            f'--counter_fitting_cos_sim_path TextFooler/cos_sim_counter_fitting/cos_sim_counter_fitting_{task}.npy ' \
            '--USE_cache_path TextFooler/tf_cache ' \
            f'--nclasses 2 --output_dir adv_results/{task}/{dataset_dir}' 
          

  command4 = 'python get_pure_adversaries.py ' \
    f'--adversaries_path adv_results/{task}/{dataset_dir}/adversaries.txt ' \
    f'--output_path GUE/{dataset_dir}/{task}/attacked_data ' \
    '--times 1 ' \
    '--change 0 ' \
    '--txtortsv tsv ' \
    '--datasize 9662'

  # 1.3 Construct the training data
  command5 = 'python combine_data.py ' \
    f'--add_file GUE/{dataset_dir}/{task}/attacked_data/pure_adversaries.tsv ' \
    '--change_label 2 ' \
    f'--original_dataset GUE/{dataset_dir} ' \
    f'--output_path GUE/{dataset_dir}/{task}/combined_data/2times_adv_0-3/ ' \
    '--isMR 0'

  command6 = 'python run_simplification.py ' \
    '--complex_threshold 3000 ' \
    '--ratio 0.25 ' \
    '--syn_num 20 ' \
    '--most_freq_num 10 ' \
    '--simplify_version random_freq_v1 ' \
    f'--cos_sim_file TextFooler/cos_sim_counter_fitting/cos_sim_counter_fitting_{task}.npy ' \
    f'--counterfitted_vectors TextFooler/embeddings/subword_{task}_embeddings.txt ' \
    f'--file_to_simplify GUE/{dataset_dir}/{task}/combined_data/2times_adv_0-3/train.tsv ' \
    f'--output_path GUE/{dataset_dir}/{task}/simplified_data/2times_adv_0-3/ ' \
    f'--freq_file GUE/{dataset_dir}/subword_frequencies.json'

  command7 = 'python combine_data.py ' \
    f'--add_file GUE/{dataset_dir}/{task}/simplified_data/2times_adv_0-3/train.tsv ' \
    '--change_label 4 ' \
    f'--original_dataset GUE/{dataset_dir}/{task}/combined_data/2times_adv_0-3/ ' \
    f'--output_path GUE/{dataset_dir}/{task}/combined_data/4times_adv_0-7/ --isMR 0 '

  # Step2. Train our proposed model on the constructed training data
  command8 = 'WANDB_DISABLED=true python run_classification_adv.py ' \
    f'--task_name {dataset_dir} ' \
    '--max_seq_len 128 ' \
    '--do_train ' \
    '--do_eval ' \
    '--attention 2 ' \
    f'--data_dir GUE/{dataset_dir}/{task}/combined_data/4times_adv_0-7/ ' \
    f'--output_dir ADFAR/src/experiments/GUE/{dataset_dir}/{task}/4times_adv_double_0-7 ' \
    f'--model_name_or_path {target_model_path} ' \
    '--per_device_train_batch_size 2 ' \
    '--per_device_eval_batch_size 2 ' \
    '--save_total_limit 2 ' \
    '--learning_rate 3e-5 ' \
    '--num_train_epochs 5 ' \
    '--svd_reserve_size 0 ' \
    '--evaluation_strategy epoch ' \
    '--overwrite_output_dir ' \
    f'--model_type {task} ' \
    '--overwrite_cache'
    
  os.system(command3)
  os.system(command4)
  os.system(command5)
  os.system(command6)
  os.system(command7)
  os.system(command8)


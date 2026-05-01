#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tasks=( "prom_300_all" "prom_300_notata" "prom_300_tata" "tf0" "tf1" "tf2" "tf3" "tf4" "0" "1" "2" "3" "4")
model="nt1"

for task in "${tasks[@]}"
do
    echo "Running task: $task"
    python "${SCRIPT_DIR}/shap_dl_analysis2.py" \
        --data_dir GUE/${task}/fimba \
        --model_name_or_path /scratch/anonymous/Attack_Benchmark/models/${model}/${task}/origin \
        --task_name $task --num_label 2  \
        --max_seq_length 128 --batch_size 1 \
        --dataset_name $task --model_type ${model} \
        --overwrite_cache \
        --shap_output_file "${SCRIPT_DIR}/shap_dicts/shap_${model}_fimba_$task.pkl"

    python "${SCRIPT_DIR}/runatk_standalone.py" \
        --data_dir GUE/${task}/fimba \
        --model_name_or_path /scratch/anonymous/Attack_Benchmark/models/${model}/${task}/origin  \
        --task_name $task --num_label 2  --max_seq_length 128 \
        --shap_file "${SCRIPT_DIR}/shap_dicts/shap_${model}_fimba_$task.pkl" \
        --increase_fn --batch_size 64 --model_type ${model} \
        --output_dir fimba-attack/results/${model} \
        --overwrite_cache --overwrite_output_dir
done

echo "Finished nt1"

######################################

export CUDA_VISIBLE_DEVICES=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tasks=( "H3K4me2" "H3K4me3" "H3K79me3" "H3K9ac" "H4" "H4ac" "prom_core_all" "prom_core_notata" "prom_core_tata" "prom_300_all" "prom_300_notata" "prom_300_tata" "tf0" "tf1" "tf2" "tf3" "tf4" "0" "1" "2" "3" "4")
model="nt2"

for task in "${tasks[@]}"
do
    echo "Running task: $task"
    python "${SCRIPT_DIR}/shap_dl_analysis2.py" \
        --data_dir GUE/${task}/fimba \
        --model_name_or_path /scratch/anonymous/Attack_Benchmark/models/${model}/${task}/origin \
        --task_name $task --num_label 2  \
        --max_seq_length 128 --batch_size 1 \
        --dataset_name $task --model_type ${model} \
        --overwrite_cache \
        --shap_output_file "${SCRIPT_DIR}/shap_dicts/shap_${model}_fimba_$task.pkl"

    python "${SCRIPT_DIR}/runatk_standalone.py" \
        --data_dir GUE/${task}/fimba \
        --model_name_or_path /scratch/anonymous/Attack_Benchmark/models/${model}/${task}/origin  \
        --task_name $task --num_label 2  --max_seq_length 128 \
        --shap_file "${SCRIPT_DIR}/shap_dicts/shap_${model}_fimba_$task.pkl" \
        --increase_fn --batch_size 64 --model_type ${model} \
        --output_dir fimba-attack/results/${model} \
        --overwrite_cache --overwrite_output_dir
done

echo "Finished nt2"

######################################

export CUDA_VISIBLE_DEVICES=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tasks=( "H4" "H4ac" "prom_core_all" "prom_core_notata" "prom_core_tata" "prom_300_all" "prom_300_notata" "prom_300_tata" "tf0" "tf1" "tf2" "tf3" "tf4" "0" "1" "2" "3" "4")
model="og"

for task in "${tasks[@]}"
do
    echo "Running task: $task"
    python "${SCRIPT_DIR}/shap_dl_analysis2.py" \
        --data_dir GUE/${task}/fimba \
        --model_name_or_path /scratch/anonymous/Attack_Benchmark/models/${model}/${task}/origin \
        --task_name $task --num_label 2  \
        --max_seq_length 128 --batch_size 1 \
        --dataset_name $task --model_type ${model} \
        --overwrite_cache \
        --shap_output_file "${SCRIPT_DIR}/shap_dicts/shap_${model}_fimba_$task.pkl"

    python "${SCRIPT_DIR}/runatk_standalone.py" \
        --data_dir GUE/${task}/fimba \
        --model_name_or_path /scratch/anonymous/Attack_Benchmark/models/${model}/${task}/origin  \
        --task_name $task --num_label 2  --max_seq_length 128 \
        --shap_file "${SCRIPT_DIR}/shap_dicts/shap_${model}_fimba_$task.pkl" \
        --increase_fn --batch_size 64 --model_type ${model} \
        --output_dir fimba-attack/results/${model} \
        --overwrite_cache --overwrite_output_dir
done

echo "Finished og"
echo "All tasks completed."

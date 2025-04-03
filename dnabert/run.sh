export WANDB_DISABLED=true

export DATA_CACHE_DIR=".hf_data"
export MODEL_CACHE_DIR=".hf_cache"

module load gcc/12.3.0-gcc 

cd /projects/p32013/DNABERT-meta/dnabert

tasks=("H3" "H3K14ac" "H3K36me3" "H3K4me1" "H3K4me2" "H3K4me3" "H3K79me3" "H3K9ac" "H4" "H4ac" "prom_core_all" "prom_core_notata" "prom_core_tata" "prom_300_all" "prom_300_notata" "prom_300_tata" "tf0" "tf1" "tf2" "tf3" "tf4" "0" "1" "2" "3" "4")

for task in "${tasks[@]}"; do
    python train.py \
        --model_name_or_path "zhihan1996/DNABERT-2-117M" \
        --data_path "/projects/p32013/DNABERT-meta/GUE/${task}/textfooler" \
        --kmer -1 \
        --run_name "hyena_dna_${task}" \
        --model_max_length 256 \
        --per_device_train_batch_size 64 \
        --per_device_eval_batch_size 64 \
        --gradient_accumulation_steps 1 \
        --learning_rate 3e-5 \
        --num_train_epochs 4 \
        --fp16 \
        --save_steps 10000 \
        --output_dir "output_pipe/${task}/textfooler" \
        --evaluation_strategy steps \
        --eval_steps 200 \
        --warmup_ratio 0.05 \
        --logging_steps 100 \
        --overwrite_output_dir True \
        --log_level info \
        --find_unused_parameters False \
        --save_model False

done

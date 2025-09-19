export WANDB_DISABLED=true

export DATA_CACHE_DIR=".hf_data"
export MODEL_CACHE_DIR=".hf_cache"

module load gcc/12.3.0-gcc 

cd test

python train.py \
    --model_name_or_path "anonymous/NT2-500M-multi_species-finetuned-tf0" \
    --data_path "GUE/0" \
    --kmer -1 \
    --run_name "test" \
    --model_max_length 256 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-5 \
    --num_train_epochs 4 \
    --fp16 \
    --save_steps 200 \
    --output_dir "/scratch/anonymous/Attack_Benchmark/test" \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --warmup_ratio 0.05 \
    --logging_steps 100 \
    --overwrite_output_dir True \
    --log_level info \
    --find_unused_parameters False \
    --save_model False

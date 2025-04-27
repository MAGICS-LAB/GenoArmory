#!/usr/bin/env bash

#tasks=( "prom_300_all" "prom_300_notata" "prom_300_tata" "tf0" "tf1" "tf2" "tf3" "tf4" "0" "1" "2" "3" "4" )
tasks=( "H3" "H3K14ac" "H3K36me3" "H3K4me1" "H3K4me2" "H3K4me3" "H3K79me3" "H3K9ac" "H4" "H4ac" "prom_core_all" "prom_core_notata" "prom_core_tata" "prom_300_all" "prom_300_notata" "prom_300_tata" "tf0" "tf1" "tf2" "tf3" "tf4" "0" "1" "2" "3" "4" )

model="dnabert"

project_root="/projects/p32013/DNABERT-meta/FreeLB/huggingface-transformers"
log_dir="${project_root}/logs"
ckpt_dir="${project_root}/checkpoints"

mkdir -p "${log_dir}"
mkdir -p "${ckpt_dir}"

function runexp {
    local task=$1

    export GUE_DIR='/projects/p32013/DNABERT-meta/GUE'

    gpu=0                  
    mname="/scratch/hlv8980/Attack_Benchmark/models/${model}/${task}/origin"
    alr=1e-1
    amag=6e-1
    anorm=0
    asteps=2
    lr=1e-5
    bsize=32
    gas=1
    seqlen=256
    hdp=0.1
    adp=0
    ts=2000
    ws=100
    seed=42
    wd=1e-2
    model_type=${model}
    expname=${model_type}_${task}

    if [ "${model_type}" = "hyena" ]; then
        script="${project_root}/examples/run_glue_freelb3.py"
    else
        script="${project_root}/examples/run_glue_freelb2.py"
    fi

    python "${script}" \
      --model_type "${model_type}" \
      --model_name_or_path "${mname}" \
      --task_name "${task}" \
      --do_train \
      --do_eval \
      --do_lower_case \
      --data_dir "${GUE_DIR}/${task}" \
      --max_seq_length "${seqlen}" \
      --per_gpu_train_batch_size "${bsize}" --gradient_accumulation_steps "${gas}" \
      --learning_rate "${lr}" --weight_decay "${wd}" \
      --gpu "${gpu}" \
      --output_dir "${ckpt_dir}/${expname}/" \
      --hidden_dropout_prob "${hdp}" --attention_probs_dropout_prob "${adp}" \
      --adv-lr "${alr}" --adv-init-mag "${amag}" --adv-max-norm "${anorm}" --adv-steps "${asteps}" \
      --expname "${expname}" --evaluate_during_training \
      --max_steps "${ts}" --warmup_steps "${ws}" --seed "${seed}" \
      --logging_steps 100 --save_steps 100 \
      --num_label 2 \
      --overwrite_output_dir \
      --overwrite_cache \
      > "${log_dir}/${expname}.log"

    echo "---
${task} finish
---"
}

for task in "${tasks[@]}"; do
    runexp "$task"
done

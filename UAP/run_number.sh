#!/bin/bash

export HF_HOME="/projects/p32013/.cache/"
cd UAP

# tasks=("H3" "H3K14ac" "H3K36me3" "H3K4me1" "H3K4me2" "H3K4me3" "H3K79me3" "H3K9ac" "H4" "H4ac" "prom_core_all" "prom_core_notata" "prom_core_tata" "prom_300_all" "prom_300_notata" "prom_300_tata" "tf0" "tf1" "tf2" "tf3" "tf4" "0" "1" "2" "3" "4")
tasks=("0" "1" "2" "3")
model='bert'

# 记录开始时间
start_time=$(date)
echo "Batch processing started at: ${start_time}"

# 成功和失败的任务计数
success_count=0
fail_count=0
failed_tasks=()

for task in "${tasks[@]}"; do
    echo "=================================================="
    echo "Starting task: ${task}"
    echo "Time: $(date)"
    echo "=================================================="
    
    # 为每个任务创建单独的缓存目录
    task_cache_dir="UAP/.cache/${task}"
    mkdir -p ${task_cache_dir}
    
    # 设置任务特定的环境变量
    export TRANSFORMERS_CACHE="${task_cache_dir}"
    export HF_DATASETS_CACHE="${task_cache_dir}/datasets"
    
    # 运行任务
    python search_new.py \
        --data_dir GUE/${task} \
        --model_name_or_path anonymous/DNABERT-2-finetuned-${task} \
        --task_name ${task} \
        --num_label 2 \
        --n_gpu 1 \
        --max_seq_length 256 \
        --batch_size 128 \
        --output_dir UAP/results/${task} \
        --model_type bert \
        --cache_dir ${task_cache_dir}
    
    # 检查退出状态
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "Task ${task} finished successfully"
        ((success_count++))
        
        # 清理任务特定的缓存（可选，如果磁盘空间紧张）
        # rm -rf ${task_cache_dir}
    else
        echo "Task ${task} failed with exit code ${exit_code}"
        ((fail_count++))
        failed_tasks+=(${task})
    fi
    
    echo "Task ${task} completed. Success: ${success_count}, Failed: ${fail_count}"
    echo ""
    
    # 在任务之间添加短暂延迟，让系统有时间清理资源
    sleep 3
done

# 总结报告
end_time=$(date)
echo "=================================================="
echo "Batch processing completed!"
echo "Start time: ${start_time}"
echo "End time: ${end_time}"
echo "Total tasks: ${#tasks[@]}"
echo "Successful: ${success_count}"
echo "Failed: ${fail_count}"

if [ ${fail_count} -gt 0 ]; then
    echo "Failed tasks: ${failed_tasks[*]}"
fi

echo "=================================================="

# 如果有失败的任务，返回非零退出码
if [ ${fail_count} -gt 0 ]; then
    exit 1
else
    exit 0
fi
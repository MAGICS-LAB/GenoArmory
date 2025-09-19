export HF_HOME="/projects/p32013/.cache/"
cd /projects/p32013/DNABERT-meta/UAP

# tasks=("H3" "H3K14ac" "H3K36me3" "H3K4me1" "H3K4me2" "H3K4me3" "H3K79me3" "H3K9ac" "H4" "H4ac" "prom_core_all" "prom_core_notata" "prom_core_tata" "prom_300_all" "prom_300_notata" "prom_300_tata" "tf0" "tf1" "tf2" "tf3" "tf4" "0" "1" "2" "3" "4")
tasks=("H3K36me3" "H3K4me2" "H3K4me3")
model='bert'

for task in "${tasks[@]}"; do
    python search.py --data_dir /projects/p32013/DNABERT-meta/GUE/${task} \
                --model_name_or_path magicslabnu/DNABERT-2-finetuned-${task} \
                --task_name ${task} --num_label 2 --n_gpu 1 \
                --max_seq_length 256 --batch_size 128 \
                --output_dir /projects/p32013/DNABERT-meta/UAP/results/${task} \
                --model_type bert \
                --cache_dir /projects/p32013/DNABERT-meta/UAP/.cache
    
    echo "${task} finished"
done
# for task in "${tasks[@]}"; do
#     (
#         python search.py --data_dir /projects/p32013/DNABERT-meta/GUE/${task} \
#             --model_name_or_path magicslabnu/DNABERT-2-finetuned-${task} \
#             --task_name ${task} --num_label 2 --n_gpu 1 \
#             --max_seq_length 256 --batch_size 128 \
#             --output_dir /projects/p32013/DNABERT-meta/UAP/results/${task} \
#             --model_type bert \
#             --cache_dir /projects/p32013/DNABERT-meta/UAP/.cache
#     ) 2>&1 | tee "log_${task}.txt"

#     echo "${task} finished"
# done




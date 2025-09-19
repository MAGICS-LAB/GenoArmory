cd PGD

#tasks=("H3" "H3K14ac" "H3K36me3" "H3K4me1" "H3K4me2" "H3K4me3" "H3K79me3" "H3K9ac" "H4" "H4ac" "prom_core_all" "prom_core_notata" "prom_core_tata" "prom_300_all" "prom_300_notata" "prom_300_tata" "tf0" "tf1" "tf2" "tf3" "tf4" "0" "1" "2" "3" "4")
tasks=("tf4")

model='og'

for task in "${tasks[@]}"; do
    python test.py --data_dir GUE/${task} \
                --model_name_or_path /scratch/anonymous/Attack_Benchmark/models/${model}/${task}/origin \
                --task_name 0 --num_label 2 --n_gpu 1 \
                --max_seq_length 256 --batch_size 16 \
                --output_dir PGD/results/${model}/${task} \
                --model_type og --overwrite_cache
    
    echo "${task} finished"
done



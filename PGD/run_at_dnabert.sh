cd /projects/p32013/DNABERT-meta/PGD

#tasks=("H3" "H3K14ac" "H3K36me3" "H3K4me1" "H3K4me2" "H3K4me3" "H3K79me3" "H3K9ac" "H4" "H4ac" "prom_core_all" "prom_core_notata" "prom_core_tata" "prom_300_all" "prom_300_notata" "prom_300_tata" "tf0" "tf1" "tf2" "tf3" "tf4" "0" "1" "2" "3" "4")
tasks=("H3K14ac" "H3K36me3" )
model='dnabert'

for task in "${tasks[@]}"; do
    python test.py --data_dir /projects/p32013/DNABERT-meta/GUE/${task}/five_percent \
                --model_name_or_path /scratch/hlv8980/Attack_Benchmark/models/${model}/${task}/textfooler/checkpoint-200 \
                --task_name 0 --num_label 2 --n_gpu 1 \
                --max_seq_length 256 --batch_size 16 \
                --output_dir /projects/p32013/DNABERT-meta/PGD/results/AT/${model}/${task} \
                --model_type bert --overwrite_cache \
                --tokenizer_name /scratch/hlv8980/Attack_Benchmark/models/${model}/${task}/origin 
    echo "${task} finished"
done



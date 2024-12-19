cd /projects/p32013/DNABERT-meta/TNF

checkpoints=( "H3" "H3K14ac" "H3K36me3" "H3K4me1" "H3K4me2" "H3K4me3" "H3K79me3" "H3K36me3" "H3K9ac" "H4" "H4ac" "prom_core_all" "prom_core_notata" "prom_core_tata" "prom_300_all" "prom_300_notata" "prom_300_tata" "tf0" "tf1" "tf2" "tf3" "tf4" "0" "1" "2" "3" "4")

for checkpoint in "${checkpoints[@]}"; do
    python xgb.py\
    --train_data_path /projects/p32013/DNABERT-meta/GUE/$checkpoint/cat.csv \
    --test_data_path /projects/p32013/DNABERT-meta/BERT-Attack/results/self_consist/$checkpoint/${checkpoint}_self.csv \
    --output_path /projects/p32013/DNABERT-meta/TNF/results/$checkpoint/xgb.csv \
    --log_path /projects/p32013/DNABERT-meta/TNF/results/$checkpoint/result.json \

done
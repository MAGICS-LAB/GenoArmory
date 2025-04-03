export HF_TOKEN='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'
export HF_HOME="/projects/p32013/.cache/"

# python bertattack.py --data_path  /projects/p32013/DNA/data/GUE/EMP/H3/test.csv --mlm_path zhihan1996/DNABERT-2-117M --tgt_path zhihan1996/DNABERT-2-117M --output_dir results/H3-test1.csv --num_label 2 --use_bpe 1 --k 48 --threshold_pred_score 0 --start 0 --end 10

python hyenaattack.py --data_path  /projects/p32013/DNABERT-meta/BERT-Attack/GUE/mouse/0/cat.csv --mlm_path zhihan1996/DNABERT-2-117M --tgt_path /projects/p32013/DNABERT-meta/hyena-dna/hyena/output_pipe/0/origin --output_dir results/hyena/0-cat1.json --num_label 2 --use_bpe 0 --k 48 --threshold_pred_score 0 --start 0 --end 100
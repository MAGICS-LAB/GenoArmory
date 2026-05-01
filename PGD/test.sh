cd PGD

python test.py --data_dir BERT-Attack/GUE/mouse/0 \
               --model_name_or_path nucleotide-transformer/nucleotide1/output_pipe/0/origin \
               --task_name 0 --num_label 2 --n_gpu 1 \
               --max_seq_length 256 --batch_size 32 \
               --output_dir PGD/results/0 \
               --model_type nt --overwrite_cache



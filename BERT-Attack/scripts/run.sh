cd BERT-Attack
cat BERT-Attack/scripts/prom_300_explain.txt | python batch_run.py --gpus 0,0,0,0,0,0 
# cat BERT-Attack/scripts/0.txt | python batch_run.py --gpus 0,0,0,0
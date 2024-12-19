cd /projects/p32013/DNABERT-meta/BERT-Attack
# cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/covid_target.txt | python batch_run.py --gpus 0,0,0,0,0,0,0,0
cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/reconstructed.txt | python batch_run.py --gpus 0,0,0,0,0,0,0,0
# cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/reconstructed_target.txt | python batch_run.py --gpus 0,0,0,0,0,0,0,0
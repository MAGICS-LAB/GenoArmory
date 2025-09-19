cd BERT-Attack
# cat BERT-Attack/scripts/covid_target.txt | python batch_run.py --gpus 0,0,0,0,0,0,0,0
cat BERT-Attack/scripts/reconstructed.txt | python batch_run.py --gpus 0,0,0,0,0,0,0,0
# cat BERT-Attack/scripts/reconstructed_target.txt | python batch_run.py --gpus 0,0,0,0,0,0,0,0
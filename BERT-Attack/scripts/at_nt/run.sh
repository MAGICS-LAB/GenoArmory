cd BERT-Attack
cat BERT-Attack/scripts/at_nt/emp1.txt | python batch_run.py --gpus 0,0,0,0,0,0,0,0,0,0
#cat BERT-Attack/scripts/at_nt/mouse.txt | python batch_run.py --gpus 0,0,0,0,0
# cat BERT-Attack/scripts/at_nt/prom.txt | python batch_run.py --gpus 0,0,0,0,0,0
# cat BERT-Attack/scripts/at_nt/tf1.txt | python batch_run.py --gpus 0
# cat BERT-Attack/scripts/at_nt/tf.txt | python batch_run.py --gpus 0,0,0,0
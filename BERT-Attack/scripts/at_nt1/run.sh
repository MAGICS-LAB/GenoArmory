cd BERT-Attack
cat BERT-Attack/scripts/at_nt1/emp.txt | python batch_run.py --gpus 0,0,0,0,0,0,0,0,0,0
cat BERT-Attack/scripts/at_nt1/mouse.txt | python batch_run.py --gpus 0,0,0,0,0
cat BERT-Attack/scripts/at_nt1/prom.txt | python batch_run.py --gpus 0,0,0,0,0,0
cat BERT-Attack/scripts/at_nt1/tf1.txt | python batch_run.py --gpus 0
cat BERT-Attack/scripts/at_nt1/tf.txt | python batch_run.py --gpus 0,0,0,0
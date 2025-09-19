cd BERT-Attack
#cat BERT-Attack/scripts/at_og/emp.txt | python batch_run.py --gpus 0,0,0,0,0,0,0,0,0,0
cat BERT-Attack/scripts/at_og/mouse.txt | python batch_run.py --gpus 0,0,0,0,0
cat BERT-Attack/scripts/at_og/prom1.txt | python batch_run.py --gpus 0,0,0,0,0,0
cat BERT-Attack/scripts/at_og/tf1.txt | python batch_run.py --gpus 0
cat BERT-Attack/scripts/at_og/tf.txt | python batch_run.py --gpus 0,0,0,0
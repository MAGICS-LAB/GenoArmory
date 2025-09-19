cd BERT-Attack
cat BERT-Attack/scripts/all_og/emp.txt | python batch_run.py --gpus 0,0,0,0,0,0,0,0,0,0
cat BERT-Attack/scripts/all_og/mouse.txt | python batch_run.py --gpus 0,0,0,0,0
cat BERT-Attack/scripts/all_og/prom.txt | python batch_run.py --gpus 0,0,0,0,0,0
cat BERT-Attack/scripts/all_og/tf.txt | python batch_run.py --gpus 0,0,0,0,0
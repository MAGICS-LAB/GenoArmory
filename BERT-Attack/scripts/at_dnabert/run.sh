cd BERT-Attack
cat BERT-Attack/scripts/at_dnabert/emp1.txt | python batch_run.py --gpus 0,0,0,0,0,0,0,0,0,0
# cat BERT-Attack/scripts/at_dnabert/mouse.txt | python batch_run.py --gpus 0,0,0,0,0
# cat BERT-Attack/scripts/at_dnabert/prom.txt | python batch_run.py --gpus 0,0,0,0,0,0
# cat BERT-Attack/scripts/at_dnabert/tf1.txt | python batch_run.py --gpus 0
# cat BERT-Attack/scripts/at_dnabert/tf.txt | python batch_run.py --gpus 0,0,0,0
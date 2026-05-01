cd BERT-Attack
# cat BERT-Attack/scripts/at_hyena/emp.txt | python batch_run.py --gpus 0,0,0,0,0,0,0,0,0,0
# cat BERT-Attack/scripts/at_hyena/mouse.txt | python batch_run.py --gpus 0,0,0,0,0
# cat BERT-Attack/scripts/at_hyena/prom.txt | python batch_run.py --gpus 0,0,0,0,0,0
cat BERT-Attack/scripts/at_hyena/tf1.txt | python batch_run.py --gpus 0
cat BERT-Attack/scripts/at_hyena/tf.txt | python batch_run.py --gpus 0,0,0,0
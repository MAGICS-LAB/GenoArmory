cd BERT-Attack
cat BERT-Attack/scripts/freelb_nt/emp.txt | python batch_run.py --gpus 0,0,0,0,0,0,0,0,0,0
cat BERT-Attack/scripts/freelb_nt/mouse.txt | python batch_run.py --gpus 0,0,0,0,0
cat BERT-Attack/scripts/freelb_nt/prom.txt | python batch_run.py --gpus 0,0,0,0,0,0
cat BERT-Attack/scripts/freelb_nt/tf.txt | python batch_run.py --gpus 0,0,0,0,0
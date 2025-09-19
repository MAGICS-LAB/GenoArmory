cd BERT-Attack
# cat BERT-Attack/scripts/freelb_hyena/emp.txt | python batch_run.py --gpus 1,1,1,1,1,1,1,1,1,0
# cat BERT-Attack/scripts/freelb_hyena/mouse.txt | python batch_run.py --gpus 1,1,1,1,0
# cat BERT-Attack/scripts/freelb_hyena/prom.txt | python batch_run.py --gpus 1,1,1,1,1,0
cat BERT-Attack/scripts/freelb_hyena/tf.txt | python batch_run.py --gpus 0,0,0,0,0
# cat BERT-Attack/scripts/freelb_hyena/emp1.txt | python batch_run.py --gpus 0
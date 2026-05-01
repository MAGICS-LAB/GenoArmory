cd BERT-Attack
cat BERT-Attack/scripts/meta/mouse.txt | python batch_run.py --gpus 0,0,0,0,0
cat BERT-Attack/scripts/meta/prom.txt | python batch_run.py --gpus 0,0,0,0,0,0
cat BERT-Attack/scripts/meta/emp.txt | python batch_run.py --gpus 0,0,0,0,0,1,1,1,1,1
cat BERT-Attack/scripts/meta/test.txt | python batch_run.py --gpus 0,0,1,1
cat BERT-Attack/scripts/meta/test2.txt | python batch_run.py --gpus 0,0,1,1
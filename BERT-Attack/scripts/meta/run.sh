cd /projects/p32013/DNABERT-meta/BERT-Attack
cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/meta/mouse.txt | python batch_run.py --gpus 0,0,0,0,0
cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/meta/prom.txt | python batch_run.py --gpus 0,0,0,0,0,0
cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/meta/emp.txt | python batch_run.py --gpus 0,0,0,0,0,1,1,1,1,1
cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/meta/test.txt | python batch_run.py --gpus 0,0,1,1
cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/meta/test2.txt | python batch_run.py --gpus 0,0,1,1
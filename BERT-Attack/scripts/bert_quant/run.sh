cd BERT-Attack
export PYTHONPATH=$PYTHONPATH:$PWD
# cat BERT-Attack/scripts/bert_quant/emp2.txt | python batch_run.py --gpus 0,0,0,0
# cat BERT-Attack/scripts/bert_quant/mouse.txt | python batch_run.py --gpus 0,0,0
# cat BERT-Attack/scripts/bert_quant/prom.txt | python batch_run.py --gpus 0,0
# cat BERT-Attack/scripts/bert_quant/prom1.txt | python batch_run.py --gpus 0,0,0
# cat BERT-Attack/scripts/bert_quant/tf.txt | python batch_run.py --gpus 0,0,0,0,0
# cat BERT-Attack/scripts/bert_quant/emp.txt | python batch_run.py --gpus 0,0,0,0,0,0
cat BERT-Attack/scripts/bert_quant/extra.txt | python batch_run.py --gpus 0,0

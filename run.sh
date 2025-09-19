cd /projects/p32013/DNABERT-meta

## Visualization
# python GenoArmory.py --model_path anonymous_model visualize --folder_path BERT-Attack/results/meta/test --save_path BERT-Attack/results/meta/test/frequency.pdf

## Attack
# python GenoArmory.py --model_path anonymous_model attack --method pgd --params_file scripts/PGD/pgd_dnabert.json
# python GenoArmory.py --model_path anonymous_model attack --method fimba --params_file scripts/FIMBA/fimba_dnabert.json
# python GenoArmory.py --model_path anonymous_model attack --method textfooler --params_file scripts/TextFooler/textfooler_dnabert.json
# python GenoArmory.py --model_path anonymous_model attack --method bertattack --params_file scripts/BertAttack/bertattack_dnabert.json


## Defense
python GenoArmory.py --model_path anonymous_model defense --method freelb --params_file scripts/FreeLB/freelb_pgd_dnabert.json
# python GenoArmory.py --model_path anonymous_model defense --method adfar --params_file scripts/ADFAR/adfar_pgd_dnabert.json
# python GenoArmory.py --model_path anonymous_model defense --method at --params_file scripts/AT/at_pgd_dnabert.json

## Read Metadata
# python GenoArmory.py --model_path anonymous_model read --type attack --method TextFooler --model_name dnabert
# python GenoArmory.py --model_path anonymous_model read --type defense --method ADFAR --model_name dnabert --attack_method textfooler


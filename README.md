
<div align="center">
  <img src="asserts/logo.jpg" alt="Image" />
</div>

<div align="center">
<p align="center">
    <p align="center">A comprehensive toolkit for DNA sequence Adversarial Attack and Defense Benchmark.
    <br>
</p>


[![arXiv](https://img.shields.io/badge/arXiv-GenoArmory-ff0000.svg?style=for-the-badge)](https://arxiv.org/abs/2505.10983)  [![Github](https://img.shields.io/badge/GenoArmory-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/MAGICS-LAB/GenoArmory)  [![Hugging Face Pretrained](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md-dark.svg)](https://huggingface.co/collections/magicslabnu/gfm-67f4d4a9327ee4acdcb3806b) [![Hugging Face Dataset](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg)](https://huggingface.co/datasets/magicslabnu/GenoAdv) 
</div>

## Installation

You can install GenoArmory using pip:
```bash
pip install genoarmory
```
You can also install package from our source code:
```bash
git clone https://github.com/MAGICS-LAB/GenoArmory.git
conda create -n genoarmory pip=3.9
pip install .
```

## Quick Start

```python
# Initialize model
from GenoArmory import GenoArmory
import json
# You need to initialize GenoArmory with a model and tokenizer.
# For visualization, you don't need a real model/tokenizer, so you can use None if the method doesn't use them.
gen = GenoArmory(model=None, tokenizer=None)
params_file = '/projects/p32013/DNABERT-meta/scripts/PGD/pgd_dnabert.json'

# Visulization
gen.visualization(
    folder_path='/projects/p32013/DNABERT-meta/BERT-Attack/results/meta/test',
    output_pdf_path='/projects/p32013/DNABERT-meta/BERT-Attack/results/meta/test'
)

# Attack
if params_file:
  try:
      with open(params_file, "r") as f:
          kwargs = json.load(f)
  except json.JSONDecodeError as e:
      raise ValueError(f"Invalid JSON in params file '{params_file}': {e}")
  except FileNotFoundError:
      raise FileNotFoundError(f"Params file '{params_file}' not found.")

gen.attack(
    attack_method='pgd',
    model_path='magicslabnu/GERM',
    **kwargs
)
```

## Command Line Usage

GenoArmory can also be used from the command line:

```bash
# Attack
python GenoArmory.py --model_path magicslabnu/GERM attack --method pgd --params_file /projects/p32013/DNABERT-meta/scripts/PGD/pgd_dnabert.json

# Defense
python GenoArmory.py --model_path magicslabnu/GERM defense --method at --params_file /projects/p32013/DNABERT-meta/scripts/AT/at_pgd_dnabert.json

# Visualization
python GenoArmory.py --model_path magicslabnu/GERM visualize --folder_path /projects/p32013/DNABERT-meta/BERT-Attack/results/meta/test --save_path /projects/p32013/DNABERT-meta/BERT-Attack/results/meta/test/frequency.pdf


# Read MetaData
python GenoArmory.py --model_path magicslabnu/GERM read --type attack --method TextFooler --model_name dnabert

```

## Features

- Multiple attack methods:

  - BERT-Attack
  - TextFooler
  - PGD
  - FIMBA

- Defense methods:

  - ADFAR
  - FreeLB
  - Traditional Adversarial Training

- Visualization tools
- Artifact management
- Batch processing
- Command-line interface

## Documentation

For detailed documentation, visit [docs](We will release soon).

## License

This project is licensed under the MIT License.

## Citation

If you have any question regarding our paper or codes, please feel free to start an issue.

If you use GenoArmory in your work, please kindly cite our paper:

```
@misc{luo2025genoarmoryunifiedevaluationframework,
      title={GenoArmory: A Unified Evaluation Framework for Adversarial Attacks on Genomic Foundation Models}, 
      author={Haozheng Luo and Chenghao Qiu and Yimin Wang and Shang Wu and Jiahao Yu and Han Liu and Binghui Wang and Yan Chen},
      year={2025},
      eprint={2505.10983},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.10983}, 
}
```


<div align="center">
  <img src="asserts/logo.jpg" alt="Image" />
</div>

<div align="center">
<p align="center">
    <p align="center">A comprehensive toolkit for DNA sequence Adversarial Attack and Defense Benchmark.
    <br>
</p>

</div>

## Installation

You can install GenoArmory using pip:
```bash
pip install genoarmory
```
You can also install package from our source code:
```bash
git clone https://anonymous.git
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
params_file = 'scripts/PGD/pgd_dnabert.json'

# Visulization
gen.visualization(
    folder_path='BERT-Attack/results/meta/test',
    output_pdf_path='BERT-Attack/results/meta/test'
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
    model_path='anonymous_model',
    **kwargs
)
```

## Command Line Usage

GenoArmory can also be used from the command line:

```bash
# Attack
python GenoArmory.py --model_path anonymous_model attack --method pgd --params_file scripts/PGD/pgd_dnabert.json

# Defense
python GenoArmory.py --model_path anonymous_model defense --method at --params_file scripts/AT/at_pgd_dnabert.json

# Visualization
python GenoArmory.py --model_path anonymous_model visualize --folder_path BERT-Attack/results/meta/test --save_path BERT-Attack/results/meta/test/frequency.pdf


# Read MetaData
python GenoArmory.py --model_path anonymous_model read --type attack --method TextFooler --model_name dnabert

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


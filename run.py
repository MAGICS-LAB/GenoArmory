from GenoArmory import GenoArmory
import json
# You need to initialize GenoArmory with a model and tokenizer.
# For visualization, you don't need a real model/tokenizer, so you can use None if the method doesn't use them.
gen = GenoArmory(model=None, tokenizer=None)
params_file = 'scripts/PGD/pgd_dnabert.json'
# gen.visualization(
#     folder_path='BERT-Attack/results/meta/test',
#     output_pdf_path='BERT-Attack/results/meta/test'
# )


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
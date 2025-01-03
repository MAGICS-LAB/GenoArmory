import os
import transformers
import torch
from torch.utils.data import Dataset

# set it to prevent the warning message when use the model
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# Change this to the directory of the model
model_dir = "/home/hlv8980/DNA/meta-100M"

"""
Parameters:
- model_dir: str
    The directory of the model
- prompts: List[str]   
    A list of prompts to generate from. If not provided, the model will generate from scratch
- num_generation_from_each_prompt: int
    The number of sequences to generate from each prompt
- temperature: float
    The temperature of the sampling
- min_length: int
    The minimum length of the generated sequence (in tokens)
- max_length: int
    The maximum length of the generated sequence (in tokens)
- top_k: int   
    The top_k parameter for sampling
- presence_penalty: float
    Penalizes new tokens based on whether they appear in the generated text so far. 
    Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.
- frequency_penalty: float
    Penalizes new tokens based on their frequency in the generated text so far. 
    Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.
- repetition_penalty: float
    Penalizes new tokens based on whether they appear in the prompt and the generated text so far. 
    Values > 1 encourage the model to use new tokens, while values < 1 encourage the model to repeat tokens.
"""
def generate_sequences(
    model_dir, 
    prompts=[""],
    num_generation_from_each_prompt=100,
    temperature=0.7,
    min_length=128,
    max_length=1024, 
    top_k=50,     
    presence_penalty=0.0,
    frequency_penalty=0.0,
    repetition_penalty=1.0,
):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)

    prompts = ["[CLS]"+ p for p in prompts]

    input_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    print(input_ids)
    
    logits_list = []
    generated_sequences = []
    for input_id in input_ids:
    # Forward pass to get logits
      input_id = torch.tensor(input_id)
      #input_id = torch.cat((input_id, torch.tensor([3])))
      outputs = model.generate(
          input_ids=input_id.unsqueeze(0),
          max_new_tokens=1,  # Generate only one new token
          temperature=0.7,
          top_k=50,
          repetition_penalty=1.0,
          pad_token_id=tokenizer.pad_token_id,
          eos_token_id=tokenizer.eos_token_id,
          return_dict_in_generate=True,
          output_scores=True  # Include logits in the output
      )
      print(outputs.keys())
      for i, sequence in enumerate(outputs.sequences):
        print(tokenizer.decode(sequence))


"""
Example usage:
"""

# Generate from scratch
# generated_sequences = generate_sequences(model_dir)

# Generate from existing dna sequences 
dna_sequences = [
"TGAAGTGGTACAGAGATGTTCATTAAGCCAACAACTAGACCCATTGCTACGAAGAAGATCATGATGACACCGTGTGCGGTGAA",
        "CCCATAATGGAGGCCGACAAGCGATGCGAATTACCAATCACAAAGGTAACCGCCATGGTTTCGCCCAATGCCCGGCCCAGCCCC",
        ] # save the sequences in a list
generate_sequences(model_dir, prompts=dna_sequences, min_length=5, max_length=50, num_generation_from_each_prompt=1)

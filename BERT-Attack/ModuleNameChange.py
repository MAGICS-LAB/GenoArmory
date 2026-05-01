import torch
from safetensors.torch import load_file, save_file
from collections import OrderedDict
import os
import glob

def find_safetensor_files():
    """Find all model.safetensors files in the specified directory structure"""
    base_path = "/scratch/anonymous/Attack_Benchmark/models/ADFAR/GUE/GUE/og"
    pattern = os.path.join(base_path, "*/4times_adv_double_0-7/model.safetensors")
    return glob.glob(pattern)

def process_safetensor_file(file_path):
    """Process a single safetensor file, renaming keys from 'mistral.' to 'model.'"""
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return False

    print(f"\nProcessing: {file_path}")
    try:
        # Load the original weights
        original_weights = load_file(file_path)
        print(f"Loaded {len(original_weights)} tensors.")
    except Exception as e:
        print(f"Error loading file '{file_path}': {e}")
        return False

    renamed_weights = OrderedDict()
    renamed_count = 0

    print("Renaming keys (mistral. -> model.)...")
    for key, tensor in original_weights.items():
        new_key = key
        if key.startswith("mistral."):
            new_key = key.replace("mistral.", "model.", 1)
            renamed_count += 1
        renamed_weights[new_key] = tensor

    print(f"Processed {len(original_weights)} keys.")
    print(f"Renamed {renamed_count} keys starting with 'mistral.' to 'model.'")

    if renamed_count == 0:
        print("No keys were renamed. Skipping file save.")
        return True

    print(f"Saving renamed weights back to: {file_path}")
    try:
        save_file(renamed_weights, file_path)
        print("Successfully saved renamed weights.")
        return True
    except Exception as e:
        print(f"Error saving file '{file_path}': {e}")
        return False

def main():
    """Find and process all model.safetensors files"""
    files = find_safetensor_files()
    total_files = len(files)
    successful = 0
    failed = 0

    print(f"Found {total_files} files to process")
    
    for file_path in files:
        print(f"\nFolder: {os.path.dirname(file_path)}")
        if process_safetensor_file(file_path):
            successful += 1
        else:
            failed += 1

    print("\n=== Processing Summary ===")
    print(f"Total files processed: {total_files}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    main()


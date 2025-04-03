import os
import pandas as pd

def calculate_avg_sequence_length_for_all(root_folder):
    # List to store results for all found files
    results = []

    # Walk through all directories and subdirectories
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if "test.csv" in filenames:
            file_path = os.path.join(dirpath, "test.csv")
            folder_name = os.path.basename(dirpath)
            
            # Try reading the file and calculating the average sequence length
            try:
                data = pd.read_csv(file_path)
                
                if 'sequence' in data.columns:
                    # Calculate sequence lengths
                    data['seq_length'] = data['sequence'].apply(len)
                    # Calculate the average length
                    avg_length = data['seq_length'].mean()
                    
                    # Append the folder name and avg length to results
                    results.append((folder_name, avg_length))
                else:
                    results.append((folder_name, "No 'sequence' column"))
            except Exception as e:
                results.append((folder_name, f"Error: {str(e)}"))
    
    # Output results
    if results:
        print("Summary of Avg Sequence Lengths:")
        for folder, avg_length in results:
            print(f"Folder: {folder}, Avg Sequence Length: {avg_length}")
    else:
        print("No test.csv files were found.")

# Example usage
root_folder_path = "/projects/p32013/DNABERT-meta/BERT-Attack/GUE"  # Replace with the root folder path
calculate_avg_sequence_length_for_all(root_folder_path)
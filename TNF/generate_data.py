import os
import pandas as pd

# Define directories
source_root = "TNF/results"
dest_root = "GUE"

# Iterate through all subdirectories in the source root directory
for subdir in os.listdir(source_root):
    source_subdir_path = os.path.join(source_root, subdir)
    dest_subdir_path = os.path.join(dest_root, subdir, "self")

    if os.path.isdir(source_subdir_path) and os.path.isdir(dest_subdir_path):
        source_file = os.path.join(source_subdir_path, "self.csv")
        dest_train_file = os.path.join(dest_subdir_path, "train.csv")

        if os.path.exists(source_file) and os.path.exists(dest_train_file):
            # Load the source and destination CSV files
            source_df = pd.read_csv(source_file)
            dest_df = pd.read_csv(dest_train_file)

            # Concatenate the dataframes
            combined_df = pd.concat([dest_df, source_df], ignore_index=True)

            # Shuffle the combined dataframe
            shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)

            # Save back to the train.csv file
            shuffled_df.to_csv(dest_train_file, index=False)

            print(f"Concatenated and shuffled {source_file} into {dest_train_file}")
        else:
            print(f"Missing file: {source_file} or {dest_train_file}. Skipping.")


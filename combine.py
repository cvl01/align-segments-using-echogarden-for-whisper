import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

def windows_to_wsl_path(windows_path: str) -> str:
    """
    Convert a Windows file path to a WSL Ubuntu file path.
    
    Args:
    - windows_path (str): The Windows file path to convert.
    
    Returns:
    - str: The converted WSL Ubuntu file path.
    """
    import os
    
    # Normalize path to ensure consistent format
    windows_path = os.path.normpath(windows_path)
    
    # Split the path into drive and the rest of the path
    drive, path = os.path.splitdrive(windows_path)
    
    # Remove the colon from the drive (e.g., 'C:' -> 'c')
    drive_letter = drive[0].lower()
    
    # Replace backslashes with forward slashes
    path = path.replace('\\', '/')
    
    # Construct the WSL path
    wsl_path = f"/mnt/{drive_letter}{path}"
    
    return wsl_path


# Define and parse command line arguments
parser = argparse.ArgumentParser(description="Process engine output files from a specified directory.")
parser.add_argument("input_directory", type=str, help="The root directory containing subdirectories")
parser.add_argument("--engine", type=str, default="dtw-ra", choices=["dtw", "dtw-ra", "whisper"], help="The engine to use (default: dtw-ra)")
args = parser.parse_args()

# Use the parsed arguments
main_directory = args.input_directory
engine = args.engine

# Initialize an empty DataFrame to hold all the merged data
merged_df = pd.DataFrame()


# Loop through each subdirectory in the main directory
for subdir, _, files in os.walk(main_directory):
    if 'segments.csv' in files:
        # Construct the full path to the CSV file
        file_path = os.path.join(subdir, 'segments.csv')
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Rename the 'filename' column to 'audio'
        df = df.rename(columns={'filename': 'audio'})
        
        # Drop the 'duration' column
        df = df.drop(columns=['duration'])
        
        # Append the DataFrame to the merged DataFrame
        merged_df = pd.concat([merged_df, df], ignore_index=True)

merged_df['audio'] = merged_df['audio'].apply(windows_to_wsl_path)

train_df, test_df = train_test_split(merged_df, test_size=0.10, random_state=42)


# Save the merged DataFrame to a new CSV file
merged_df.to_csv(os.path.join(main_directory, 'merged_segments.csv'), index=False, quoting=csv.QUOTE_ALL)
train_df.to_csv(os.path.join(main_directory, 'merged_segments_train.csv'), index=False, quoting=csv.QUOTE_ALL)
test_df.to_csv(os.path.join(main_directory, 'merged_segments_test.csv'), index=False, quoting=csv.QUOTE_ALL)

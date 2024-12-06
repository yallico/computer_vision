import os
import pandas as pd

def process_csv_files(folder_path):
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if not all_files:
        print("No CSV files found in the folder.")
        return
    
    dataframes = []
    for file in all_files:
        file_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(file_path)
            df["image"] = file
            dataframes.append(df)
        except Exception as e:
            print(f"Error loading file {file}: {e}")
    
    if not dataframes:
        print("No valid CSV files to process.")
        return
    
    combined_df = pd.concat(dataframes, ignore_index=True)

    return combined_df
    
    

folder_path = "data-collection/annotations/"
combined_df = process_csv_files(folder_path)
print("\nDataFrame Statistics:")
print(combined_df.describe(include='all'))
print("done")

import os
import pandas as pd

# Folder containing all extracted feature files
folder_path = r"C:\Users\BHUMIKA\OneDrive\文件\ML_Project\mental_workload_detection\data_processed\features"

all_data = []

# Read all Excel files
for file in os.listdir(folder_path):
    if file.endswith(".xlsx") or file.endswith(".xls"):
        file_path = os.path.join(folder_path, file)

        print(f"Reading: {file}")

        # Read Excel file
        df = pd.read_excel(file_path)

        # Extract Subject ID
        # Example: Subject00_1_features.xlsx -> Subject00
        subject_id = file.split("_")[0]

        # Add Subject column
        df["Subject"] = subject_id

        # Append dataframe
        all_data.append(df)

# Combine all files
final_df = pd.concat(all_data, ignore_index=True)

# Save as CSV for ML use
final_df.to_csv("combined_eeg_dataset.csv", index=False)

print("\nCombined dataset created successfully!")
print("Final shape:", final_df.shape)
print("\nColumns:")
print(final_df.columns)
print("\nFirst 5 rows:")
print(final_df.head())
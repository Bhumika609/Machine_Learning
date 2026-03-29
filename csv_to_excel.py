import pandas as pd

# Read CSV file
df = pd.read_csv("combined_eeg_dataset.csv")

# Convert to Excel
df.to_excel("final_data_for_project.xlsx", index=False)
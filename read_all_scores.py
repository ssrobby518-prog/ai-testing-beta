import pandas as pd
import os
import glob

data_dir = 'output/data'
# Find the latest cumulative file
files = glob.glob(os.path.join(data_dir, 'cumulative*.xlsx'))
latest_file = max(files, key=os.path.getmtime)
print(f"Reading from: {latest_file}\n")

df = pd.read_excel(latest_file)

# List of files of interest
files_of_interest = ['a.mp4', 'b.mp4', 'c.mp4', 'd.mp4', 'e.mp4', 'f.mp4', 'g.mp4', 'h.mp4', 'i.mp4', 'j.mp4']

for file in files_of_interest:
    matching_rows = df[df['File Path'].str.contains(file)]
    if not matching_rows.empty:
        row = matching_rows.iloc[-1]  # Latest row for each file
        print(f"{file}: AI Probability = {row['AI Probability']:.2f}")
    else:
        print(f"{file}: No data found")
import glob
import os

import pandas as pd


def main() -> None:
    data_dir = os.path.join('output', 'data')
    files = glob.glob(os.path.join(data_dir, 'cumulative*.xlsx'))
    if not files:
        raise FileNotFoundError(f"No cumulative*.xlsx found in {data_dir}")

    latest_file = max(files, key=os.path.getmtime)
    print(f"Reading from: {latest_file}")

    df = pd.read_excel(latest_file)

    files_of_interest = ['d.mp4', 'e.mp4', 'f.mp4']
    for filename in files_of_interest:
        matched = df[df['File Path'].astype(str).str.contains(filename, na=False)]
        if matched.empty:
            print(f"{filename}: no rows")
            continue
        row = matched.iloc[-1]
        print(f"{filename}: AI Probability = {float(row['AI Probability']):.2f}")


if __name__ == '__main__':
    main()

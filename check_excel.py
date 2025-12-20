import pandas as pd
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_excel('tiktok_labeler/tiktok tinder videos/data/excel_a_labels_raw.xlsx')

print(f'Excel A 共有 {len(df)} 條記錄:\n')
for i, row in df.iterrows():
    print(f'{i+1}. [{row["判定結果"]}] {row["影片網址"]}')

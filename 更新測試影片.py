#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新 Excel A 使用可下載的測試影片
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

EXCEL_PATH = Path(r"C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok tinder videos\data\excel_a_labels_raw.xlsx")

# 使用已驗證可下載的視頻
real_urls = [
    "https://www.tiktok.com/@bellapoarch/video/6862153058223197445",  # 已測試成功
    "https://www.tiktok.com/@zachking/video/6768504823336815877",     # Zach King (通常可訪問)
    "https://www.tiktok.com/@charlidamelio/video/6897974718560423173", # Charli D'Amelio
]

print("=" * 80)
print("更新 Excel A - 使用可下載的測試影片")
print("=" * 80)
print()

data = []
for i, url in enumerate(real_urls, 1):
    video_id = url.split('/')[-1]
    author = url.split('@')[1].split('/')[0]
    data.append({
        '序號': i,
        '影片網址': url,
        '判定結果': 'REAL',
        '標註時間': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        '視頻ID': video_id,
        '作者': author,
        '標題': '',
        '點贊數': 0,
        '來源': 'auto',
        '版本': '1.0.0'
    })

df = pd.DataFrame(data)
EXCEL_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_excel(EXCEL_PATH, index=False)

print(f"✅ 已更新 Excel A：{len(real_urls)} 個視頻")
for i, (url, author) in enumerate(zip(real_urls, [d['作者'] for d in data]), 1):
    print(f"   {i}. @{author}")
print()
print("=" * 80)

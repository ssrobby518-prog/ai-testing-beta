#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新 Excel A 使用經過驗證可下載的影片
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

EXCEL_PATH = Path(r"C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok tinder videos\data\excel_a_labels_raw.xlsx")

# 使用多個經過驗證可下載的視頻（較小的創作者，成功率更高）
test_urls = [
    ("https://www.tiktok.com/@bellapoarch/video/6862153058223197445", "REAL", "bellapoarch"),  # 已驗證成功
    ("https://www.tiktok.com/@zachking/video/6768504823336815877", "REAL", "zachking"),       # 已驗證成功
    ("https://www.tiktok.com/@mrbeast/video/7145811890956569899", "REAL", "mrbeast"),
    ("https://www.tiktok.com/@gordonramsayofficial/video/7285043558775836971", "REAL", "gordonramsay"),
    ("https://www.tiktok.com/@therock/video/7283845742701112619", "REAL", "therock"),
]

print("=" * 80)
print("更新 Excel A - 使用可下載的測試影片")
print("=" * 80)
print()

data = []
for i, (url, label, author) in enumerate(test_urls, 1):
    video_id = url.split('/')[-1]
    data.append({
        '序號': i,
        '影片網址': url,
        '判定結果': label,
        '下載狀態': '未下載',
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

print(f"✅ 已更新 Excel A：{len(test_urls)} 個視頻")
for i, (url, label, author) in enumerate(test_urls, 1):
    status = "✅ 已驗證" if author in ["bellapoarch", "zachking"] else "🆕 新增"
    print(f"   {i}. {status} @{author} ({label})")
print()
print("=" * 80)
print("執行下載：python tiktok_labeler/downloader/run_with_cookies.py")

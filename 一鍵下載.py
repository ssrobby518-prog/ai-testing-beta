#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一鍵下載 - 獲取真實 TikTok URL 並下載
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
import subprocess
import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 設置環境變量
os.environ['YTDLP_COOKIES_FROM_BROWSER'] = 'chrome'

EXCEL_PATH = Path(r"C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok tinder videos\data\excel_a_labels_raw.xlsx")

# 真實的熱門 TikTok URL（公開視頻）
real_urls = [
    "https://www.tiktok.com/@bellapoarch/video/6862153058223197445",  # Bella Poarch 最熱門視頻
    "https://www.tiktok.com/@khaby.lame/video/7124466626970348806",   # Khaby Lame
]

print("準備下載...")

# 更新 Excel A
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

print(f"已更新 Excel A：{len(real_urls)} 個真實 URL")
print("\n開始下載...\n")

# 執行下載
subprocess.run([
    sys.executable,
    "tiktok_labeler/downloader/run_with_cookies.py"
])

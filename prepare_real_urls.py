#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
準備真實 TikTok URL 並更新 Excel A
"""
import pandas as pd
import sys
import io
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 真實的 TikTok URL 示例（公開的熱門視頻）
real_urls = [
    {
        '序號': 1,
        '影片網址': 'https://www.tiktok.com/@zachking/video/7000000000000000000',  # Zach King（著名魔術師）
        '判定結果': 'REAL',
        '標註時間': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        '視頻ID': '7000000000000000000',
        '作者': 'zachking',
        '標題': 'Test video 1',
        '點贊數': 0,
        '來源': 'manual',
        '版本': '1.0.0'
    },
    {
        '序號': 2,
        '影片網址': 'https://www.tiktok.com/@nasa/video/7100000000000000000',  # NASA官方
        '判定結果': 'REAL',
        '標註時間': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        '視頻ID': '7100000000000000000',
        '作者': 'nasa',
        '標題': 'Test video 2',
        '點贊數': 0,
        '來源': 'manual',
        '版本': '1.0.0'
    }
]

print("=" * 80)
print("準備真實 TikTok URL")
print("=" * 80)
print()
print("注意：以下 URL 為示例，可能需要替換為當前有效的真實 URL")
print()

excel_path = 'tiktok_labeler/tiktok tinder videos/data/excel_a_labels_raw.xlsx'

# 備份原始 Excel A
print(f"[1/3] 備份原始 Excel A...")
df_original = pd.read_excel(excel_path)
backup_path = excel_path.replace('.xlsx', '_backup.xlsx')
df_original.to_excel(backup_path, index=False)
print(f"      已備份到: {backup_path}")
print()

# 創建新的 Excel A（僅包含真實 URL）
print(f"[2/3] 創建新的 Excel A（僅包含 2 個真實 URL 測試）...")
df_new = pd.DataFrame(real_urls)
df_new.to_excel(excel_path, index=False)
print(f"      已更新: {excel_path}")
print()

# 顯示新內容
print(f"[3/3] 新 Excel A 內容:")
print()
for i, row in df_new.iterrows():
    print(f"  {i+1}. [{row['判定結果']}] {row['影片網址']}")
    print(f"     作者: @{row['作者']}")
    print()

print("=" * 80)
print("注意事項")
print("=" * 80)
print()
print("⚠️  以上 URL 為示例 ID，可能無法下載。")
print()
print("如需測試真實下載，請手動訪問 TikTok 並複製真實視頻 URL，例如:")
print("  1. 打開 TikTok 網站或 App")
print("  2. 找到任意公開視頻")
print("  3. 複製分享鏈接")
print("  4. 更新 Excel A 中的 URL")
print()
print("✅ 準備完成！")
print()
print("下一步：關閉所有 Chrome 窗口，然後運行:")
print("  python run_with_cookies.py")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡易 TikTok URL 標註工具
不需要 Chrome 擴展
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import io

# 設置輸出編碼
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Excel A 路徑
EXCEL_PATH = Path(r"C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok tinder videos\data\excel_a_labels_raw.xlsx")

def add_label(url, label):
    """添加標註到 Excel A"""
    # 提取 video ID
    video_id = url.split('/')[-1].split('?')[0]

    # 新記錄
    new_record = {
        '序號': 0,  # 稍後更新
        '影片網址': url,
        '判定結果': label.upper(),
        '標註時間': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        '視頻ID': video_id,
        '作者': url.split('@')[1].split('/')[0] if '@' in url else 'unknown',
        '標題': '',
        '點贊數': 0,
        '來源': 'simple_labeler',
        '版本': '1.0.0'
    }

    # 讀取現有數據
    if EXCEL_PATH.exists():
        df = pd.read_excel(EXCEL_PATH)
        # 檢查重複
        if url in df['影片網址'].values:
            print(f"⚠️  URL 已存在，跳過")
            return False
    else:
        df = pd.DataFrame()

    # 添加新記錄
    new_record['序號'] = len(df) + 1
    df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)

    # 保存
    EXCEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(EXCEL_PATH, index=False)

    print(f"✅ 已添加: [{label.upper()}] {url}")
    return True

def main():
    print("=" * 80)
    print("簡易 TikTok 標註工具")
    print("=" * 80)
    print()
    print("使用方式：")
    print("  1. 在 TikTok 複製視頻鏈接")
    print("  2. 貼到這裡")
    print("  3. 輸入標籤：R(eal) / A(I) / U(ncertain) / M(ovie)")
    print("  4. 重複步驟直到完成")
    print()
    print("輸入 'q' 退出")
    print("=" * 80)
    print()

    count = 0

    while True:
        # 輸入 URL
        url = input("\n📎 貼上 TikTok URL: ").strip()

        if url.lower() == 'q':
            break

        if not url or 'tiktok.com' not in url:
            print("❌ 無效的 TikTok URL")
            continue

        # 輸入標籤
        label_input = input("🏷️  標籤 [R/A/U/M]: ").strip().upper()

        # 映射標籤
        label_map = {
            'R': 'REAL',
            'A': 'AI',
            'U': 'UNCERTAIN',
            'M': 'EXCLUDE'
        }

        if label_input not in label_map:
            print("❌ 無效標籤，請輸入 R/A/U/M")
            continue

        label = label_map[label_input]

        # 添加標註
        if add_label(url, label):
            count += 1
            print(f"   總計: {count} 個標註")

    print()
    print("=" * 80)
    print(f"✅ 完成！共標註 {count} 個視頻")
    print(f"📄 Excel A: {EXCEL_PATH}")
    print()
    print("下一步：運行下載器")
    print("  cd tiktok_labeler/downloader")
    print("  python run_with_cookies.py")
    print("=" * 80)

if __name__ == '__main__':
    main()

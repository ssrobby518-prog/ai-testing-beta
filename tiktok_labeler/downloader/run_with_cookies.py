#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
運行 TikTok 下載器（使用自定義 User-Agent）
"""
import os
import sys
import io

# 設置輸出編碼
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("TikTok 下載器")
print("=" * 80)
print()
print("[→] 開始下載...")
print()

# 導入並運行下載器
from tiktok_downloader_classified import TikTokDownloaderClassified

downloader = TikTokDownloaderClassified(max_workers=1)  # 單線程避免IP封鎖

# 執行下載
stats = downloader.download_from_excel_a()

print()
print("=" * 80)
print("下載完成")
print("=" * 80)
print(f"成功: {stats.get('success', 0)} 個")
print(f"失敗: {stats.get('failed', 0)} 個")
if stats.get('by_category'):
    print(f"\n分類統計:")
    for category, count in stats['by_category'].items():
        print(f"  • {category}: {count}")

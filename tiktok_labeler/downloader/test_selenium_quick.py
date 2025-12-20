#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick Selenium test for debugging"""
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from selenium_downloader import SeleniumDownloader

print("=" * 80)
print("Selenium快速測試 - 2024新視頻")
print("=" * 80)
print()

# 測試2024新視頻
test_url = "https://www.tiktok.com/@mrbeast/video/7145811890956569899"
output = Path("test_new.mp4")

print(f"測試視頻: {test_url}")
print()

with SeleniumDownloader(headless=True) as downloader:
    success, error = downloader.download(test_url, output)

    print()
    if success and output.exists():
        print(f"✅ 測試成功: {output.stat().st_size / (1024*1024):.2f} MB")
        output.unlink()
    else:
        print(f"❌ 測試失敗: {error}")

print()
print("=" * 80)

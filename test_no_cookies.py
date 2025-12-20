#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試不使用 Cookies 下載 TikTok 視頻
"""
import subprocess
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("測試不使用 Cookies 下載")
print("=" * 80)
print()

# 測試視頻
test_url = "https://www.tiktok.com/@bellapoarch/video/6862153058223197445"
output_file = Path(__file__).parent / "test_output.mp4"

print(f"測試 URL: {test_url}")
print()

# 方法1: 無 cookies，只用 user-agent
print("方法1: 使用自定義 User-Agent")
cmd1 = [
    sys.executable, "-m", "yt_dlp",
    "-o", str(output_file),
    "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "--sleep-requests", "2",
    test_url
]

result1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=120)

if result1.returncode == 0 and output_file.exists():
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"✅ 成功! 文件大小: {size_mb:.2f} MB")
    output_file.unlink()  # 刪除測試文件
else:
    print(f"❌ 失敗")
    print(f"錯誤: {result1.stderr[:300]}")

print()
print("=" * 80)

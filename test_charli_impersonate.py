#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試 charlidamelio 視頻下載（使用瀏覽器模擬）
"""
import subprocess
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("測試 charlidamelio 視頻下載（瀏覽器模擬 - Chrome 120）")
print("=" * 80)
print()

test_url = "https://www.tiktok.com/@charlidamelio/video/6897974718560423173"
output_file = Path(__file__).parent / "test_charli_imp.mp4"

print(f"測試 URL: {test_url}")
print()
print("使用 --impersonate chrome120...")
print()

cmd = [
    sys.executable, "-m", "yt_dlp",
    "-o", str(output_file),
    "--impersonate", "chrome120",  # 關鍵！模擬 Chrome 120
    "--sleep-requests", "3",
    "--sleep-interval", "2",
    "--max-sleep-interval", "5",
    "--retries", "5",
    "--fragment-retries", "5",
    test_url
]

result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

if result.returncode == 0 and output_file.exists():
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"✅ 成功! 文件大小: {size_mb:.2f} MB")
    print(f"\n使用 --impersonate 成功繞過IP封鎖！")
    output_file.unlink()  # 刪除測試文件
else:
    print(f"❌ 失敗")
    print(f"Return code: {result.returncode}")
    print(f"\nStderr:\n{result.stderr[:500]}")

print()
print("=" * 80)

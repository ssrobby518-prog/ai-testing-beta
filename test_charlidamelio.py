#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試 charlidamelio 視頻下載（完整瀏覽器模擬）
"""
import subprocess
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("測試 charlidamelio 視頻下載（完整瀏覽器模擬）")
print("=" * 80)
print()

test_url = "https://www.tiktok.com/@charlidamelio/video/6897974718560423173"
output_file = Path(__file__).parent / "test_charli.mp4"

print(f"測試 URL: {test_url}")
print()
print("使用完整瀏覽器headers模擬...")
print()

cmd = [
    sys.executable, "-m", "yt_dlp",
    "-o", str(output_file),
    "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "--add-header", "Accept:text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "--add-header", "Accept-Language:en-US,en;q=0.9",
    "--add-header", "Accept-Encoding:gzip, deflate, br",
    "--add-header", "DNT:1",
    "--add-header", "Connection:keep-alive",
    "--add-header", "Upgrade-Insecure-Requests:1",
    "--add-header", "Sec-Fetch-Dest:document",
    "--add-header", "Sec-Fetch-Mode:navigate",
    "--add-header", "Sec-Fetch-Site:none",
    "--add-header", "Sec-Fetch-User:?1",
    '--add-header', 'sec-ch-ua:"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    "--add-header", "sec-ch-ua-mobile:?0",
    '--add-header', 'sec-ch-ua-platform:"Windows"',
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
    output_file.unlink()  # 刪除測試文件
else:
    print(f"❌ 失敗")
    print(f"Return code: {result.returncode}")
    print(f"\nStdout:\n{result.stdout}")
    print(f"\nStderr:\n{result.stderr}")

print()
print("=" * 80)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試下載不同的TikTok視頻
"""
import subprocess
import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 設置環境變量
os.environ['YTDLP_COOKIES_FROM_BROWSER'] = 'chrome'

# 測試不同的熱門視頻
test_urls = [
    ("zach.king", "https://www.tiktok.com/@zachking/video/6768504823336815877"),
    ("mrbeast", "https://www.tiktok.com/@mrbeast/video/7234304298139782446"),
    ("charlidamelio", "https://www.tiktok.com/@charlidamelio/video/6897974718560423173"),
]

print("=" * 80)
print("測試下載不同視頻以找出IP封鎖模式")
print("=" * 80)
print()

for name, url in test_urls:
    print(f"測試: {name}")
    print(f"URL: {url}")
    print()

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--cookies-from-browser", "chrome",
        "--sleep-requests", "3",
        "--print", "title",
        "--skip-download",
        url
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    if result.returncode == 0:
        print(f"✅ {name}: 可以訪問")
        print(f"   標題: {result.stdout.strip()}")
    else:
        error = result.stderr.lower()
        if "ip" in error and "block" in error:
            print(f"🚫 {name}: IP被封鎖")
        elif "private" in error or "unavailable" in error:
            print(f"🔒 {name}: 私密或不可用")
        else:
            print(f"❌ {name}: 其他錯誤")
            print(f"   錯誤: {result.stderr[:200]}")

    print()
    print("-" * 80)
    print()

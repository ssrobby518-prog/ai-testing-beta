#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
強制下載新TikTok影片 - 多種方法嘗試
"""
import subprocess
import sys
import io
from pathlib import Path
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("強制下載新TikTok影片測試")
print("=" * 80)
print()

# 測試最新的mrbeast視頻
test_url = "https://www.tiktok.com/@mrbeast/video/7145811890956569899"
output_file = Path(__file__).parent / "test_force.mp4"

methods = [
    {
        "name": "方法1: 強制使用API提取",
        "args": ["--extractor-args", "tiktok:api_hostname=api16-normal-c-useast1a.tiktokv.com"]
    },
    {
        "name": "方法2: 使用移動版網址",
        "url": "https://m.tiktok.com/v/7145811890956569899.html",
        "args": []
    },
    {
        "name": "方法3: 強制使用不同User-Agent",
        "args": ["--user-agent", "TikTok 26.1.3 rv:261030 (iPhone; iOS 14.4.2; en_US) Cronet"]
    },
    {
        "name": "方法4: 使用yt-dlp最激進設置",
        "args": [
            "--extractor-retries", "20",
            "--socket-timeout", "60",
            "--force-ipv4"
        ]
    }
]

for i, method in enumerate(methods, 1):
    print(f"\n{'='*60}")
    print(f"嘗試 {method['name']}")
    print(f"{'='*60}")

    url = method.get('url', test_url)
    extra_args = method.get('args', [])

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-o", str(output_file),
        "--no-warnings",
        *extra_args,
        url
    ]

    print(f"執行: {' '.join(cmd[:5])}...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0 and output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"✅✅✅ 成功！文件大小: {size_mb:.2f} MB")
            print(f"\n🎉 找到有效方法: {method['name']}")
            output_file.unlink()
            sys.exit(0)
        else:
            error = result.stderr[:200] if result.stderr else "未知錯誤"
            print(f"❌ 失敗: {error}")
    except subprocess.TimeoutExpired:
        print(f"❌ 超時")
    except Exception as e:
        print(f"❌ 異常: {e}")

    time.sleep(2)

print(f"\n{'='*80}")
print("所有方法都失敗。TikTok新影片需要更高級的繞過技術。")
print("建議：")
print("1. 手動下載影片後放入文件夾")
print("2. 或使用第三方TikTok下載服務")
print("=" * 80)

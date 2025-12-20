#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 browser_cookie3 提取 Chrome Cookies 並保存為 Netscape 格式
"""
import browser_cookie3
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("提取 Chrome Cookies")
print("=" * 80)
print()

# 提取 TikTok cookies
cookies_file = Path(__file__).parent / "tiktok_cookies.txt"

try:
    print("正在提取 Chrome cookies...")
    cj = browser_cookie3.chrome(domain_name='tiktok.com')

    # 保存為 Netscape 格式
    with open(cookies_file, 'w', encoding='utf-8') as f:
        f.write("# Netscape HTTP Cookie File\n")
        for cookie in cj:
            if 'tiktok.com' in cookie.domain:
                # Netscape 格式: domain, flag, path, secure, expiration, name, value
                f.write(f"{cookie.domain}\tTRUE\t{cookie.path}\t")
                f.write(f"{'TRUE' if cookie.secure else 'FALSE'}\t")
                f.write(f"{cookie.expires if cookie.expires else 0}\t")
                f.write(f"{cookie.name}\t{cookie.value}\n")

    print(f"✅ Cookies 已保存到: {cookies_file}")
    print(f"   共 {len(list(cj))} 個 cookies")

except Exception as e:
    print(f"❌ 錯誤: {e}")
    print()
    print("解決方案:")
    print("1. 關閉所有 Chrome 瀏覽器窗口")
    print("2. 重新運行此腳本")
    sys.exit(1)

print()
print("=" * 80)

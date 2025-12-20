#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
強制從Chrome提取TikTok cookies（需要先登入TikTok）
"""
import sys
import io
import os
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("強制提取 Chrome TikTok Cookies")
print("=" * 80)
print()

cookies_file = Path(__file__).parent / "tiktok_cookies.txt"

print("⚠️  重要：執行前請確保：")
print("1. 已在Chrome登入 TikTok")
print("2. 關閉所有Chrome視窗")
print()
input("按Enter繼續...")
print()

try:
    import browser_cookie3

    print("正在提取cookies...")
    cj = browser_cookie3.chrome(domain_name='tiktok.com')

    # 保存為Netscape格式
    with open(cookies_file, 'w', encoding='utf-8') as f:
        f.write("# Netscape HTTP Cookie File\n")
        cookie_count = 0
        for cookie in cj:
            if 'tiktok.com' in cookie.domain:
                f.write(f"{cookie.domain}\tTRUE\t{cookie.path}\t")
                f.write(f"{'TRUE' if cookie.secure else 'FALSE'}\t")
                f.write(f"{cookie.expires if cookie.expires else 0}\t")
                f.write(f"{cookie.name}\t{cookie.value}\n")
                cookie_count += 1

    print(f"✅ 成功提取 {cookie_count} 個cookies")
    print(f"✅ 已保存到: {cookies_file}")
    print()
    print("現在可以使用cookies下載視頻了")

except Exception as e:
    print(f"❌ 錯誤: {e}")
    print()
    print("解決方案:")
    print("1. 確保已關閉所有Chrome視窗")
    print("2. 以管理員身份運行此腳本")
    print("3. 或手動導出cookies:")
    print("   - Chrome擴展: Get cookies.txt LOCALLY")
    print("   - 訪問 tiktok.com")
    print("   - 導出cookies到 tiktok_cookies.txt")

print()
print("=" * 80)

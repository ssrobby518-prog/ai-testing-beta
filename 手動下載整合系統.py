#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手動下載整合系統 - 完美整合手動下載的影片到TSAR-RAPTOR
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil
import re
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

EXCEL_PATH = Path(r"C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok tinder videos\data\excel_a_labels_raw.xlsx")
MANUAL_DIR = Path(r"C:\Users\s_robby518\Documents\trae_projects\ai testing\手動下載")
VIDEO_FOLDERS = {
    'real': Path(r"C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok tinder videos\real"),
    'ai': Path(r"C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok tinder videos\ai"),
}

print("=" * 80)
print("手動下載整合系統 - TSAR-RAPTOR 手動下載流程")
print("=" * 80)
print()
print("📱 使用方式：")
print()
print("1. 在手機/電腦上下載 TikTok 視頻")
print("   - Android: 使用 SnapTik (snaptik.app)")
print("   - iOS: 使用 TikMate (tikmate.app)")
print("   - 電腦: 使用 SaveTT (savett.cc)")
print()
print("2. 將下載的視頻放入指定文件夾：")
print(f"   真實視頻 → {MANUAL_DIR}/real/")
print(f"   AI視頻  → {MANUAL_DIR}/ai/")
print()
print("3. 視頻命名格式（任意，系統會自動處理）：")
print("   - video_123456.mp4")
print("   - @username_video.mp4")
print("   - 或保持原名稱")
print()
print("4. 運行此腳本，自動整合到系統")
print()
print("=" * 80)
print()

# 創建手動下載文件夾
MANUAL_DIR.mkdir(exist_ok=True)
(MANUAL_DIR / "real").mkdir(exist_ok=True)
(MANUAL_DIR / "ai").mkdir(exist_ok=True)

# 檢查手動下載文件夾
real_videos = list((MANUAL_DIR / "real").glob("*.mp4"))
ai_videos = list((MANUAL_DIR / "ai").glob("*.mp4"))

if not real_videos and not ai_videos:
    print("⚠️  未找到手動下載的視頻")
    print()
    print(f"請將視頻放入：")
    print(f"  - 真實視頻 → {MANUAL_DIR / 'real'}")
    print(f"  - AI視頻 → {MANUAL_DIR / 'ai'}")
    print()
    print("然後重新運行此腳本")
    sys.exit(0)

print(f"✅ 找到 {len(real_videos)} 個 REAL 視頻")
print(f"✅ 找到 {len(ai_videos)} 個 AI 視頻")
print()

# 讀取 Excel A
df = pd.read_excel(EXCEL_PATH) if EXCEL_PATH.exists() else pd.DataFrame()
next_id = len(df) + 1

new_records = []
processed_count = 0

# 處理視頻
for label, videos in [('REAL', real_videos), ('AI', ai_videos)]:
    for video_file in videos:
        # 生成唯一ID
        video_id = f"manual_{int(datetime.now().timestamp())}_{processed_count}"

        # 重命名並移動到對應文件夾
        new_filename = f"{label.lower()}_{video_id}.mp4"
        dest_folder = VIDEO_FOLDERS[label.lower()]
        dest_folder.mkdir(parents=True, exist_ok=True)
        dest_path = dest_folder / new_filename

        # 移動文件
        shutil.move(str(video_file), str(dest_path))

        # 添加到 Excel A
        new_records.append({
            '序號': next_id + processed_count,
            '影片網址': f'manual://{video_file.name}',
            '判定結果': label,
            '下載狀態': '已下載',
            '標註時間': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            '視頻ID': video_id,
            '作者': 'manual',
            '標題': video_file.stem,
            '點贊數': 0,
            '來源': 'manual',
            '版本': '1.0.0'
        })

        processed_count += 1
        print(f"✅ 處理: {video_file.name} → {new_filename}")

# 更新 Excel A
if new_records:
    df_new = pd.concat([df, pd.DataFrame(new_records)], ignore_index=True)
    df_new.to_excel(EXCEL_PATH, index=False)

    print()
    print("=" * 80)
    print(f"✅ 成功整合 {processed_count} 個視頻到系統")
    print(f"✅ Excel A 已更新")
    print()
    print("下一步：運行流水線進行特徵提取與分析")
    print("python 執行第一層下載.py")
    print("=" * 80)
else:
    print("⚠️  沒有需要處理的視頻")

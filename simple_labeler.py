#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
超簡單標註工具 - 一個視窗搞定
"""
import subprocess
import pandas as pd
from pathlib import Path
import shutil
import time

video_folder = Path(r"C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok videos download\tiktok videos download")
excel_output = Path(r"C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok videos download\data\human_labels_all.xlsx")

base_dir = Path(r"C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok videos download")
folders = {
    'REAL': base_dir / 'real',
    'AI': base_dir / 'ai',
    'NOT_SURE': base_dir / 'not sure',
    'MOVIE': base_dir / '電影動畫'
}

# Create folders
for f in folders.values():
    f.mkdir(parents=True, exist_ok=True)

# Get videos
videos = list(video_folder.glob("*.mp4"))
labels = []

# Load existing
if excel_output.exists():
    df = pd.read_excel(excel_output)
    labels = df.to_dict('records')
    labeled = set(str(r['Video_ID']) for r in labels)
    videos = [v for v in videos if v.stem not in labeled]

print(f"\n{'='*70}")
print(f"影片數量: {len(videos)}")
print(f"{'='*70}")
print("\n每個影片播放後，直接按鍵標註：")
print("  A = REAL (真實)")
print("  D = AI (AI生成)")
print("  W = NOT_SURE (不確定)")
print("  S = MOVIE (電影/動畫)")
print("  Q = 退出")
print(f"{'='*70}\n")

for i, video in enumerate(videos, 1):
    print(f"\n[{i}/{len(videos)}] {video.name}")

    # Open with default Windows player (has sound)
    subprocess.Popen(['start', '', str(video)], shell=True)

    print("影片已打開，看完後在這裡輸入...")

    # Get keyboard input
    print("\n標註 (A=REAL / D=AI / W=NOT_SURE / S=MOVIE / Q=退出): ", end='', flush=True)
    label_key = input().strip().lower()

    if label_key == 'q':
        print("\n退出")
        break
    elif label_key == 'a':
        label = 'REAL'
    elif label_key == 'd':
        label = 'AI'
    elif label_key == 'w':
        label = 'NOT_SURE'
    elif label_key == 's':
        label = 'MOVIE'
    else:
        print("無效按鍵，跳過")
        continue

    print(f"✓ 標註為: {label}")

    # Save
    labels.append({
        'Video_ID': video.stem,
        'Filename': video.name,
        'Label': label,
        'Timestamp': pd.Timestamp.now()
    })

    # Copy to folder
    dest = folders[label] / video.name
    shutil.copy2(video, dest)
    print(f"✓ 已分類到: {folders[label].name}/")

    # Auto-save
    if len(labels) % 5 == 0:
        df = pd.DataFrame(labels)
        excel_output.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(excel_output, index=False)
        print(f"[自動保存: {len(labels)} 個標註]")

# Final save
if labels:
    df = pd.DataFrame(labels)
    excel_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(excel_output, index=False)
    print(f"\n完成！共標註 {len(labels)} 個影片")

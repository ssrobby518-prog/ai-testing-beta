#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Automated Pipeline: Wait for detection -> Generate Excel -> Classify videos
"""
import time
import json
import os
import shutil
import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok videos download")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = Path("output")
INPUT_DIR = Path("input")

# Expected video count
total_videos = len([f for f in INPUT_DIR.glob("*.mp4")])
print(f"Total videos to process: {total_videos}")

# Wait for all videos to be processed
print("Waiting for AI detection to complete...")
while True:
    processed_count = len([f for f in OUTPUT_DIR.glob("diagnostic_*.json")
                          if not f.name.startswith("diagnostic_75")
                          and not f.name.startswith("diagnostic_Download")])

    print(f"  Progress: {processed_count}/{total_videos}", end='\r')

    if processed_count >= total_videos - 5:  # Allow some tolerance
        print(f"\n{processed_count} videos processed. Generating report...")
        break

    time.sleep(30)

# Generate Excel and classify
results = []
FOLDERS = {
    'SAFE': BASE_DIR / "real",
    'GRAY_ZONE': BASE_DIR / "not sure",
    'KILL_ZONE': BASE_DIR / "ai",
}

for file in OUTPUT_DIR.glob("diagnostic_*.json"):
    if file.name.startswith("diagnostic_75") or file.name.startswith("diagnostic_Download"):
        continue

    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        video_id = file.name.replace('diagnostic_', '').replace('_mp4.json', '')
        ai_p = data.get('global_probability', 0)

        classification = 'SAFE' if ai_p < 20 else ('GRAY_ZONE' if ai_p < 50 else 'KILL_ZONE')

        results.append({
            'Video_ID': video_id,
            'AI_Probability': ai_p,
            'Threat_Level': data.get('threat_level', 'UNKNOWN'),
            'Classification': classification,
            'Is_Phone': data.get('video_characteristics', {}).get('is_phone_video', False),
            'Face_Presence': data.get('video_characteristics', {}).get('face_presence', 0) * 100,
            'Bitrate_Mbps': data.get('video_characteristics', {}).get('bitrate', 0) / 1000000
        })

# Save Excel
df = pd.DataFrame(results)
excel_path = DATA_DIR / "detection_results.xlsx"
df.to_excel(excel_path, index=False)
print(f"\n[OK] Excel saved: {excel_path}")

# Classify videos
for result in results:
    video_file = f"{result['Video_ID']}.mp4"
    source = INPUT_DIR / video_file

    classification = result['Classification']
    dest_folder = FOLDERS[classification]
    dest = dest_folder / video_file

    if source.exists():
        shutil.copy2(source, dest)

safe = sum(1 for r in results if r['Classification']=='SAFE')
gray = sum(1 for r in results if r['Classification']=='GRAY_ZONE')
kill = sum(1 for r in results if r['Classification']=='KILL_ZONE')

print(f"\n{'='*60}")
print("CLASSIFICATION COMPLETE")
print(f"{'='*60}")
print(f"Total: {len(results)} videos")
print(f"  REAL (SAFE):     {safe} videos -> real/")
print(f"  NOT SURE (GRAY): {gray} videos -> not sure/")
print(f"  AI (KILL):       {kill} videos -> ai/")
print(f"\nExcel: {excel_path}")
print(f"{'='*60}")

# Auto-cleanup: delete processed videos
print(f"\n清理已分類影片...")
download_dir = BASE_DIR / "tiktok videos download"
deleted = 0
for result in results:
    video_file = download_dir / f"{result['Video_ID']}.mp4"
    if video_file.exists():
        try:
            video_file.unlink()
            deleted += 1
        except:
            pass

print(f"✓ 已清理 {deleted} 個影片")
print(f"{'='*60}")

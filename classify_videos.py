#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Classify videos based on AI detection results and generate Excel
"""
import json
import os
import shutil
import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok videos download")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = Path("output")

# Classification folders
FOLDERS = {
    'SAFE': BASE_DIR / "real",
    'GRAY_ZONE': BASE_DIR / "not sure",
    'KILL_ZONE': BASE_DIR / "ai",
    'MOVIE': BASE_DIR / "電影動畫"
}

def main():
    results = []

    # Read all diagnostic JSON files
    for file in os.listdir(OUTPUT_DIR):
        if file.startswith('diagnostic_') and file.endswith('.json'):
            with open(OUTPUT_DIR / file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                video_id = file.replace('diagnostic_', '').replace('_mp4.json', '')
                results.append({
                    'Video_ID': video_id,
                    'AI_Probability': data.get('global_probability', 0),
                    'Threat_Level': data.get('threat_level', 'UNKNOWN'),
                    'Is_Phone': data.get('video_characteristics', {}).get('is_phone_video', False),
                    'Face_Presence': data.get('video_characteristics', {}).get('face_presence', 0),
                    'Bitrate': data.get('video_characteristics', {}).get('bitrate', 0),
                    'Classification': 'SAFE' if data.get('global_probability', 0) < 20 else ('GRAY_ZONE' if data.get('global_probability', 0) < 50 else 'KILL_ZONE')
                })

    # Create Excel
    df = pd.DataFrame(results)
    excel_path = DATA_DIR / "detection_results.xlsx"
    df.to_excel(excel_path, index=False)
    print(f"Excel saved: {excel_path}")

    # Move videos to classified folders
    for result in results:
        video_file = f"{result['Video_ID']}.mp4"
        source = Path("input") / video_file

        classification = result['Classification']
        dest_folder = FOLDERS.get(classification, FOLDERS['GRAY_ZONE'])
        dest = dest_folder / video_file

        if source.exists():
            shutil.copy2(source, dest)
            print(f"[{classification}] {video_file}")

    print(f"\nProcessed {len(results)} videos")
    print(f"SAFE: {sum(1 for r in results if r['Classification']=='SAFE')}")
    print(f"GRAY_ZONE: {sum(1 for r in results if r['Classification']=='GRAY_ZONE')}")
    print(f"KILL_ZONE: {sum(1 for r in results if r['Classification']=='KILL_ZONE')}")

    # Auto-cleanup: delete processed videos from download folder
    print(f"\n清理已分類影片...")
    download_dir = BASE_DIR / "tiktok videos download"
    deleted = 0
    for result in results:
        video_file = download_dir / f"{result['Video_ID']}.mp4"
        if video_file.exists():
            try:
                video_file.unlink()
                deleted += 1
            except Exception as e:
                print(f"[ERROR] 刪除失敗 {video_file.name}: {e}")

    print(f"✓ 已清理 {deleted} 個影片")

if __name__ == "__main__":
    main()

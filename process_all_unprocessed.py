#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Process ALL unprocessed videos regardless of filename"""
import os
import subprocess
from pathlib import Path

input_dir = Path("input")
output_dir = Path("output")

# Get all videos in input
all_videos = [f.name for f in input_dir.glob("*.mp4")]

# Get all processed videos
processed = {f.name.replace('diagnostic_', '').replace('_mp4.json', '.mp4')
             for f in output_dir.glob("diagnostic_*.json")}

# Get unprocessed
unprocessed = [v for v in all_videos if v not in processed]

print(f"Total videos in INPUT: {len(all_videos)}")
print(f"Already processed: {len(processed)}")
print(f"To process: {len(unprocessed)}\n")

for i, video in enumerate(unprocessed, 1):
    print(f"[{i}/{len(unprocessed)}] Processing {video}...")

    env = os.environ.copy()
    env['ONLY_FILE'] = video

    try:
        subprocess.run(
            ['python', 'autotesting.py'],
            env=env,
            timeout=300,
            capture_output=True
        )
        print(f"  [OK] {video}")
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {video}")
    except Exception as e:
        print(f"  [ERROR] {video}: {e}")

print(f"\n[DONE] Processed {len(unprocessed)} videos")

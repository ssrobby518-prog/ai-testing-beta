#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Process remaining 7 unprocessed videos
"""
import os
import subprocess

# List of unprocessed videos
unprocessed_videos = [
    '7529197470543269133.mp4',
    '7529599151269596429.mp4',
    '7530022335135386935.mp4',
    '7531121309086911799.mp4',
    '7531384288722341175.mp4',
    '7531387460744465719.mp4',
    '7531568017566715150.mp4'
]

print(f"Processing {len(unprocessed_videos)} remaining videos...\n")

for i, video in enumerate(unprocessed_videos, 1):
    print(f"[{i}/{len(unprocessed_videos)}] Processing {video}...")

    # Set environment variable to process only this video
    env = os.environ.copy()
    env['ONLY_FILE'] = video

    # Run autotesting for this video
    try:
        subprocess.run(
            ['python', 'autotesting.py'],
            env=env,
            timeout=300,  # 5 minutes per video
            capture_output=True
        )
        print(f"    [OK] Completed {video}")
    except subprocess.TimeoutExpired:
        print(f"    [WARN] Timeout for {video}, moving to next")
    except Exception as e:
        print(f"    [ERR] Error processing {video}: {e}")

print("\n[DONE] All remaining videos processed!")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Monitor processing progress in real-time"""
import time
from pathlib import Path

input_dir = Path("input")
output_dir = Path("output")

total_videos = len(list(input_dir.glob("*.mp4")))
print(f"Total videos to process: {total_videos}\n")

last_count = 0
start_time = time.time()

while True:
    processed = len(list(output_dir.glob("diagnostic_*.json")))
    remaining = total_videos - processed
    pct = (processed / total_videos * 100) if total_videos > 0 else 0

    # Calculate ETA
    if processed > last_count:
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        eta_sec = remaining / rate if rate > 0 else 0
        eta_min = eta_sec / 60
        eta_str = f"ETA: {eta_min:.1f} min" if eta_min > 0 else "ETA: calculating..."
    else:
        eta_str = "ETA: calculating..."

    print(f"\rProgress: {processed}/{total_videos} ({pct:.1f}%) | Remaining: {remaining} | {eta_str}", end='', flush=True)

    if processed >= total_videos:
        print("\n\n[DONE] All videos processed!")
        break

    last_count = processed
    time.sleep(10)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wait for processing to complete, then run analysis automatically
"""
import time
import subprocess
from pathlib import Path

input_dir = Path("input")
output_dir = Path("output")

total_videos = len(list(input_dir.glob("*.mp4")))
print(f"Waiting for {total_videos} videos to be processed...\n")

# Wait for processing to complete
while True:
    processed = len(list(output_dir.glob("diagnostic_*.json")))
    remaining = total_videos - processed
    pct = (processed / total_videos * 100) if total_videos > 0 else 0

    print(f"\rProgress: {processed}/{total_videos} ({pct:.1f}%) | Remaining: {remaining}", end='', flush=True)

    if remaining <= 0:
        print("\n\n[OK] All videos processed!")
        break

    time.sleep(30)  # Check every 30 seconds

# Run analysis
print("\n" + "="*80)
print("RUNNING AUTOMATED ANALYSIS")
print("="*80)

print("\n[1/2] Generating classification and Excel...")
result = subprocess.run(['python', 'classify_videos.py'], capture_output=True, text=True)
print(result.stdout)

print("\n[2/2] Comparing human vs AI labels...")
result = subprocess.run(['python', 'compare_human_vs_ai.py'], capture_output=True, text=True)
print(result.stdout)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

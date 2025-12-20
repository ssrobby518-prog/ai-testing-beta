#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full Pipeline: Process → Classify → Compare
"""
import subprocess
import sys
from pathlib import Path

print("="*80)
print("FULL ANALYSIS PIPELINE")
print("="*80)

# Step 1: Process all unprocessed videos
print("\n[Step 1/4] Processing unprocessed videos...")
try:
    result = subprocess.run(['python', 'process_all_unprocessed.py'],
                          capture_output=True, text=True, timeout=600)
    print(result.stdout)
    if result.returncode != 0:
        print(f"[ERROR] Processing failed: {result.stderr}")
except Exception as e:
    print(f"[ERROR] {e}")

# Step 2: Generate detection results Excel
print("\n[Step 2/4] Generating detection results...")
try:
    result = subprocess.run(['python', 'classify_videos.py'],
                          capture_output=True, text=True)
    print(result.stdout)
except Exception as e:
    print(f"[ERROR] {e}")

# Step 3: Compare human vs AI
print("\n[Step 3/4] Comparing human labels vs AI detection...")
try:
    result = subprocess.run(['python', 'compare_human_vs_ai.py'],
                          capture_output=True, text=True)
    print(result.stdout)
except Exception as e:
    print(f"[ERROR] {e}")

# Step 4: Re-run classification with updated results
print("\n[Step 4/4] Re-classifying videos with latest data...")
try:
    result = subprocess.run(['python', 'classify_videos.py'],
                          capture_output=True, text=True)
    print(result.stdout)
except Exception as e:
    print(f"[ERROR] {e}")

print("\n" + "="*80)
print("PIPELINE COMPLETE")
print("="*80)

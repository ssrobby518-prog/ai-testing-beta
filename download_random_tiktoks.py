#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download 50 random TikTok videos from For You Page / Trending
Using TikTok API or scraping trending videos from different accounts
"""
import subprocess
import random
import time
from pathlib import Path

# Correct directory structure
BASE_DIR = Path(r"C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok videos download")
OUTPUT_DIR = BASE_DIR / "tiktok videos download"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Popular TikTok hashtags for random sampling
HASHTAGS = [
    'fyp', 'foryou', 'viral', 'trending', 'dance', 'comedy',
    'food', 'travel', 'fashion', 'beauty', 'fitness', 'pets',
    'music', 'art', 'diy', 'cooking', 'tutorial', 'funny'
]

# Popular diverse TikTok accounts (mix of real and potentially AI)
DIVERSE_ACCOUNTS = [
    'charlidamelio', 'addisonre', 'zachking', 'bellapoarch',
    'khaby.lame', 'willsmith', 'therock', 'gordonramsayofficial',
    'nasa', 'natgeo', 'wired', 'vogue', 'nba', 'espn'
]

def download_from_hashtag(hashtag, max_videos=5):
    """Download videos from a trending hashtag"""
    print(f"Downloading from #{hashtag}...")
    try:
        cmd = [
            'python', '-m', 'yt_dlp',
            f'https://www.tiktok.com/tag/{hashtag}',
            '--playlist-end', str(max_videos),
            '-o', str(OUTPUT_DIR / '%(id)s.%(ext)s'),
            '--no-warnings', '--quiet'
        ]
        subprocess.run(cmd, timeout=180, capture_output=True)
        return True
    except:
        return False

def download_from_account(account, max_videos=3):
    """Download few videos from an account"""
    print(f"Downloading from @{account}...")
    try:
        cmd = [
            'python', '-m', 'yt_dlp',
            f'https://www.tiktok.com/@{account}',
            '--playlist-end', str(max_videos),
            '-o', str(OUTPUT_DIR / '%(id)s.%(ext)s'),
            '--no-warnings', '--quiet'
        ]
        subprocess.run(cmd, timeout=180, capture_output=True)
        return True
    except:
        return False

def main():
    print("Downloading 50 RANDOM TikTok videos from diverse sources...")
    print(f"Target: {OUTPUT_DIR}\n")

    downloaded = 0
    target = 50

    # Strategy: Mix of hashtags and diverse accounts
    random.shuffle(HASHTAGS)
    random.shuffle(DIVERSE_ACCOUNTS)

    # Download from 10 random hashtags (5 videos each)
    for hashtag in HASHTAGS[:10]:
        if downloaded >= target:
            break
        download_from_hashtag(hashtag, max_videos=5)
        downloaded += len(list(OUTPUT_DIR.glob('*.mp4')))
        print(f"  Progress: {downloaded}/{target}")
        time.sleep(2)

    # Download from diverse accounts if needed
    for account in DIVERSE_ACCOUNTS:
        if downloaded >= target:
            break
        download_from_account(account, max_videos=3)
        downloaded = len(list(OUTPUT_DIR.glob('*.mp4')))
        print(f"  Progress: {downloaded}/{target}")
        time.sleep(2)

    final_count = len(list(OUTPUT_DIR.glob('*.mp4')))
    print(f"\n[DONE] Downloaded {final_count} random TikTok videos")
    print(f"Location: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

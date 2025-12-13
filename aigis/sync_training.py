#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from pathlib import Path

SERVER_DIR = Path(__file__).parent / "TikTok_Labeler_Server"
DATASET_FILE = SERVER_DIR / "dataset.csv"
TRAINING_FILE = SERVER_DIR / "training_data.csv"

TRAINING_HEADER = [
    "video_url","label","timestamp",
    "model_fingerprint","frequency_analysis","sensor_noise","physics_violation",
    "texture_noise","text_fingerprint","metadata_score","heartbeat","blink_dynamics",
    "lighting_geometry","av_sync","semantic_stylometry","bitrate","fps","duration",
    "resolution","author_id","reason"
]


def ensure_training_header():
    if not TRAINING_FILE.exists():
        with open(TRAINING_FILE, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=TRAINING_HEADER)
            writer.writeheader()


def sync():
    ensure_training_header()

    existing_urls = set()
    with open(TRAINING_FILE, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            existing_urls.add(row.get('video_url', ''))

    appended = 0
    with open(DATASET_FILE, 'r', encoding='utf-8-sig') as f_ds, \
         open(TRAINING_FILE, 'a', encoding='utf-8-sig', newline='') as f_tr:
        ds_reader = csv.DictReader(f_ds)
        tr_writer = csv.DictWriter(f_tr, fieldnames=TRAINING_HEADER)
        for ds in ds_reader:
            url = ds.get('video_url', '')
            if url and url not in existing_urls:
                row = {k: '' for k in TRAINING_HEADER}
                row.update({
                    'video_url': url,
                    'label': ds.get('label', ''),
                    'timestamp': ds.get('timestamp', ''),
                    'author_id': ds.get('author_id', ''),
                    'reason': ds.get('reason', '')
                })
                tr_writer.writerow(row)
                appended += 1

    print(f"✓ Synced training_data.csv with {appended} new rows")


if __name__ == '__main__':
    sync()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aigis Pipeline - 夜間自動化ETL + 主動學習
第一性原理：睡覺時訓練，醒來時收穫
"""

import os
import sys
import subprocess
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# === 配置 ===
BASE_DIR = Path(__file__).parent
DATASET_FILE = BASE_DIR.parent / "TikTok_Labeler_Server" / "dataset.csv"
DOWNLOAD_DIR = BASE_DIR / "downloaded_videos"
FEATURES_FILE = BASE_DIR / "features_matrix.csv"
MODEL_FILE = BASE_DIR / "model_latest.json"

# 藍隊模組路徑
BLUE_TEAM_DIR = BASE_DIR.parent.parent / "modules"

def phase_1_download():
    """
    Phase 1: 下載視頻

    沙皇炸彈：增量下載（只下載新增）
    """
    logging.info("📥 [Phase 1] 下載視頻中...")

    if not DATASET_FILE.exists():
        logging.error("❌ dataset.csv 不存在，請先標註數據")
        return

    df = pd.read_csv(DATASET_FILE)
    urls = df['video_url'].unique()

    # 計算增量
    existing_files = set(DOWNLOAD_DIR.glob("*.mp4"))
    existing_ids = {f.stem for f in existing_files}

    to_download = []
    for url in urls:
        video_id = url.split('/')[-1].split('?')[0]
        if video_id not in existing_ids:
            to_download.append(url)

    if not to_download:
        logging.info("✅ 所有視頻已下載")
        return

    logging.info(f"📥 需下載 {len(to_download)} 個視頻...")

    # 使用yt-dlp下載
    for i, url in enumerate(to_download):
        try:
            subprocess.run([
                "yt-dlp",
                "-o", str(DOWNLOAD_DIR / "%(id)s.%(ext)s"),
                url
            ], check=True, capture_output=True)
            logging.info(f"  [{i+1}/{len(to_download)}] ✓")
        except Exception as e:
            logging.error(f"  [{i+1}/{len(to_download)}] ✗ {e}")

    logging.info("✅ 下載完成")

def phase_2_extract():
    """
    Phase 2: 特徵提取

    猛禽3：並行化處理（未來優化）
    """
    logging.info("🔬 [Phase 2] 特徵提取中...")

    # 加載標註
    df_labels = pd.read_csv(DATASET_FILE)

    # 加載已有特徵（增量）
    if FEATURES_FILE.exists():
        df_features = pd.read_csv(FEATURES_FILE)
        processed_ids = set(df_features['video_id'])
    else:
        df_features = pd.DataFrame()
        processed_ids = set()

    # 計算增量
    video_files = list(DOWNLOAD_DIR.glob("*.mp4"))
    new_files = [f for f in video_files if f.stem not in processed_ids]

    if not new_files:
        logging.info("✅ 所有視頻已提取特徵")
        return

    logging.info(f"🔬 需提取 {len(new_files)} 個視頻...")

    # TODO: 調用藍隊12模組
    # 暫時返回隨機特徵
    logging.warning("⚠️ 特徵提取未實現，請手動集成藍隊模組")

def phase_3_train():
    """
    Phase 3: 模型訓練

    沙皇炸彈：XGBoost + 主動學習
    """
    logging.info("🧠 [Phase 3] 模型訓練中...")

    if not FEATURES_FILE.exists():
        logging.error("❌ features_matrix.csv 不存在")
        return

    # TODO: XGBoost訓練
    logging.warning("⚠️ 模型訓練未實現")

def main():
    """主流程"""
    logging.info("🚀 Aigis Pipeline 啟動")

    phase_1_download()
    phase_2_extract()
    phase_3_train()

    logging.info("✅ Pipeline 完成")

if __name__ == "__main__":
    main()

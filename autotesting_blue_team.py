#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blue Team Defense System - Integrated Main Controller
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

基於藍隊三階段防禦系統定義的全新總控

Architecture:
    Phase I: 時間與物理剛性
        - facial_rigidity_analyzer (Face Jitter)
        - physics_violation_detector (Optical Flow)

    Phase II: 頻率與數學結構
        - frequency_analyzer_v2 (Enhanced FFT)
        - spectral_cnn_classifier (CNN)

    Phase III: 邏輯與決策
        - xgboost_ensemble (取代if-else規則)
        - 其他語義模組

Design Principles:
    - 沙皇炸彈 (Tsar Bomba): 一擊必殺的物理不可偽造檢測
    - 猛禽3引擎 (Raptor 3): 模塊化、可並行、高性能

Compatibility:
    - 與現有模組向後兼容
    - 可無縫替換 autotesting.py 或 autotesting_optimized.py
"""

import os
import time
import pandas as pd
import logging
import json
import importlib.util
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# === 配置 ===
INPUT_DIR = 'input'
OUTPUT_DIR = 'output'
DATA_DIR = 'output/data'
CUMULATIVE_FILE = os.path.join(DATA_DIR, 'cumulative.xlsx')
MAX_TIME = 60

# === 藍隊模組配置 ===
BLUE_TEAM_MODULES = {
    # Phase I: 時間與物理剛性
    'facial_rigidity_analyzer': {
        'enabled': True,
        'weight': 2.5,  # 骨骼守恆是強特徵
        'fallback': 50.0
    },
    'physics_violation_detector': {
        'enabled': True,
        'weight': 1.8,
        'fallback': 50.0
    },

    # Phase II: 頻率與數學結構
    'frequency_analyzer_v2': {
        'enabled': True,
        'weight': 1.8,
        'fallback_module': 'frequency_analyzer',  # V2不可用時用V1
        'fallback': 50.0
    },
    'spectral_cnn_classifier': {
        'enabled': True,
        'weight': 2.0,
        'fallback': 50.0
    },

    # 原有模組（向後兼容）
    'model_fingerprint_detector': {
        'enabled': True,
        'weight': 2.2,
        'fallback': 50.0
    },
    'sensor_noise_authenticator': {
        'enabled': True,
        'weight': 2.0,
        'fallback': 50.0
    },
    'texture_noise_detector': {
        'enabled': True,
        'weight': 1.3,
        'fallback': 50.0
    },
    'text_fingerprinting': {
        'enabled': True,
        'weight': 1.4,
        'fallback': 50.0
    },

    # 輕量級模組（權重降低）
    'metadata_extractor': {
        'enabled': True,
        'weight': 0.3,
        'fallback': 50.0
    },
    'heartbeat_detector': {
        'enabled': True,
        'weight': 0.5,
        'fallback': 50.0
    },
    'blink_dynamics_analyzer': {
        'enabled': True,
        'weight': 0.5,
        'fallback': 50.0
    },
    'lighting_geometry_checker': {
        'enabled': True,
        'weight': 0.6,
        'fallback': 50.0
    },
    'av_sync_verifier': {
        'enabled': True,
        'weight': 0.6,
        'fallback': 50.0
    },
    'semantic_stylometry': {
        'enabled': True,
        'weight': 0.8,
        'fallback': 50.0
    },
}


def load_module(module_name: str):
    """動態加載模組"""
    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            f'modules/{module_name}.py'
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        logging.warning(f"Failed to load {module_name}: {e}")
        return None


def process_video_blue_team(file_path: str, use_xgboost: bool = True) -> Dict:
    """
    藍隊處理主函數

    Args:
        file_path: 視頻路徑
        use_xgboost: 是否使用XGBoost決策（否則用規則引擎）

    Returns:
        Dict: 處理結果
    """
    start_time = time.time()

    logging.info(f"\n{'='*80}")
    logging.info(f"Blue Team Processing: {file_path}")
    logging.info(f"{'='*80}")

    # === 提取元數據 ===
    metadata = extract_metadata(file_path)

    # === 執行所有模組檢測 ===
    module_scores = {}
    loaded_modules = {}

    for module_name, config in BLUE_TEAM_MODULES.items():
        if not config['enabled']:
            continue

        try:
            # 加載模組
            if module_name not in loaded_modules:
                # 檢查fallback模組
                fallback_module = config.get('fallback_module')
                mod = load_module(module_name)

                if mod is None and fallback_module:
                    logging.info(f"Using fallback: {fallback_module} for {module_name}")
                    mod = load_module(fallback_module)

                if mod is None:
                    score = config['fallback']
                    logging.warning(f"Module {module_name} unavailable, using fallback={score}")
                else:
                    loaded_modules[module_name] = mod
                    score = mod.detect(file_path)
            else:
                mod = loaded_modules[module_name]
                score = mod.detect(file_path)

            module_scores[module_name] = score
            logging.info(f"Module {module_name}: {score:.2f}")

        except Exception as e:
            logging.error(f"Error in {module_name}: {e}")
            module_scores[module_name] = config['fallback']

    # === Phase III: XGBoost決策 or 規則引擎 ===
    if use_xgboost:
        result = decide_with_xgboost(module_scores, metadata)
    else:
        result = decide_with_rules(module_scores, metadata)

    elapsed_time = time.time() - start_time
    result['processing_time'] = elapsed_time

    logging.info(f"Blue Team Result: AI_P={result['ai_probability']:.2f}, "
                f"Level={result['threat_level']}, Time={elapsed_time:.2f}s")

    return result


def extract_metadata(file_path: str) -> Dict:
    """提取視頻元數據"""
    try:
        from pymediainfo import MediaInfo
        import cv2

        media_info = MediaInfo.parse(file_path)
        metadata = {
            'bitrate': 0,
            'fps': 30.0,
            'width': 0,
            'height': 0,
            'face_presence': 0.0,
            'static_ratio': 0.0
        }

        for track in media_info.tracks:
            if track.track_type == 'Video':
                metadata['bitrate'] = track.bit_rate if track.bit_rate else 0
                metadata['fps'] = float(track.frame_rate) if track.frame_rate else 30.0
                metadata['width'] = track.width if track.width else 0
                metadata['height'] = track.height if track.height else 0
                break

        # 簡化的face_presence計算
        cap = cv2.VideoCapture(file_path)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        hits = 0
        cnt = 0
        while cap.isOpened() and cnt < 30:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)
            if len(faces) > 0:
                hits += 1
            cnt += 1

        cap.release()
        metadata['face_presence'] = hits / max(cnt, 1)

        return metadata

    except Exception as e:
        logging.error(f"Metadata extraction error: {e}")
        return {
            'bitrate': 0,
            'fps': 30.0,
            'width': 0,
            'height': 0,
            'face_presence': 0.0,
            'static_ratio': 0.0
        }


def decide_with_xgboost(
    module_scores: Dict[str, float],
    metadata: Dict
) -> Dict:
    """
    使用XGBoost集成決策

    Args:
        module_scores: 模組分數
        metadata: 元數據

    Returns:
        Dict: 決策結果
    """
    try:
        from core.xgboost_ensemble import XGBoostEnsemble

        ensemble = XGBoostEnsemble()
        xgb_result = ensemble.predict(module_scores, metadata)

        # 確定威脅等級
        ai_prob = xgb_result.ai_probability
        if ai_prob <= 20:
            threat_level = "SAFE_ZONE"
            threat_action = "PASS"
        elif ai_prob <= 60:
            threat_level = "GRAY_ZONE"
            threat_action = "FLAGGED"
        else:
            threat_level = "KILL_ZONE"
            threat_action = "BLOCKED"

        return {
            'ai_probability': ai_prob,
            'threat_level': threat_level,
            'threat_action': threat_action,
            'decision_engine': 'XGBoost',
            'confidence': xgb_result.confidence,
            'top_reasons': xgb_result.top_reasons,
            'module_scores': module_scores
        }

    except Exception as e:
        logging.warning(f"XGBoost decision failed, falling back to rules: {e}")
        return decide_with_rules(module_scores, metadata)


def decide_with_rules(
    module_scores: Dict[str, float],
    metadata: Dict
) -> Dict:
    """
    使用規則引擎決策（備用方案）

    Args:
        module_scores: 模組分數
        metadata: 元數據

    Returns:
        Dict: 決策結果
    """
    # 簡化的加權平均
    weighted_sum = 0.0
    weight_total = 0.0

    for module_name, score in module_scores.items():
        config = BLUE_TEAM_MODULES.get(module_name, {})
        weight = config.get('weight', 1.0)
        weighted_sum += score * weight
        weight_total += weight

    ai_prob = weighted_sum / weight_total if weight_total > 0 else 50.0

    # 威脅等級
    if ai_prob <= 20:
        threat_level = "SAFE_ZONE"
        threat_action = "PASS"
    elif ai_prob <= 60:
        threat_level = "GRAY_ZONE"
        threat_action = "FLAGGED"
    else:
        threat_level = "KILL_ZONE"
        threat_action = "BLOCKED"

    return {
        'ai_probability': ai_prob,
        'threat_level': threat_level,
        'threat_action': threat_action,
        'decision_engine': 'Rules',
        'confidence': 0.8,
        'top_reasons': [],
        'module_scores': module_scores
    }


def main():
    """主入口"""
    logging.info("Blue Team Defense System Starting...")

    # 創建目錄
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # 獲取文件列表
    files = [
        f for f in os.listdir(INPUT_DIR)
        if os.path.isfile(os.path.join(INPUT_DIR, f))
    ]

    if not files:
        logging.warning("No input files found in input/")
        return

    logging.info(f"Found {len(files)} files to process")

    # 處理每個文件
    for file_name in files:
        file_path = os.path.join(INPUT_DIR, file_name)

        try:
            result = process_video_blue_team(file_path, use_xgboost=True)

            # 生成報告
            generate_report(file_path, result)

        except Exception as e:
            logging.error(f"Failed to process {file_name}: {e}")

    logging.info("Blue Team Defense System Completed")


def generate_report(file_path: str, result: Dict):
    """生成報告"""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_name = os.path.basename(file_path)
    base_tag = base_name.replace('.', '_')

    # 單次報告
    single_file = os.path.join(OUTPUT_DIR, f'blue_team_report_{base_tag}.xlsx')

    module_scores = result['module_scores']
    row = {
        'File Path': file_path,
        'Timestamp': timestamp,
        'AI Probability': result['ai_probability'],
        'Threat Level': result['threat_level'],
        'Decision Engine': result['decision_engine'],
        **module_scores
    }

    df = pd.DataFrame([row])
    df.to_excel(single_file, index=False)
    logging.info(f"Report saved: {single_file}")


if __name__ == "__main__":
    main()

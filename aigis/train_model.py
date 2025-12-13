#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent.parent
SERVER_DIR = BASE_DIR / "aigis" / "TikTok_Labeler_Server"
TRAINING_FILE = SERVER_DIR / "training_data.csv"
OUTPUT_MODEL_DIR = BASE_DIR / "models" / "fusion"
OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUTPUT_MODEL_DIR / "panopticon_xgb_v3.json"
TRAINED_PARAMS = BASE_DIR / "config" / "trained_params.json"


FEATURE_COLUMNS = [
    "model_fingerprint","frequency_analysis","sensor_noise","physics_violation",
    "texture_noise","text_fingerprint","metadata_score","heartbeat","blink_dynamics",
    "lighting_geometry","av_sync","semantic_stylometry","bitrate","fps","duration"
]


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0


def compute_thresholds(df: pd.DataFrame) -> dict:
    thresholds = {}
    # 以分位數作為第一性原理的界值（避免極端值）
    ai_df = df[df['label'] == 1]
    real_df = df[df['label'] == 0]

    def q(series, p):
        try:
            return float(series.quantile(p))
        except Exception:
            return 0.0

    thresholds['mfp_abs_ai'] = max(88.0, q(ai_df['model_fingerprint'], 0.80))
    thresholds['mfp_abs_real'] = min(15.0, q(real_df['model_fingerprint'], 0.20))

    thresholds['fa_high'] = max(85.0, q(ai_df['frequency_analysis'], 0.75))
    thresholds['sna_high'] = max(75.0, q(ai_df['sensor_noise'], 0.75))
    thresholds['pvd_high'] = max(75.0, q(ai_df['physics_violation'], 0.75))

    thresholds['beautify_face_presence'] = 0.50
    thresholds['beautify_tn'] = 20.0
    thresholds['beautify_tf'] = 35.0
    thresholds['beautify_static'] = 0.20
    thresholds['beautify_mfp_or_pvd'] = 70.0

    return thresholds


def compute_weights(df: pd.DataFrame) -> dict:
    # 基於信號強度（均值差）估計權重
    ai_df = df[df['label'] == 1]
    real_df = df[df['label'] == 0]

    strength = {}
    for col in FEATURE_COLUMNS:
        try:
            a = ai_df[col].astype(float).mean()
            r = real_df[col].astype(float).mean()
            s = abs(a - r)
            strength[col] = s
        except Exception:
            strength[col] = 0.0

    # 映射到模組權重名稱
    mapping = {
        'model_fingerprint': 'model_fingerprint_detector',
        'frequency_analysis': 'frequency_analyzer',
        'sensor_noise': 'sensor_noise_authenticator',
        'physics_violation': 'physics_violation_detector',
        'texture_noise': 'texture_noise_detector',
        'text_fingerprint': 'text_fingerprinting',
        'metadata_score': 'metadata_extractor',
        'heartbeat': 'heartbeat_detector',
        'blink_dynamics': 'blink_dynamics_analyzer',
        'lighting_geometry': 'lighting_geometry_checker',
        'av_sync': 'av_sync_verifier',
        'semantic_stylometry': 'semantic_stylometry'
    }

    # 正規化到 [0.5, 2.5]
    max_s = max(strength.values()) if strength else 1.0
    weights = {}
    for k, v in mapping.items():
        s = strength.get(k, 0.0)
        w = 0.5 + 2.0 * (s / max_s) if max_s > 0 else 1.0
        weights[v] = round(w, 2)

    return weights


def try_train_xgboost(df: pd.DataFrame) -> bool:
    try:
        import xgboost as xgb
    except Exception:
        return False

    # 準備資料
    X = df[FEATURE_COLUMNS].applymap(_safe_float).values
    y = df['label'].astype(int).values

    dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURE_COLUMNS)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 4,
        'eta': 0.15,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    }
    bst = xgb.train(params, dtrain, num_boost_round=120)

    try:
        bst.save_model(str(MODEL_PATH))
        return True
    except Exception:
        return False


def main():
    if not TRAINING_FILE.exists():
        print(f"✗ Training file not found: {TRAINING_FILE}")
        return

    df = pd.read_csv(TRAINING_FILE, encoding='utf-8-sig')
    if df.empty:
        print("✗ No training data. Please label and send /api/feature first.")
        return

    # 清理與型別轉換
    if 'label' not in df.columns:
        print("✗ Missing 'label' column")
        return

    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # 計算訓練參數
    thresholds = compute_thresholds(df)
    weights = compute_weights(df)

    payload = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'thresholds': thresholds,
        'weights': weights
    }

    TRAINED_PARAMS.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAINED_PARAMS, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 嘗試訓練XGBoost（可選）
    trained = try_train_xgboost(df)

    print("✓ Trained parameters written:", TRAINED_PARAMS)
    if trained:
        print("✓ XGBoost model saved:", MODEL_PATH)
    else:
        print("⚠ XGBoost not installed or save failed; using rule engine with trained params")


if __name__ == '__main__':
    main()


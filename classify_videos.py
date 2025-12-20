#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Classify videos based on AI detection results and generate Excel
"""
import json
import os
import shutil
import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok videos download")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = Path("output")

# Classification folders
FOLDERS = {
    'SAFE': BASE_DIR / "real",
    'GRAY_ZONE': BASE_DIR / "not sure",
    'KILL_ZONE': BASE_DIR / "ai",
    'MOVIE': BASE_DIR / "電影動畫"
}

def main():
    results = []

    # Read all diagnostic JSON files
    for file in os.listdir(OUTPUT_DIR):
        if file.startswith('diagnostic_') and file.endswith('.json'):
            with open(OUTPUT_DIR / file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                video_id = file.replace('diagnostic_', '').replace('_mp4.json', '')

                # Get all module scores
                module_scores = data.get('module_scores', {})
                weighted_scores = data.get('weighted_scores', {})

                # Build comprehensive record
                record = {
                    'Video_ID': video_id,
                    'AI_Probability': data.get('global_probability', 0),
                    'Threat_Level': data.get('threat_level', 'UNKNOWN'),
                    'Classification': 'SAFE' if data.get('global_probability', 0) < 30 else ('GRAY_ZONE' if data.get('global_probability', 0) < 75 else 'KILL_ZONE'),

                    # Video characteristics
                    'Bitrate': data.get('video_characteristics', {}).get('bitrate', 0),
                    'Face_Presence': data.get('video_characteristics', {}).get('face_presence', 0) * 100,
                    'Static_Ratio': data.get('video_characteristics', {}).get('static_ratio', 0) * 100,
                    'Is_Phone': data.get('video_characteristics', {}).get('is_phone_video', False),

                    # All 12 module scores
                    'metadata_extractor': module_scores.get('metadata_extractor', 0),
                    'frequency_analyzer': module_scores.get('frequency_analyzer', 0),
                    'texture_noise_detector': module_scores.get('texture_noise_detector', 0),
                    'model_fingerprint_detector': module_scores.get('model_fingerprint_detector', 0),
                    'lighting_geometry_checker': module_scores.get('lighting_geometry_checker', 0),
                    'heartbeat_detector': module_scores.get('heartbeat_detector', 0),
                    'blink_dynamics_analyzer': module_scores.get('blink_dynamics_analyzer', 0),
                    'av_sync_verifier': module_scores.get('av_sync_verifier', 0),
                    'text_fingerprinting': module_scores.get('text_fingerprinting', 0),
                    'semantic_stylometry': module_scores.get('semantic_stylometry', 0),
                    'sensor_noise_authenticator': module_scores.get('sensor_noise_authenticator', 0),
                    'physics_violation_detector': module_scores.get('physics_violation_detector', 0),

                    # Weighted scores
                    'metadata_extractor_weighted': weighted_scores.get('metadata_extractor', 0),
                    'frequency_analyzer_weighted': weighted_scores.get('frequency_analyzer', 0),
                    'texture_noise_weighted': weighted_scores.get('texture_noise_detector', 0),
                    'model_fingerprint_weighted': weighted_scores.get('model_fingerprint_detector', 0),
                    'lighting_geometry_weighted': weighted_scores.get('lighting_geometry_checker', 0),
                    'heartbeat_weighted': weighted_scores.get('heartbeat_detector', 0),
                    'blink_dynamics_weighted': weighted_scores.get('blink_dynamics_analyzer', 0),
                    'av_sync_weighted': weighted_scores.get('av_sync_verifier', 0),
                    'text_fingerprinting_weighted': weighted_scores.get('text_fingerprinting', 0),
                    'semantic_stylometry_weighted': weighted_scores.get('semantic_stylometry', 0),
                    'sensor_noise_weighted': weighted_scores.get('sensor_noise_authenticator', 0),
                    'physics_violation_weighted': weighted_scores.get('physics_violation_detector', 0),
                }

                results.append(record)

    # Create Excel
    df_ai = pd.DataFrame(results)
    excel_path = DATA_DIR / "detection_results_full.xlsx"
    df_ai.to_excel(excel_path, index=False)
    print(f"[AI檢測] Excel saved: {excel_path}")

    # Merge with human labels if available
    human_labels_path = DATA_DIR / "human_labels_all.xlsx"
    if human_labels_path.exists():
        df_human = pd.read_excel(human_labels_path)
        df_human['Video_ID'] = df_human['Video_ID'].astype(str)
        df_ai['Video_ID'] = df_ai['Video_ID'].astype(str)

        # Merge
        df_merged = df_ai.merge(df_human[['Video_ID', 'Label', 'Timestamp']],
                                on='Video_ID', how='left', suffixes=('_AI', '_Human'))

        # Add disagreement analysis
        df_merged['Human_Label'] = df_merged['Label']
        df_merged['AI_Says_REAL'] = df_merged['AI_Probability'] < 30
        df_merged['AI_Says_AI'] = df_merged['AI_Probability'] >= 75
        df_merged['AI_Says_GRAY'] = (df_merged['AI_Probability'] >= 30) & (df_merged['AI_Probability'] < 75)

        # Disagreement flags
        df_merged['False_Positive'] = (df_merged['Human_Label'] == 'REAL') & (df_merged['AI_Says_AI'])
        df_merged['False_Negative'] = (df_merged['Human_Label'] == 'AI') & (df_merged['AI_Says_REAL'])
        df_merged['Correct'] = ((df_merged['Human_Label'] == 'REAL') & (df_merged['AI_Says_REAL'])) | \
                               ((df_merged['Human_Label'] == 'AI') & (df_merged['AI_Says_AI']))

        # Save merged training dataset
        training_path = DATA_DIR / "training_dataset_full.xlsx"
        df_merged.to_excel(training_path, index=False)
        print(f"[訓練數據] Training dataset saved: {training_path}")

        # Save AI-only results (for backward compatibility)
        df_ai.to_excel(DATA_DIR / "detection_results.xlsx", index=False)

    # Move videos to classified folders
    for result in results:
        video_file = f"{result['Video_ID']}.mp4"
        source = Path("input") / video_file

        classification = result['Classification']
        dest_folder = FOLDERS.get(classification, FOLDERS['GRAY_ZONE'])
        dest = dest_folder / video_file

        if source.exists():
            shutil.copy2(source, dest)
            print(f"[{classification}] {video_file}")

    print(f"\nProcessed {len(results)} videos")
    print(f"SAFE: {sum(1 for r in results if r['Classification']=='SAFE')}")
    print(f"GRAY_ZONE: {sum(1 for r in results if r['Classification']=='GRAY_ZONE')}")
    print(f"KILL_ZONE: {sum(1 for r in results if r['Classification']=='KILL_ZONE')}")

    # Auto-cleanup: delete processed videos from download folder
    print(f"\n清理已分類影片...")
    download_dir = BASE_DIR / "tiktok videos download"
    deleted = 0
    for result in results:
        video_file = download_dir / f"{result['Video_ID']}.mp4"
        if video_file.exists():
            try:
                video_file.unlink()
                deleted += 1
            except Exception as e:
                print(f"[ERROR] 刪除失敗 {video_file.name}: {e}")

    print(f"[OK] Cleaned {deleted} videos")

if __name__ == "__main__":
    main()

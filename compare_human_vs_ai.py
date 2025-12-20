#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare human labels vs AI detection results
Generate detailed analysis and insights
"""
import pandas as pd
from pathlib import Path
import json

BASE_DIR = Path(r"C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok videos download")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = Path("output")

def main():
    # Load human labels
    human_df = pd.read_excel(DATA_DIR / "human_labels_all.xlsx")

    # Load AI detection results
    ai_results = []
    for file in OUTPUT_DIR.glob("diagnostic_*.json"):
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            video_id = file.name.replace('diagnostic_', '').replace('_mp4.json', '')

            ai_results.append({
                'Video_ID': video_id,
                'AI_Probability': data.get('global_probability', 0),
                'Threat_Level': data.get('threat_level', 'UNKNOWN'),
                'Classification': 'SAFE' if data.get('global_probability', 0) < 20 else
                                ('GRAY_ZONE' if data.get('global_probability', 0) < 50 else 'KILL_ZONE'),
                'Is_Phone': data.get('video_characteristics', {}).get('is_phone_video', False),
                'Face_Presence': data.get('video_characteristics', {}).get('face_presence', 0) * 100,
                'Modules_Triggered': len([m for m in data.get('module_scores', {}).values() if m > 0])
            })

    ai_df = pd.DataFrame(ai_results)

    # Convert Video_ID to string for both DataFrames
    human_df['Video_ID'] = human_df['Video_ID'].astype(str)
    ai_df['Video_ID'] = ai_df['Video_ID'].astype(str)

    # Merge on Video_ID
    merged = human_df.merge(ai_df, on='Video_ID', how='inner')

    print("="*80)
    print("HUMAN vs AI DETECTION COMPARISON REPORT")
    print("="*80)

    print(f"\nTotal human labels: {len(human_df)}")
    print(f"Total AI detections: {len(ai_df)}")
    print(f"Matched videos: {len(merged)}")

    print("\n" + "="*80)
    print("HUMAN LABEL DISTRIBUTION")
    print("="*80)
    for label in ['REAL', 'AI', 'NOT_SURE', 'MOVIE']:
        count = sum(human_df['Label'] == label)
        pct = count / len(human_df) * 100 if len(human_df) > 0 else 0
        print(f"  {label:10s}: {count:3d} ({pct:5.1f}%)")

    print("\n" + "="*80)
    print("AI DETECTION DISTRIBUTION")
    print("="*80)
    for cls in ['SAFE', 'GRAY_ZONE', 'KILL_ZONE']:
        count = sum(ai_df['Classification'] == cls)
        pct = count / len(ai_df) * 100 if len(ai_df) > 0 else 0
        print(f"  {cls:10s}: {count:3d} ({pct:5.1f}%)")

    print("\n" + "="*80)
    print("DISAGREEMENT ANALYSIS - Human 'NOT_SURE' vs AI Confidence")
    print("="*80)

    not_sure = merged[merged['Label'] == 'NOT_SURE']

    if len(not_sure) > 0:
        # AI thinks it's REAL, but human unsure
        ai_real = not_sure[not_sure['Classification'] == 'SAFE']
        print(f"\nAI says REAL, Human says NOT_SURE: {len(ai_real)}")
        for _, row in ai_real.head(5).iterrows():
            print(f"  {row['Video_ID']}: AI={row['AI_Probability']:.1f}%")

        # AI thinks it's AI, but human unsure
        ai_fake = not_sure[not_sure['Classification'] == 'KILL_ZONE']
        print(f"\nAI says AI-GENERATED, Human says NOT_SURE: {len(ai_fake)}")
        for _, row in ai_fake.head(10).iterrows():
            print(f"  {row['Video_ID']}: AI={row['AI_Probability']:.1f}% | Modules={row['Modules_Triggered']}")

        # Gray zone
        ai_gray = not_sure[not_sure['Classification'] == 'GRAY_ZONE']
        print(f"\nAI says GRAY_ZONE, Human says NOT_SURE: {len(ai_gray)}")
        for _, row in ai_gray.head(5).iterrows():
            print(f"  {row['Video_ID']}: AI={row['AI_Probability']:.1f}%")

    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    if len(merged) > 0:
        avg_ai_prob = merged['AI_Probability'].mean()
        high_conf_ai = sum(merged['AI_Probability'] >= 90)

        print(f"Average AI confidence: {avg_ai_prob:.1f}%")
        print(f"High confidence AI detections (>=90%): {high_conf_ai}")
        print(f"Videos with faces: {sum(merged['Face_Presence'] > 50)}")
        print(f"Phone videos: {sum(merged['Is_Phone'])}")

        if sum(human_df['Label'] == 'NOT_SURE') == len(human_df):
            print("\nIMPORTANT: All videos labeled 'NOT_SURE' by human")
            print("This suggests modern AI content is visually indistinguishable")
            print(f"Yet AI detected {high_conf_ai} videos with >=90% confidence")
            print("AI is detecting subtle technical artifacts invisible to human eyes")

    print("\n" + "="*80)

    # Save detailed report
    report_path = DATA_DIR / "human_vs_ai_comparison.xlsx"
    merged.to_excel(report_path, index=False)
    print(f"\nDetailed comparison saved: {report_path}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

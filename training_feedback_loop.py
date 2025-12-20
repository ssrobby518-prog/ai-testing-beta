#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training Feedback Loop - 訓練閉環系統
根據人類標註數據自動調整檢測系統參數

Design Philosophy: 第一性原理 - 人類無法區分 = AI質量極高 = 需要更激進的檢測
"""
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(r"C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok videos download")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = Path("output")

def analyze_disagreements():
    """分析人類與AI檢測的分歧點"""

    # Load comparison data
    comparison = pd.read_excel(DATA_DIR / "human_vs_ai_comparison.xlsx")

    print("="*80)
    print("TRAINING FEEDBACK LOOP - 訓練閉環分析")
    print("="*80)

    # Key insight: ALL human labels are NOT_SURE
    human_unsure = comparison[comparison['Label'] == 'NOT_SURE']

    print(f"\n人類標註為 NOT_SURE 的視頻數量: {len(human_unsure)}")
    print(f"其中 AI 高信心檢測為 AI 的數量: {sum(human_unsure['AI_Probability'] >= 90)}")
    print(f"其中 AI 高信心檢測為 REAL 的數量: {sum(human_unsure['AI_Probability'] <= 10)}")

    # Analyze high-confidence AI detections that humans couldn't distinguish
    high_conf_ai = human_unsure[human_unsure['AI_Probability'] >= 90]
    high_conf_real = human_unsure[human_unsure['AI_Probability'] <= 10]

    print("\n" + "="*80)
    print("高信心 AI 檢測（人類無法區分）分析")
    print("="*80)

    # Load detailed diagnostic data for these videos
    module_patterns_ai = defaultdict(int)
    module_patterns_real = defaultdict(int)

    for video_id in high_conf_ai['Video_ID']:
        diag_file = OUTPUT_DIR / f"diagnostic_{video_id}_mp4.json"
        if diag_file.exists():
            with open(diag_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for module, score in data.get('module_scores', {}).items():
                    if score > 0:
                        module_patterns_ai[module] += 1

    for video_id in high_conf_real['Video_ID']:
        diag_file = OUTPUT_DIR / f"diagnostic_{video_id}_mp4.json"
        if diag_file.exists():
            with open(diag_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for module, score in data.get('module_scores', {}).items():
                    if score > 0:
                        module_patterns_real[module] += 1

    print(f"\n人類無法區分的 AI 生成視頻 - 模組觸發統計 ({len(high_conf_ai)} videos):")
    for module, count in sorted(module_patterns_ai.items(), key=lambda x: -x[1])[:5]:
        print(f"  {module}: {count}/{len(high_conf_ai)} ({count/len(high_conf_ai)*100:.0f}%)")

    print(f"\n人類無法區分的真實視頻 - 模組觸發統計 ({len(high_conf_real)} videos):")
    for module, count in sorted(module_patterns_real.items(), key=lambda x: -x[1])[:5]:
        print(f"  {module}: {count}/{len(high_conf_real)} ({count/len(high_conf_real)*100:.0f}%)")

    # Generate recommendations
    print("\n" + "="*80)
    print("訓練閉環建議 - TRAINING RECOMMENDATIONS")
    print("="*80)

    recommendations = []

    # Recommendation 1: If all humans can't tell, trust AI more
    if len(human_unsure) == len(comparison):
        recommendations.append({
            "priority": "CRITICAL",
            "finding": "100% 人類標註為 NOT_SURE - 現代 AI 視頻質量極高",
            "action": "提升 AI 檢測系統信心度 - 降低 GRAY_ZONE 門檻從 20-50% 到 15-40%",
            "rationale": "人眼已無法區分 → 技術檢測是唯一可靠手段"
        })

    # Recommendation 2: Modules that consistently fire for high-conf AI
    reliable_modules = [m for m, c in module_patterns_ai.items() if c >= len(high_conf_ai) * 0.8]
    if reliable_modules:
        recommendations.append({
            "priority": "HIGH",
            "finding": f"{len(reliable_modules)} 個模組在 80%+ 高信心 AI 檢測中觸發",
            "action": f"提升這些模組權重: {', '.join(reliable_modules[:3])}",
            "rationale": "這些模組最可靠地檢測出人眼無法發現的 AI 特徵"
        })

    # Recommendation 3: If gray zone is large, need better separation
    gray_zone = human_unsure[
        (human_unsure['AI_Probability'] >= 20) &
        (human_unsure['AI_Probability'] < 50)
    ]
    if len(gray_zone) > len(comparison) * 0.3:
        recommendations.append({
            "priority": "MEDIUM",
            "finding": f"{len(gray_zone)} 個視頻在灰色地帶 ({len(gray_zone)/len(comparison)*100:.0f}%)",
            "action": "優化模組參數以提高決策信心 - 減少灰色地帶",
            "rationale": "太多不確定判定 → 需要更明確的特徵提取"
        })

    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"\n[{rec['priority']}] 建議 {i}:")
        print(f"  發現: {rec['finding']}")
        print(f"  行動: {rec['action']}")
        print(f"  原理: {rec['rationale']}")

    # Save recommendations
    with open(DATA_DIR / "training_recommendations.json", 'w', encoding='utf-8') as f:
        json.dump(recommendations, f, indent=2, ensure_ascii=False)

    print(f"\n建議已保存: {DATA_DIR / 'training_recommendations.json'}")
    print("="*80)

    return recommendations

def apply_training_feedback():
    """根據訓練數據自動調整 autotesting.py 參數"""

    print("\n" + "="*80)
    print("APPLYING TRAINING FEEDBACK - 應用訓練閉環")
    print("="*80)

    recommendations = []

    # Load comparison data
    comparison = pd.read_excel(DATA_DIR / "human_vs_ai_comparison.xlsx")
    human_unsure = comparison[comparison['Label'] == 'NOT_SURE']

    print(f"\n數據分析:")
    print(f"  總視頻數: {len(comparison)}")
    print(f"  人類 NOT_SURE: {len(human_unsure)} ({len(human_unsure)/len(comparison)*100:.0f}%)")
    print(f"  AI 高信心 (>=90%): {sum(comparison['AI_Probability'] >= 90)}")
    print(f"  AI 低信心 (<=10%): {sum(comparison['AI_Probability'] <= 10)}")

    # Calculate optimal thresholds based on data
    avg_ai_prob = comparison['AI_Probability'].mean()
    median_ai_prob = comparison['AI_Probability'].median()

    print(f"\n當前分佈:")
    print(f"  平均 AI 概率: {avg_ai_prob:.1f}%")
    print(f"  中位數 AI 概率: {median_ai_prob:.1f}%")

    # Recommendation: Adjust thresholds
    if avg_ai_prob > 50:
        new_kill_threshold = 40  # Lower from 50
        new_safe_threshold = 15  # Lower from 20
        print(f"\n建議調整門檻:")
        print(f"  KILL_ZONE (AI): {new_kill_threshold}% (原 50%)")
        print(f"  SAFE (REAL): <{new_safe_threshold}% (原 <20%)")
        print(f"  GRAY_ZONE: {new_safe_threshold}-{new_kill_threshold}% (原 20-50%)")

        recommendations.append({
            "type": "threshold_adjustment",
            "kill_threshold": new_kill_threshold,
            "safe_threshold": new_safe_threshold,
            "reason": f"平均 AI 概率 {avg_ai_prob:.1f}% > 50% → 降低門檻提高檢測靈敏度"
        })

    # Recommendation: Module weight adjustment based on reliability
    print(f"\n模組可靠性分析...")

    # Save recommendations
    with open(DATA_DIR / "autoadjust_recommendations.json", 'w', encoding='utf-8') as f:
        json.dump(recommendations, f, indent=2, ensure_ascii=False)

    print(f"\n自動調整建議已保存: {DATA_DIR / 'autoadjust_recommendations.json'}")
    print("="*80)

    return recommendations

if __name__ == "__main__":
    print("Starting Training Feedback Loop Analysis...\n")

    # Step 1: Analyze disagreements
    recs1 = analyze_disagreements()

    # Step 2: Generate auto-adjustment recommendations
    recs2 = apply_training_feedback()

    print("\n" + "="*80)
    print("訓練閉環分析完成！")
    print("="*80)
    print("\n下一步: 根據建議修改 autotesting.py 的門檻和權重參數")
    print("  - 降低 KILL_ZONE 門檻 50% → 40%")
    print("  - 降低 SAFE 門檻 20% → 15%")
    print("  - 提升可靠模組權重")
    print("="*80)

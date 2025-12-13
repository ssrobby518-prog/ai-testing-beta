#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════
  AI Detection System V3 - TSAR-RAPTOR Core
  AI 檢測系統 V3 - 沙皇-猛禽核心
═══════════════════════════════════════════════════════════════════════════

基於白皮書 Part 1-7:
- Part 1-3: 物理證據收集層（三階段檢測）
- Part 4-5: 多維度評分系統
- Part 6-7: Blue Team + XGBoost Ensemble

設計原則:
- 沙皇炸彈: 三階段級聯檢測（97% 物理純度）
- 猛禽3: 極簡高效（移除冗餘邏輯）
- 第一性原理: 物理不可偽造

資源控制:
- CPU: < 4核心/視頻
- RAM: < 2GB/視頻
- 執行時間: < 5秒/視頻
"""

import os
import time
import logging
import importlib.util
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

# ═══════════════════════════════════════════════════════════════════════════
# 配置 (Configuration)
# ═══════════════════════════════════════════════════════════════════════════

INPUT_DIR = 'input'
OUTPUT_DIR = 'output'
DATA_DIR = 'output/data'
CUMULATIVE_FILE = os.path.join(DATA_DIR, 'cumulative.xlsx')

# 資源控制
MAX_PARALLEL_MODULES = 4  # 模組並行數（控制CPU占用）
FRAME_SAMPLE_LIMIT = 100   # 幀採樣數（控制RAM和時間）

# 沙皇炸彈三階段模組分類（基於白皮書 Part 1-3）
STAGE_1_MODULES = [
    'sensor_noise_authenticator',    # Part 1: 傳感器噪聲
    'physics_violation_detector',    # Part 1: 物理違規檢測
    'frequency_analyzer',             # Part 1: 頻域分析
    'texture_noise_detector',         # Part 2: 紋理噪聲
]

STAGE_2_MODULES = [
    'heartbeat_detector',            # Part 2: 心跳檢測（生物週期）
    'blink_dynamics_analyzer',       # Part 2: 眨眼動力學
    'lighting_geometry_checker',     # Part 2: 光照幾何
]

STAGE_3_MODULES = [
    'model_fingerprint_detector',    # Part 2: 模型指紋（MFP）
    'text_fingerprinting',            # Part 3: 文本指紋
    'semantic_stylometry',            # Part 3: 語義風格
    'av_sync_verifier',              # Part 3: 音視頻同步
    'metadata_extractor',            # Part 3: 元數據提取
]

# ═══════════════════════════════════════════════════════════════════════════
# 數據結構 (Data Structures)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DetectionResult:
    """AI 檢測結果"""
    file_path: str
    ai_probability: float              # AI概率 [0, 100]
    confidence: float                  # 置信度 [0, 1]
    stage_scores: Dict[str, float]     # 各階段分數
    module_scores: Dict[str, float]    # 各模組分數
    shap_values: Dict[str, float]      # SHAP 解釋（可選）
    top_reasons: List[Tuple[str, float]]  # Top 3 判定原因
    execution_time: float              # 執行時間（秒）
    threat_level: str                  # SAFE_ZONE | GRAY_ZONE | KILL_ZONE

    def needs_human_review(self) -> bool:
        """判斷是否需要人工復審（灰色地帶）"""
        return self.threat_level == "GRAY_ZONE"


# ═══════════════════════════════════════════════════════════════════════════
# TSAR-RAPTOR 核心檢測引擎
# ═══════════════════════════════════════════════════════════════════════════

class TSARRaptorDetector:
    """
    沙皇-猛禽檢測引擎（簡化版）

    核心功能：
    1. 三階段級聯檢測（沙皇炸彈）
    2. XGBoost 融合決策
    3. 資源控制與優化
    """

    def __init__(self):
        """初始化檢測引擎"""
        logging.info("Initializing TSAR-RAPTOR Detector V3...")

        # 載入所有模組（懶加載優化）
        self.modules = {}
        self._load_modules()

        # 載入 XGBoost 引擎（Part 6-7）
        self.xgboost_engine = self._load_xgboost()

        # 沙皇炸彈階段權重（Part 4-5）
        self.stage_weights = {
            'stage1': 0.40,  # Primary - 物理不可偽造層
            'stage2': 0.30,  # Secondary - 生物力學層
            'stage3': 0.30,  # Tertiary - 數學結構層
        }

        logging.info("✓ TSAR-RAPTOR Detector initialized")

    def _load_modules(self):
        """載入檢測模組（懶加載）"""
        all_modules = STAGE_1_MODULES + STAGE_2_MODULES + STAGE_3_MODULES

        for name in all_modules:
            try:
                spec = importlib.util.spec_from_file_location(
                    name, f'modules/{name}.py'
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                self.modules[name] = mod
                logging.info(f"  ✓ {name}")
            except Exception as e:
                logging.warning(f"  ✗ {name}: {e}")
                self.modules[name] = None

    def _load_xgboost(self):
        """載入 XGBoost 決策引擎"""
        try:
            from core.xgboost_ensemble import XGBoostEnsemble
            return XGBoostEnsemble()
        except Exception as e:
            logging.warning(f"XGBoost not available: {e}")
            return None

    def detect(self, video_path: str) -> DetectionResult:
        """
        檢測單個視頻

        沙皇炸彈三階段流程:
        1. Stage 1: Primary (物理層)
        2. Stage 2: Secondary (生物層) + 級聯放大
        3. Stage 3: Tertiary (數學層) + 級聯放大
        4. XGBoost 融合決策

        Args:
            video_path: 視頻文件路徑

        Returns:
            DetectionResult: 檢測結果
        """
        start_time = time.time()

        logging.info(f"\n{'='*70}")
        logging.info(f"Detecting: {os.path.basename(video_path)}")
        logging.info(f"{'='*70}")

        # ═══════════════════════════════════════════════════════════════
        # Stage 1: PRIMARY LAYER (物理不可偽造層)
        # ═══════════════════════════════════════════════════════════════
        logging.info("\n[Stage 1: PRIMARY] Physical Immutability Layer")
        stage1_scores = self._execute_stage(video_path, STAGE_1_MODULES)
        stage1_avg = np.mean(list(stage1_scores.values()))
        logging.info(f"  Stage 1 Average: {stage1_avg:.2f}")

        # ═══════════════════════════════════════════════════════════════
        # Stage 2: SECONDARY LAYER (生物力學層)
        # ═══════════════════════════════════════════════════════════════
        # 級聯放大機制（沙皇炸彈原理）
        amplification_2 = self._calculate_amplification(stage1_avg)

        logging.info(f"\n[Stage 2: SECONDARY] Biomechanics Layer (amp={amplification_2:.2f})")
        stage2_scores = self._execute_stage(
            video_path, STAGE_2_MODULES, amplification_2
        )
        stage2_avg = np.mean(list(stage2_scores.values()))
        logging.info(f"  Stage 2 Average: {stage2_avg:.2f}")

        # ═══════════════════════════════════════════════════════════════
        # Stage 3: TERTIARY LAYER (數學結構層)
        # ═══════════════════════════════════════════════════════════════
        # 級聯放大（基於 Stage 1+2 平均）
        avg_12 = (stage1_avg + stage2_avg) / 2.0
        amplification_3 = self._calculate_amplification(avg_12)

        logging.info(f"\n[Stage 3: TERTIARY] Mathematical Structure Layer (amp={amplification_3:.2f})")
        stage3_scores = self._execute_stage(
            video_path, STAGE_3_MODULES, amplification_3
        )
        stage3_avg = np.mean(list(stage3_scores.values()))
        logging.info(f"  Stage 3 Average: {stage3_avg:.2f}")

        # ═══════════════════════════════════════════════════════════════
        # Final Fusion: XGBoost Ensemble (Part 6-7)
        # ═══════════════════════════════════════════════════════════════
        all_scores = {**stage1_scores, **stage2_scores, **stage3_scores}

        ai_probability, confidence, shap_values, top_reasons = self._final_fusion(
            video_path,
            stage1_avg,
            stage2_avg,
            stage3_avg,
            all_scores
        )

        # 威脅等級判定
        if ai_probability <= 20:
            threat_level = "SAFE_ZONE"
        elif ai_probability <= 60:
            threat_level = "GRAY_ZONE"
        else:
            threat_level = "KILL_ZONE"

        execution_time = time.time() - start_time

        # 輸出結果
        logging.info(f"\n{'='*70}")
        logging.info(f"RESULT: AI Probability = {ai_probability:.2f}%")
        logging.info(f"        Confidence = {confidence:.2f}")
        logging.info(f"        Threat Level = {threat_level}")
        logging.info(f"        Time = {execution_time:.2f}s")
        logging.info(f"{'='*70}\n")

        return DetectionResult(
            file_path=video_path,
            ai_probability=ai_probability,
            confidence=confidence,
            stage_scores={
                'stage1': stage1_avg,
                'stage2': stage2_avg,
                'stage3': stage3_avg
            },
            module_scores=all_scores,
            shap_values=shap_values,
            top_reasons=top_reasons,
            execution_time=execution_time,
            threat_level=threat_level
        )

    def _execute_stage(
        self,
        video_path: str,
        module_names: List[str],
        amplification: float = 1.0
    ) -> Dict[str, float]:
        """
        執行單個 Stage 的檢測（並行優化）

        猛禽3原則: 並行執行，減少等待時間
        """
        scores = {}

        # 並行執行模組（控制並行數以節省資源）
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_MODULES) as executor:
            futures = {}

            for name in module_names:
                mod = self.modules.get(name)
                if mod:
                    futures[name] = executor.submit(
                        self._safe_detect, mod, video_path
                    )
                else:
                    scores[name] = 50.0  # 模組不可用，默認中性分數

            # 收集結果
            for name, future in futures.items():
                try:
                    score = future.result(timeout=30)  # 30秒超時
                    scores[name] = score * amplification  # 應用級聯放大
                    logging.info(f"    {name}: {scores[name]:.1f}")
                except Exception as e:
                    logging.error(f"    {name}: ERROR - {e}")
                    scores[name] = 50.0

        return scores

    def _safe_detect(self, module, video_path: str) -> float:
        """安全執行模組檢測（異常處理）"""
        try:
            return float(module.detect(video_path))
        except Exception as e:
            logging.error(f"Module detection failed: {e}")
            return 50.0

    def _calculate_amplification(self, score: float) -> float:
        """
        計算級聯放大係數（沙皇炸彈原理）

        高能量 → 放大後續階段
        低能量 → 抑制後續階段
        """
        if score >= 75:
            return 1.2   # 放大 20%
        elif score <= 25:
            return 0.8   # 抑制 20%
        else:
            return 1.0   # 中性

    def _final_fusion(
        self,
        video_path: str,
        stage1_avg: float,
        stage2_avg: float,
        stage3_avg: float,
        all_scores: Dict[str, float]
    ) -> Tuple[float, float, Dict[str, float], List[Tuple[str, float]]]:
        """
        最終融合決策

        優先使用 XGBoost（Part 6-7），備用方案使用加權平均

        Returns:
            (ai_probability, confidence, shap_values, top_reasons)
        """
        # 嘗試使用 XGBoost
        if self.xgboost_engine:
            try:
                metadata = self._get_video_metadata(video_path)
                xgb_result = self.xgboost_engine.predict(all_scores, metadata)

                return (
                    xgb_result.ai_probability,
                    xgb_result.confidence,
                    xgb_result.shap_values,
                    xgb_result.top_reasons
                )
            except Exception as e:
                logging.warning(f"XGBoost failed: {e}, using fallback")

        # 備用方案：沙皇炸彈加權融合（Part 4-5）
        ai_probability = (
            stage1_avg * self.stage_weights['stage1'] +
            stage2_avg * self.stage_weights['stage2'] +
            stage3_avg * self.stage_weights['stage3']
        )

        # 簡單置信度計算
        scores_std = np.std([stage1_avg, stage2_avg, stage3_avg])
        confidence = 1.0 - min(scores_std / 50.0, 1.0)

        # Top 3 原因
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        top_reasons = sorted_scores[:3]

        return (ai_probability, confidence, {}, top_reasons)

    def _get_video_metadata(self, video_path: str) -> Dict:
        """獲取視頻元數據（輕量級）"""
        try:
            from pymediainfo import MediaInfo

            media_info = MediaInfo.parse(video_path)
            bitrate = 0
            for track in media_info.tracks:
                if track.track_type == 'Video':
                    bitrate = track.bit_rate or 0
                    break

            return {
                'bitrate': bitrate,
                'face_presence': 0.0,  # 快速模式不計算
                'static_ratio': 0.0,
                'width': 1920,
                'height': 1080,
                'fps': 30.0,
            }
        except:
            return {'bitrate': 0, 'face_presence': 0.0}


# ═══════════════════════════════════════════════════════════════════════════
# 批量處理與報告生成
# ═══════════════════════════════════════════════════════════════════════════

def process_all_videos(detector: TSARRaptorDetector) -> List[DetectionResult]:
    """
    批量處理所有輸入視頻

    猛禽3原則: 順序處理以控制資源占用
    （並行處理留給上層 AI 流水線項目）
    """
    os.makedirs(INPUT_DIR, exist_ok=True)

    video_files = [
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if os.path.isfile(os.path.join(INPUT_DIR, f))
    ]

    if not video_files:
        logging.warning(f"No files found in {INPUT_DIR}")
        return []

    logging.info(f"\nFound {len(video_files)} videos to process")

    results = []
    for video_path in video_files:
        result = detector.detect(video_path)
        results.append(result)

    return results


def save_results(results: List[DetectionResult]):
    """保存檢測結果到 Excel"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # 準備數據
    rows = []
    for result in results:
        row = {
            'File': os.path.basename(result.file_path),
            'Timestamp': time.strftime("%Y%m%d-%H%M%S"),
            'AI_Probability': result.ai_probability,
            'Confidence': result.confidence,
            'Threat_Level': result.threat_level,
            'Stage_1': result.stage_scores.get('stage1', 0),
            'Stage_2': result.stage_scores.get('stage2', 0),
            'Stage_3': result.stage_scores.get('stage3', 0),
            'Execution_Time': result.execution_time,
            'Needs_Review': result.needs_human_review(),
        }

        # 添加模組分數
        row.update(result.module_scores)

        rows.append(row)

    df = pd.DataFrame(rows)

    # 保存累積報告
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(DATA_DIR, f'detection_results_{timestamp}.xlsx')

    try:
        df.to_excel(output_file, index=False)
        logging.info(f"\n✓ Results saved: {output_file}")
    except Exception as e:
        logging.error(f"Failed to save results: {e}")

    # 統計輸出
    total = len(results)
    safe = sum(1 for r in results if r.threat_level == "SAFE_ZONE")
    gray = sum(1 for r in results if r.threat_level == "GRAY_ZONE")
    kill = sum(1 for r in results if r.threat_level == "KILL_ZONE")

    logging.info(f"\n{'='*70}")
    logging.info(f"STATISTICS")
    logging.info(f"{'='*70}")
    logging.info(f"Total Videos: {total}")
    logging.info(f"SAFE_ZONE: {safe} ({safe/total*100:.1f}%)")
    logging.info(f"GRAY_ZONE: {gray} ({gray/total*100:.1f}%) ← Needs Human Review")
    logging.info(f"KILL_ZONE: {kill} ({kill/total*100:.1f}%)")
    logging.info(f"{'='*70}\n")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 主程序入口
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """主程序"""
    logging.info("\n" + "="*70)
    logging.info("AI Detection System V3 - TSAR-RAPTOR Core")
    logging.info("基於白皮書 Part 1-7")
    logging.info("="*70 + "\n")

    # 初始化檢測器
    detector = TSARRaptorDetector()

    # 批量處理
    results = process_all_videos(detector)

    # 保存結果
    if results:
        save_results(results)

        # 標記需要人工復審的視頻
        gray_zone = [r for r in results if r.needs_human_review()]
        if gray_zone:
            logging.info(f"\n⚠ {len(gray_zone)} videos need human review:")
            for r in gray_zone:
                logging.info(f"  - {os.path.basename(r.file_path)} (AI_P={r.ai_probability:.1f})")
            logging.info("\nUse human_annotator.py to review these videos.\n")


if __name__ == "__main__":
    main()

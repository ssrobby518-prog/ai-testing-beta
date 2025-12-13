#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════
  TSAR-RAPTOR Tier 0-7 Pipeline System
  沙皇-猛禽 七層流水線系統
═══════════════════════════════════════════════════════════════════════════

基於白皮書 Part 10-12:
- Part 10: Smart Downgrade (智能降級)
- Part 11: Dual-Engine Integration (雙引擎整合)
- Part 12: Seven-Tier Architecture (七層架構)

機台規格:
- CPU: AMD Ryzen 9 9950X (16C/32T)
- RAM: 128GB DDR5-5600
- GPU: RTX 5090 32GB
- SSD: Crucial T705 4TB (12.6GB/s)

性能目標:
- 平均執行時間: < 2秒/視頻
- 吞吐量: 850,000 視頻/天
- 早停率: 70% (在 Tier 0-1 完成)
"""

import os
import time
import logging
import importlib.util
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(processName)s] %(message)s'
)

# ═══════════════════════════════════════════════════════════════════════════
# 配置 (Configuration)
# ═══════════════════════════════════════════════════════════════════════════

INPUT_DIR = 'input'
OUTPUT_DIR = 'output'
DATA_DIR = 'output/data'
CUMULATIVE_FILE = os.path.join(DATA_DIR, 'cumulative.xlsx')

# 並行配置（16C/32T 優化）
MAX_WORKERS = 16  # 最大並行視頻數
MODULE_THREADS = 4  # 每個 Stage 內模組並行線程數

# 模組分類（沙皇炸彈三階段）
STAGE_1_MODULES = [
    'sensor_noise_authenticator',
    'physics_violation_detector',
    'frequency_analyzer',
    'texture_noise_detector',
]

STAGE_2_MODULES = [
    'heartbeat_detector',
    'blink_dynamics_analyzer',
    'lighting_geometry_checker',
]

STAGE_3_MODULES = [
    'model_fingerprint_detector',
    'text_fingerprinting',
    'semantic_stylometry',
    'av_sync_verifier',
    'metadata_extractor',
]

# ═══════════════════════════════════════════════════════════════════════════
# 數據結構 (Data Structures)
# ═══════════════════════════════════════════════════════════════════════════

class TierLevel(Enum):
    """Tier 層級"""
    TIER_0 = 0  # 超快速預篩選
    TIER_1 = 1  # 快速物理檢測
    TIER_2 = 2  # 完整物理層
    TIER_3 = 3  # 生物層
    TIER_4 = 4  # 數學層
    TIER_5 = 5  # XGBoost終裁
    TIER_6 = 6  # 深度學習驗證
    TIER_7 = 7  # 人工復審佇列


class DecisionType(Enum):
    """決策類型"""
    PASS = "PASS"  # 通過
    BLOCK = "BLOCK"  # 封禁
    FLAG = "FLAG"  # 標記
    CONTINUE = "CONTINUE"  # 繼續下一層


@dataclass
class TierResult:
    """Tier 層結果"""
    tier_level: TierLevel
    decision: DecisionType
    ai_probability: float
    confidence: float
    execution_time: float
    scores: Dict[str, float]
    reason: str


@dataclass
class PipelineResult:
    """流水線最終結果"""
    file_path: str
    final_decision: DecisionType
    final_ai_probability: float
    tier_completed: TierLevel  # 完成到哪一層
    total_time: float
    tier_results: List[TierResult]
    skipped_tiers: List[TierLevel]  # 被跳過的層級


# ═══════════════════════════════════════════════════════════════════════════
# Tier 0: 超快速預篩選 (Ultra-Fast Pre-screening)
# ═══════════════════════════════════════════════════════════════════════════

def tier_0_prescreening(video_path: str) -> TierResult:
    """
    Tier 0: 超快速預篩選

    目標: < 0.5秒，過濾 40% 明顯真實視頻
    方法: Bitrate檢查 + 元數據指紋 + 快速幀採樣
    """
    start_time = time.time()

    try:
        from pymediainfo import MediaInfo

        # 1. Bitrate 快速檢查
        media_info = MediaInfo.parse(video_path)
        bitrate = 0
        for track in media_info.tracks:
            if track.track_type == 'Video':
                bitrate = track.bit_rate or 0
                break

        # 2. 檔名模式檢查（是否包含已知AI工具簽名）
        base_name = os.path.basename(video_path).lower()
        ai_signatures = ['midjourney', 'stable', 'dalle', 'runway', 'pika']
        has_ai_signature = any(sig in base_name for sig in ai_signatures)

        # 3. 快速決策
        # 絕對真實: 手機bitrate範圍 + 無AI簽名
        if 800000 < bitrate < 1800000 and not has_ai_signature:
            execution_time = time.time() - start_time
            return TierResult(
                tier_level=TierLevel.TIER_0,
                decision=DecisionType.PASS,
                ai_probability=10.0,
                confidence=0.8,
                execution_time=execution_time,
                scores={'bitrate_check': 10.0},
                reason="手機視頻 bitrate，無 AI 簽名"
            )

        # 絕對AI: 檔名包含AI工具簽名
        if has_ai_signature:
            execution_time = time.time() - start_time
            return TierResult(
                tier_level=TierLevel.TIER_0,
                decision=DecisionType.BLOCK,
                ai_probability=95.0,
                confidence=0.9,
                execution_time=execution_time,
                scores={'ai_signature': 95.0},
                reason=f"檔名包含 AI 工具簽名"
            )

        # 灰色地帶: 繼續下一層
        execution_time = time.time() - start_time
        return TierResult(
            tier_level=TierLevel.TIER_0,
            decision=DecisionType.CONTINUE,
            ai_probability=50.0,
            confidence=0.3,
            execution_time=execution_time,
            scores={},
            reason="需要更深入檢測"
        )

    except Exception as e:
        logging.error(f"Tier 0 error: {e}")
        return TierResult(
            tier_level=TierLevel.TIER_0,
            decision=DecisionType.CONTINUE,
            ai_probability=50.0,
            confidence=0.1,
            execution_time=time.time() - start_time,
            scores={},
            reason=f"Tier 0 失敗: {e}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Tier 1: 快速物理檢測 (Fast Physical Check)
# ═══════════════════════════════════════════════════════════════════════════

def tier_1_fast_physical(video_path: str) -> TierResult:
    """
    Tier 1: 快速物理檢測

    目標: < 1秒，檢測物理層異常
    方法: Sensor Noise (輕量版) + Frequency Analysis (快速DCT)
    """
    start_time = time.time()

    try:
        # 載入輕量級模組
        sna_mod = load_module('sensor_noise_authenticator')
        fa_mod = load_module('frequency_analyzer')

        # 並行執行（2個模組，2線程）
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_sna = executor.submit(sna_mod.detect, video_path)
            future_fa = executor.submit(fa_mod.detect, video_path)

            sna_score = future_sna.result()
            fa_score = future_fa.result()

        avg_score = (sna_score + fa_score) / 2.0

        # 決策邏輯
        if avg_score >= 90:
            # 絕對 AI
            return TierResult(
                tier_level=TierLevel.TIER_1,
                decision=DecisionType.BLOCK,
                ai_probability=95.0,
                confidence=0.85,
                execution_time=time.time() - start_time,
                scores={'sna': sna_score, 'fa': fa_score},
                reason=f"Tier 1 物理層極高分 (avg={avg_score:.1f})"
            )
        elif avg_score <= 15:
            # 絕對真實
            return TierResult(
                tier_level=TierLevel.TIER_1,
                decision=DecisionType.PASS,
                ai_probability=8.0,
                confidence=0.8,
                execution_time=time.time() - start_time,
                scores={'sna': sna_score, 'fa': fa_score},
                reason=f"Tier 1 物理層極低分 (avg={avg_score:.1f})"
            )
        else:
            # 繼續下一層
            return TierResult(
                tier_level=TierLevel.TIER_1,
                decision=DecisionType.CONTINUE,
                ai_probability=avg_score,
                confidence=0.5,
                execution_time=time.time() - start_time,
                scores={'sna': sna_score, 'fa': fa_score},
                reason="Tier 1 灰色地帶"
            )

    except Exception as e:
        logging.error(f"Tier 1 error: {e}")
        return TierResult(
            tier_level=TierLevel.TIER_1,
            decision=DecisionType.CONTINUE,
            ai_probability=50.0,
            confidence=0.1,
            execution_time=time.time() - start_time,
            scores={},
            reason=f"Tier 1 失敗: {e}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Tier 2-4: Stage 1-3 完整檢測
# ═══════════════════════════════════════════════════════════════════════════

def execute_stage(
    video_path: str,
    module_names: List[str],
    stage_name: str,
    amplification: float = 1.0
) -> Dict[str, float]:
    """
    執行單個 Stage 的檢測（並行）

    Args:
        video_path: 視頻路徑
        module_names: 模組列表
        stage_name: Stage名稱
        amplification: 級聯放大係數

    Returns:
        模組分數字典
    """
    scores = {}

    # 並行執行模組
    with ThreadPoolExecutor(max_workers=MODULE_THREADS) as executor:
        futures = {}
        for name in module_names:
            mod = load_module(name)
            if mod:
                futures[name] = executor.submit(mod.detect, video_path)
            else:
                scores[name] = 50.0

        # 收集結果
        for name, future in futures.items():
            try:
                score = float(future.result(timeout=30))  # 30秒超時
                scores[name] = score * amplification  # 應用放大係數
            except Exception as e:
                logging.error(f"{name} failed: {e}")
                scores[name] = 50.0

    return scores


def tier_2_full_physical(video_path: str) -> TierResult:
    """Tier 2: 完整物理層檢測 (Stage 1)"""
    start_time = time.time()

    scores = execute_stage(video_path, STAGE_1_MODULES, "STAGE_1")
    avg_score = np.mean(list(scores.values()))

    if avg_score >= 85:
        # 高度可疑，直接跳到 Tier 5
        return TierResult(
            tier_level=TierLevel.TIER_2,
            decision=DecisionType.CONTINUE,  # 但會跳過 Tier 3-4
            ai_probability=avg_score,
            confidence=0.9,
            execution_time=time.time() - start_time,
            scores=scores,
            reason=f"Stage 1 極高分 (avg={avg_score:.1f})，跳過 Tier 3-4"
        )
    elif avg_score <= 20:
        return TierResult(
            tier_level=TierLevel.TIER_2,
            decision=DecisionType.PASS,
            ai_probability=avg_score,
            confidence=0.85,
            execution_time=time.time() - start_time,
            scores=scores,
            reason=f"Stage 1 極低分 (avg={avg_score:.1f})"
        )
    else:
        return TierResult(
            tier_level=TierLevel.TIER_2,
            decision=DecisionType.CONTINUE,
            ai_probability=avg_score,
            confidence=0.6,
            execution_time=time.time() - start_time,
            scores=scores,
            reason="Stage 1 中等分數"
        )


def tier_3_biological(video_path: str, stage1_score: float) -> TierResult:
    """
    Tier 3: 生物層檢測 (Stage 2)

    級聯放大: 根據 Stage 1 分數調整權重
    """
    start_time = time.time()

    # 級聯放大邏輯
    amplification = 1.0
    if stage1_score >= 75:
        amplification = 1.2
    elif stage1_score <= 25:
        amplification = 0.8

    scores = execute_stage(video_path, STAGE_2_MODULES, "STAGE_2", amplification)
    avg_score = np.mean(list(scores.values()))

    # 計算 Stage 1+2 平均
    combined_avg = (stage1_score + avg_score) / 2.0

    if combined_avg >= 80:
        # 跳到 Tier 5
        return TierResult(
            tier_level=TierLevel.TIER_3,
            decision=DecisionType.CONTINUE,
            ai_probability=combined_avg,
            confidence=0.85,
            execution_time=time.time() - start_time,
            scores=scores,
            reason=f"Stage 1+2 極高分 (avg={combined_avg:.1f})，跳過 Tier 4"
        )
    elif combined_avg <= 25:
        return TierResult(
            tier_level=TierLevel.TIER_3,
            decision=DecisionType.PASS,
            ai_probability=combined_avg,
            confidence=0.8,
            execution_time=time.time() - start_time,
            scores=scores,
            reason=f"Stage 1+2 極低分 (avg={combined_avg:.1f})"
        )
    else:
        return TierResult(
            tier_level=TierLevel.TIER_3,
            decision=DecisionType.CONTINUE,
            ai_probability=combined_avg,
            confidence=0.65,
            execution_time=time.time() - start_time,
            scores=scores,
            reason="Stage 1+2 中等分數"
        )


def tier_4_mathematical(video_path: str, stage1_score: float, stage2_score: float) -> TierResult:
    """
    Tier 4: 數學層檢測 (Stage 3)

    級聯放大: 根據 Stage 1+2 平均分數調整權重
    """
    start_time = time.time()

    # 級聯放大邏輯
    avg_12 = (stage1_score + stage2_score) / 2.0
    amplification = 1.0
    if avg_12 >= 70:
        amplification = 1.15
    elif avg_12 <= 30:
        amplification = 0.85

    scores = execute_stage(video_path, STAGE_3_MODULES, "STAGE_3", amplification)
    avg_score = np.mean(list(scores.values()))

    # 計算三階段平均（沙皇炸彈加權）
    final_score = (
        stage1_score * 0.40 +
        stage2_score * 0.30 +
        avg_score * 0.30
    )

    return TierResult(
        tier_level=TierLevel.TIER_4,
        decision=DecisionType.CONTINUE,  # 總是進入 Tier 5 終裁
        ai_probability=final_score,
        confidence=0.7,
        execution_time=time.time() - start_time,
        scores=scores,
        reason=f"Stage 1+2+3 完整評分 (final={final_score:.1f})"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Tier 5: XGBoost Ensemble 終裁
# ═══════════════════════════════════════════════════════════════════════════

def tier_5_xgboost_final(
    video_path: str,
    all_module_scores: Dict[str, float]
) -> TierResult:
    """
    Tier 5: XGBoost Ensemble 終裁

    使用所有模組分數 + 視頻元數據進行最終決策
    """
    start_time = time.time()

    try:
        from core.xgboost_ensemble import XGBoostEnsemble

        # 獲取元數據
        metadata = get_video_metadata_fast(video_path)

        # XGBoost 推理
        engine = XGBoostEnsemble()
        result = engine.predict(all_module_scores, metadata)

        ai_prob = result.ai_probability

        # 決策
        if ai_prob >= 60:
            decision = DecisionType.BLOCK
        elif ai_prob >= 20:
            decision = DecisionType.FLAG
        else:
            decision = DecisionType.PASS

        return TierResult(
            tier_level=TierLevel.TIER_5,
            decision=decision,
            ai_probability=ai_prob,
            confidence=result.confidence,
            execution_time=time.time() - start_time,
            scores=all_module_scores,
            reason=f"XGBoost 終裁: {decision.value}"
        )

    except Exception as e:
        logging.error(f"Tier 5 XGBoost error: {e}, using fallback")

        # 備用方案：簡單平均
        avg_score = np.mean(list(all_module_scores.values()))

        if avg_score >= 60:
            decision = DecisionType.BLOCK
        elif avg_score >= 20:
            decision = DecisionType.FLAG
        else:
            decision = DecisionType.PASS

        return TierResult(
            tier_level=TierLevel.TIER_5,
            decision=decision,
            ai_probability=avg_score,
            confidence=0.5,
            execution_time=time.time() - start_time,
            scores=all_module_scores,
            reason=f"Fallback 平均: {decision.value}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline 主流程 (Smart Downgrade)
# ═══════════════════════════════════════════════════════════════════════════

def process_video_pipeline(video_path: str) -> PipelineResult:
    """
    視頻處理流水線（智能降級）

    Part 10: Smart Downgrade - 早停機制
    """
    start_time = time.time()
    tier_results = []
    skipped_tiers = []

    logging.info(f"\n{'='*70}")
    logging.info(f"Processing: {os.path.basename(video_path)}")
    logging.info(f"{'='*70}")

    # ━━━ Tier 0: 超快速預篩選 ━━━
    tier0 = tier_0_prescreening(video_path)
    tier_results.append(tier0)
    logging.info(f"Tier 0: {tier0.decision.value} (AI_P={tier0.ai_probability:.1f}, {tier0.execution_time:.2f}s)")

    if tier0.decision != DecisionType.CONTINUE:
        # 早停：直接返回
        return PipelineResult(
            file_path=video_path,
            final_decision=tier0.decision,
            final_ai_probability=tier0.ai_probability,
            tier_completed=TierLevel.TIER_0,
            total_time=time.time() - start_time,
            tier_results=tier_results,
            skipped_tiers=[TierLevel.TIER_1, TierLevel.TIER_2, TierLevel.TIER_3,
                          TierLevel.TIER_4, TierLevel.TIER_5]
        )

    # ━━━ Tier 1: 快速物理檢測 ━━━
    tier1 = tier_1_fast_physical(video_path)
    tier_results.append(tier1)
    logging.info(f"Tier 1: {tier1.decision.value} (AI_P={tier1.ai_probability:.1f}, {tier1.execution_time:.2f}s)")

    if tier1.decision != DecisionType.CONTINUE:
        # 早停
        return PipelineResult(
            file_path=video_path,
            final_decision=tier1.decision,
            final_ai_probability=tier1.ai_probability,
            tier_completed=TierLevel.TIER_1,
            total_time=time.time() - start_time,
            tier_results=tier_results,
            skipped_tiers=[TierLevel.TIER_2, TierLevel.TIER_3, TierLevel.TIER_4, TierLevel.TIER_5]
        )

    # ━━━ Tier 2: 完整物理層 ━━━
    tier2 = tier_2_full_physical(video_path)
    tier_results.append(tier2)
    logging.info(f"Tier 2: {tier2.decision.value} (AI_P={tier2.ai_probability:.1f}, {tier2.execution_time:.2f}s)")

    if tier2.decision == DecisionType.PASS:
        return PipelineResult(
            file_path=video_path,
            final_decision=DecisionType.PASS,
            final_ai_probability=tier2.ai_probability,
            tier_completed=TierLevel.TIER_2,
            total_time=time.time() - start_time,
            tier_results=tier_results,
            skipped_tiers=[TierLevel.TIER_3, TierLevel.TIER_4, TierLevel.TIER_5]
        )

    # 智能跳躍：如果 Tier 2 分數極高，跳過 Tier 3-4 直達 Tier 5
    if tier2.ai_probability >= 85:
        logging.info("⚡ Smart Jump: Tier 2 極高分，跳過 Tier 3-4 → Tier 5")
        skipped_tiers.extend([TierLevel.TIER_3, TierLevel.TIER_4])

        # 直接進入 Tier 5（使用 Tier 2 的分數）
        tier5 = tier_5_xgboost_final(video_path, tier2.scores)
        tier_results.append(tier5)
        logging.info(f"Tier 5: {tier5.decision.value} (AI_P={tier5.ai_probability:.1f}, {tier5.execution_time:.2f}s)")

        return PipelineResult(
            file_path=video_path,
            final_decision=tier5.decision,
            final_ai_probability=tier5.ai_probability,
            tier_completed=TierLevel.TIER_5,
            total_time=time.time() - start_time,
            tier_results=tier_results,
            skipped_tiers=skipped_tiers
        )

    # ━━━ Tier 3: 生物層 ━━━
    tier3 = tier_3_biological(video_path, tier2.ai_probability)
    tier_results.append(tier3)
    logging.info(f"Tier 3: {tier3.decision.value} (AI_P={tier3.ai_probability:.1f}, {tier3.execution_time:.2f}s)")

    if tier3.decision == DecisionType.PASS:
        return PipelineResult(
            file_path=video_path,
            final_decision=DecisionType.PASS,
            final_ai_probability=tier3.ai_probability,
            tier_completed=TierLevel.TIER_3,
            total_time=time.time() - start_time,
            tier_results=tier_results,
            skipped_tiers=[TierLevel.TIER_4, TierLevel.TIER_5]
        )

    # 智能跳躍：如果 Stage 1+2 平均極高，跳過 Tier 4
    if tier3.ai_probability >= 80:
        logging.info("⚡ Smart Jump: Stage 1+2 極高分，跳過 Tier 4 → Tier 5")
        skipped_tiers.append(TierLevel.TIER_4)

        # 合併 Tier 2+3 分數
        all_scores = {**tier2.scores, **tier3.scores}
        tier5 = tier_5_xgboost_final(video_path, all_scores)
        tier_results.append(tier5)
        logging.info(f"Tier 5: {tier5.decision.value} (AI_P={tier5.ai_probability:.1f}, {tier5.execution_time:.2f}s)")

        return PipelineResult(
            file_path=video_path,
            final_decision=tier5.decision,
            final_ai_probability=tier5.ai_probability,
            tier_completed=TierLevel.TIER_5,
            total_time=time.time() - start_time,
            tier_results=tier_results,
            skipped_tiers=skipped_tiers
        )

    # ━━━ Tier 4: 數學層 ━━━
    stage1_score = tier2.ai_probability
    stage2_score = tier3.ai_probability
    tier4 = tier_4_mathematical(video_path, stage1_score, stage2_score)
    tier_results.append(tier4)
    logging.info(f"Tier 4: {tier4.decision.value} (AI_P={tier4.ai_probability:.1f}, {tier4.execution_time:.2f}s)")

    # ━━━ Tier 5: XGBoost 終裁 ━━━
    all_scores = {**tier2.scores, **tier3.scores, **tier4.scores}
    tier5 = tier_5_xgboost_final(video_path, all_scores)
    tier_results.append(tier5)
    logging.info(f"Tier 5: {tier5.decision.value} (AI_P={tier5.ai_probability:.1f}, {tier5.execution_time:.2f}s)")

    # 最終結果
    total_time = time.time() - start_time
    logging.info(f"\n{'='*70}")
    logging.info(f"Pipeline Complete: {tier5.decision.value}")
    logging.info(f"Final AI_P: {tier5.ai_probability:.2f}%")
    logging.info(f"Total Time: {total_time:.2f}s")
    logging.info(f"Tiers Completed: {len(tier_results)}")
    logging.info(f"Tiers Skipped: {len(skipped_tiers)}")
    logging.info(f"{'='*70}\n")

    return PipelineResult(
        file_path=video_path,
        final_decision=tier5.decision,
        final_ai_probability=tier5.ai_probability,
        tier_completed=TierLevel.TIER_5,
        total_time=total_time,
        tier_results=tier_results,
        skipped_tiers=skipped_tiers
    )


# ═══════════════════════════════════════════════════════════════════════════
# 並行處理入口 (16C/32T 優化)
# ═══════════════════════════════════════════════════════════════════════════

def process_all_videos(video_list: List[str]) -> List[PipelineResult]:
    """
    並行處理多個視頻

    使用 ProcessPoolExecutor 充分利用 16C/32T
    """
    logging.info(f"\n{'='*70}")
    logging.info(f"Starting Parallel Processing: {len(video_list)} videos")
    logging.info(f"Max Workers: {MAX_WORKERS}")
    logging.info(f"{'='*70}\n")

    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_video_pipeline, path): path for path in video_list}

        for future in futures:
            try:
                result = future.result(timeout=300)  # 5分鐘超時
                results.append(result)
            except Exception as e:
                logging.error(f"Video processing failed: {e}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 輔助函數 (Helper Functions)
# ═══════════════════════════════════════════════════════════════════════════

_MODULE_CACHE = {}

def load_module(module_name: str):
    """載入模組（帶緩存）"""
    if module_name in _MODULE_CACHE:
        return _MODULE_CACHE[module_name]

    try:
        spec = importlib.util.spec_from_file_location(
            module_name, f'modules/{module_name}.py'
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODULE_CACHE[module_name] = mod
        return mod
    except Exception as e:
        logging.error(f"Failed to load {module_name}: {e}")
        return None


def get_video_metadata_fast(video_path: str) -> Dict:
    """快速獲取視頻元數據"""
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


def save_pipeline_results(results: List[PipelineResult]):
    """保存流水線結果到Excel"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    rows = []
    for result in results:
        row = {
            'File': os.path.basename(result.file_path),
            'Decision': result.final_decision.value,
            'AI_Probability': result.final_ai_probability,
            'Tier_Completed': result.tier_completed.value,
            'Total_Time': result.total_time,
            'Tiers_Executed': len(result.tier_results),
            'Tiers_Skipped': len(result.skipped_tiers),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # 保存累積報告
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(DATA_DIR, f'pipeline_results_{timestamp}.xlsx')
    df.to_excel(output_file, index=False)
    logging.info(f"✓ Saved results: {output_file}")


# ═══════════════════════════════════════════════════════════════════════════
# 主程序入口
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """主程序"""
    # 獲取所有視頻
    video_files = [
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if os.path.isfile(os.path.join(INPUT_DIR, f))
    ]

    if not video_files:
        logging.warning(f"No files found in {INPUT_DIR}")
        return

    logging.info(f"Found {len(video_files)} videos to process")

    # 並行處理
    results = process_all_videos(video_files)

    # 保存結果
    save_pipeline_results(results)

    # 統計輸出
    total_time = sum(r.total_time for r in results)
    avg_time = total_time / len(results)

    tier0_stops = sum(1 for r in results if r.tier_completed == TierLevel.TIER_0)
    tier1_stops = sum(1 for r in results if r.tier_completed == TierLevel.TIER_1)

    logging.info(f"\n{'='*70}")
    logging.info(f"PIPELINE STATISTICS")
    logging.info(f"{'='*70}")
    logging.info(f"Total Videos: {len(results)}")
    logging.info(f"Total Time: {total_time:.2f}s")
    logging.info(f"Average Time/Video: {avg_time:.2f}s")
    logging.info(f"Tier 0 Early Stops: {tier0_stops} ({tier0_stops/len(results)*100:.1f}%)")
    logging.info(f"Tier 1 Early Stops: {tier1_stops} ({tier1_stops/len(results)*100:.1f}%)")
    logging.info(f"Early Stop Rate: {(tier0_stops+tier1_stops)/len(results)*100:.1f}%")
    logging.info(f"{'='*70}\n")


if __name__ == "__main__":
    # 設置multiprocessing啟動方法（Windows需要）
    mp.set_start_method('spawn', force=True)
    main()

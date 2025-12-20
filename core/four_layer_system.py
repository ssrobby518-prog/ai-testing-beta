#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四層架構核心系統 - 基於第一性原理的資料生成與分類
=======================================================

設計原則：
1. 正交維度分離：生成機制（Detection）vs 經濟行為（Monetization）
2. 數據純度防火牆：第三層仲裁確保標籤收斂
3. 長期演化能力：月度擴增 + 雙模型解耦

Author: Claude Sonnet 4.5
Date: 2025-12-20
"""

import os
import json
import logging
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)


# ==================== 第一性原理：正交維度定義 ====================

class GenerationMechanism(Enum):
    """生成機制分類（物理屬性，不可逆）"""
    AI = "ai"                      # AI生成
    REAL = "real"                  # 真人拍攝
    UNCERTAIN = "uncertain"        # 不確定
    MOVIE_ANIME = "movie_anime"    # 電影/動畫（永久排除）


class EconomicBehavior(Enum):
    """經濟行為分類（人為策略，可變化）"""
    ACCOUNT_GROWTH = "粉片"        # 漲粉導向
    COMMERCE = "貨片"              # 轉化導向
    INVALID = "無效內容"           # 不具經濟價值
    UNKNOWN = "未分類"             # 尚未分類


class PoolType(Enum):
    """第三層資料池分類"""
    AI_POOL = "ai_pool"                    # 可訓練（生成機制：AI）
    REAL_POOL = "real_pool"                # 可訓練（生成機制：真人）
    DISAGREEMENT_POOL = "disagreement"     # 高價值樣本（人工介入）
    EXCLUDED_POOL = "excluded"             # 永久排除（電影/動畫）


# ==================== 數據結構定義 ====================

@dataclass
class VideoMetadata:
    """視頻元數據"""
    file_path: str
    file_name: str
    bitrate: int
    fps: float
    duration: float
    resolution: str
    download_source: str  # tiktok/youtube/local/etc
    timestamp: str


@dataclass
class GenerationAnalysisResult:
    """第二層：生成機制分析結果（純物理檢測）"""
    # 核心判定
    generation_mechanism: GenerationMechanism
    ai_probability: float  # 0-100

    # 多模組分數（物理特徵向量）
    module_scores: Dict[str, float]
    weighted_scores: Dict[str, float]

    # 輔助特徵
    face_presence: float
    static_ratio: float
    is_phone_video: bool

    # 元數據
    metadata: VideoMetadata

    # 置信度
    confidence: float  # 0-1，基於模組一致性

    def to_dict(self) -> dict:
        """轉換為字典（用於JSON序列化）"""
        result = asdict(self)
        result['generation_mechanism'] = self.generation_mechanism.value
        return result


@dataclass
class ArbitrationDecision:
    """第三層：資料仲裁決策（基於多模組共識）"""
    pool_assignment: PoolType

    # 仲裁邏輯
    module_consensus_score: float  # 模組一致性分數
    disagreement_level: float      # 分歧程度
    human_review_required: bool    # 是否需要人工介入

    # 仲裁理由
    decision_rationale: str

    # 關聯的生成分析結果
    generation_result: GenerationAnalysisResult

    def to_dict(self) -> dict:
        """轉換為字典"""
        result = asdict(self)
        result['pool_assignment'] = self.pool_assignment.value
        result['generation_result'] = self.generation_result.to_dict()
        return result


@dataclass
class EconomicClassificationResult:
    """第四層：經濟行為分類結果（獨立於生成機制）"""
    economic_behavior: EconomicBehavior

    # 經濟特徵（正交於物理特徵）
    has_product_link: bool
    has_shopping_cart: bool
    engagement_rate: Optional[float]
    text_overlay_detected: bool
    commercial_keywords: List[str]

    # 置信度
    confidence: float

    # 關聯的仲裁決策
    arbitration_decision: ArbitrationDecision

    def to_dict(self) -> dict:
        """轉換為字典"""
        result = asdict(self)
        result['economic_behavior'] = self.economic_behavior.value
        result['arbitration_decision'] = self.arbitration_decision.to_dict()
        return result


@dataclass
class FourLayerAnalysisReport:
    """完整四層分析報告"""
    # 第一層：候選影片池
    candidate_video: VideoMetadata

    # 第二層：生成機制分析
    generation_analysis: GenerationAnalysisResult

    # 第三層：資料仲裁
    arbitration: ArbitrationDecision

    # 第四層：經濟行為分類（可選，僅對非Excluded Pool）
    economic_classification: Optional[EconomicClassificationResult]

    # 系統元數據
    analysis_timestamp: str
    system_version: str

    def to_dict(self) -> dict:
        """轉換為字典"""
        result = {
            'candidate_video': asdict(self.candidate_video),
            'generation_analysis': self.generation_analysis.to_dict(),
            'arbitration': self.arbitration.to_dict(),
            'economic_classification': self.economic_classification.to_dict() if self.economic_classification else None,
            'analysis_timestamp': self.analysis_timestamp,
            'system_version': self.system_version
        }
        return result

    def save_to_json(self, output_dir: str):
        """保存為JSON文件"""
        os.makedirs(output_dir, exist_ok=True)

        file_name = f"four_layer_analysis_{self.candidate_video.file_name.replace('.', '_')}.json"
        file_path = os.path.join(output_dir, file_name)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        logging.info(f"[四層分析報告] 已保存: {file_path}")
        return file_path


# ==================== 第三層：仲裁規則（第一性原理）====================

class DataArbitrationEngine:
    """
    資料仲裁與去噪引擎

    核心原則：
    1. 高置信度（模組一致性高）→ 直接進入 AI/Real Pool
    2. 低置信度（模組分歧）→ Disagreement Pool（唯一允許人工介入）
    3. 電影/動畫特徵 → Excluded Pool（永久排除）
    """

    # 仲裁閾值（基於第一性原理）
    HIGH_CONFIDENCE_THRESHOLD = 0.85  # 高置信度：模組一致性 > 85%
    DISAGREEMENT_THRESHOLD = 0.60     # 分歧閾值：模組一致性 < 60%

    @staticmethod
    def calculate_module_consensus(module_scores: Dict[str, float]) -> float:
        """
        計算模組一致性分數

        原理：
        - 所有模組都說AI（>70）或都說Real（<30）→ 高一致性
        - 模組分歧（有的>70，有的<30）→ 低一致性
        """
        if not module_scores:
            return 0.0

        scores = list(module_scores.values())

        # 計算分歧度：標準差越大，分歧越大
        std_dev = float(sum((s - sum(scores)/len(scores))**2 for s in scores) ** 0.5 / len(scores))

        # 轉換為一致性分數（標準差越小，一致性越高）
        # 標準差範圍：0-50（0=完全一致，50=完全分歧）
        consensus = max(0.0, min(1.0, 1.0 - (std_dev / 50.0)))

        return consensus

    @staticmethod
    def detect_movie_anime(generation_result: GenerationAnalysisResult) -> bool:
        """
        檢測是否為電影/動畫

        特徵：
        - 高靜態比（>0.8）+ 高分辨率
        - 極低face_presence（<0.1）+ 高頻域異常
        - 檔名包含 "movie"、"anime"、"download" 等
        """
        metadata = generation_result.metadata

        # 檔名檢測
        file_name_lower = metadata.file_name.lower()
        movie_keywords = ['movie', 'anime', 'animation', 'film', '電影', '動畫']
        if any(keyword in file_name_lower for keyword in movie_keywords):
            return True

        # 特徵檢測
        if generation_result.static_ratio > 0.8 and generation_result.face_presence < 0.1:
            return True

        return False

    @classmethod
    def arbitrate(cls, generation_result: GenerationAnalysisResult) -> ArbitrationDecision:
        """
        執行資料仲裁（第三層核心邏輯）

        決策樹：
        1. 檢測電影/動畫 → Excluded Pool
        2. 計算模組一致性
        3. 高一致性 + AI → AI Pool
        4. 高一致性 + Real → Real Pool
        5. 低一致性 → Disagreement Pool（人工介入）
        """
        # Step 1: 檢測電影/動畫
        if cls.detect_movie_anime(generation_result):
            return ArbitrationDecision(
                pool_assignment=PoolType.EXCLUDED_POOL,
                module_consensus_score=0.0,
                disagreement_level=0.0,
                human_review_required=False,
                decision_rationale="檢測到電影/動畫特徵，永久排除",
                generation_result=generation_result
            )

        # Step 2: 計算模組一致性
        consensus = cls.calculate_module_consensus(generation_result.module_scores)
        disagreement = 1.0 - consensus

        # Step 3: 基於一致性分配Pool
        if consensus >= cls.HIGH_CONFIDENCE_THRESHOLD:
            # 高一致性：直接分配
            if generation_result.ai_probability > 75:
                pool = PoolType.AI_POOL
                rationale = f"高置信度AI（一致性={consensus:.2f}，AI_P={generation_result.ai_probability:.1f}）"
            elif generation_result.ai_probability < 30:
                pool = PoolType.REAL_POOL
                rationale = f"高置信度REAL（一致性={consensus:.2f}，AI_P={generation_result.ai_probability:.1f}）"
            else:
                # 一致性高但AI_P在灰色地帶
                pool = PoolType.DISAGREEMENT_POOL
                rationale = f"高一致性但AI_P在灰色地帶（{generation_result.ai_probability:.1f}）"
        else:
            # 低一致性：需要人工介入
            pool = PoolType.DISAGREEMENT_POOL
            rationale = f"模組分歧（一致性={consensus:.2f}），需要人工確認"

        return ArbitrationDecision(
            pool_assignment=pool,
            module_consensus_score=consensus,
            disagreement_level=disagreement,
            human_review_required=(pool == PoolType.DISAGREEMENT_POOL),
            decision_rationale=rationale,
            generation_result=generation_result
        )


# ==================== 第四層：經濟行為分類引擎 ====================

class EconomicBehaviorClassifier:
    """
    經濟行為分類器（獨立於生成機制）

    核心原則：
    - 僅對已確認生成機制的影片進行分類
    - 特徵空間：元數據（標題、連結、互動率）
    - 與物理特徵正交
    """

    @staticmethod
    def classify(arbitration_decision: ArbitrationDecision,
                 video_metadata: Optional[dict] = None) -> Optional[EconomicClassificationResult]:
        """
        對已確認生成機制的影片進行經濟行為分類

        Args:
            arbitration_decision: 第三層仲裁結果
            video_metadata: 可選的額外元數據（標題、描述、互動數據等）

        Returns:
            EconomicClassificationResult 或 None（Excluded Pool不分類）
        """
        # 永久排除的影片不進行經濟分類
        if arbitration_decision.pool_assignment == PoolType.EXCLUDED_POOL:
            return None

        # Disagreement Pool 暫時不分類（等待人工確認）
        if arbitration_decision.pool_assignment == PoolType.DISAGREEMENT_POOL:
            logging.info("[經濟分類] Disagreement Pool 影片暫不分類，等待人工確認")
            return None

        # 簡化實現：基於元數據分類
        # TODO: 實現完整的經濟行為檢測邏輯

        has_product_link = False
        has_shopping_cart = False
        engagement_rate = None
        text_overlay_detected = False
        commercial_keywords = []

        # 如果提供了額外元數據，進行分析
        if video_metadata:
            # TODO: 實現商業特徵檢測
            pass

        # 簡化分類邏輯（暫時返回UNKNOWN）
        economic_behavior = EconomicBehavior.UNKNOWN
        confidence = 0.5

        return EconomicClassificationResult(
            economic_behavior=economic_behavior,
            has_product_link=has_product_link,
            has_shopping_cart=has_shopping_cart,
            engagement_rate=engagement_rate,
            text_overlay_detected=text_overlay_detected,
            commercial_keywords=commercial_keywords,
            confidence=confidence,
            arbitration_decision=arbitration_decision
        )


# ==================== 四層系統協調器 ====================

class FourLayerSystemCoordinator:
    """
    四層系統總協調器

    職責：
    1. 協調第二層（生成機制分析）
    2. 執行第三層（資料仲裁）
    3. 執行第四層（經濟行為分類）
    4. 生成完整報告
    """

    VERSION = "1.0.0"

    def __init__(self, output_dir: str = "output/four_layer_reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def process_video(self,
                     generation_result: GenerationAnalysisResult,
                     video_metadata: Optional[dict] = None) -> FourLayerAnalysisReport:
        """
        處理單個視頻，執行四層分析

        Args:
            generation_result: 第二層生成機制分析結果
            video_metadata: 可選的額外元數據

        Returns:
            FourLayerAnalysisReport: 完整四層分析報告
        """
        logging.info(f"\n{'='*60}")
        logging.info(f"[四層系統] 開始處理: {generation_result.metadata.file_name}")
        logging.info(f"{'='*60}")

        # 第三層：資料仲裁
        logging.info("[第三層] 執行資料仲裁...")
        arbitration = DataArbitrationEngine.arbitrate(generation_result)
        logging.info(f"[第三層] 仲裁結果: {arbitration.pool_assignment.value}")
        logging.info(f"[第三層] 決策理由: {arbitration.decision_rationale}")

        # 第四層：經濟行為分類（僅對AI Pool和Real Pool）
        economic_classification = None
        if arbitration.pool_assignment in [PoolType.AI_POOL, PoolType.REAL_POOL]:
            logging.info("[第四層] 執行經濟行為分類...")
            economic_classification = EconomicBehaviorClassifier.classify(
                arbitration, video_metadata
            )
            if economic_classification:
                logging.info(f"[第四層] 分類結果: {economic_classification.economic_behavior.value}")

        # 生成完整報告
        report = FourLayerAnalysisReport(
            candidate_video=generation_result.metadata,
            generation_analysis=generation_result,
            arbitration=arbitration,
            economic_classification=economic_classification,
            analysis_timestamp=datetime.now().isoformat(),
            system_version=self.VERSION
        )

        # 保存報告
        report.save_to_json(self.output_dir)

        logging.info(f"[四層系統] 處理完成\n")

        return report


if __name__ == "__main__":
    # 測試代碼
    print("四層架構核心系統已加載")
    print("設計原則：正交維度分離 × 數據純度防火牆 × 長期演化能力")

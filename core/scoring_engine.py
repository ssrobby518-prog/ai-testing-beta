#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TERTIARY_TIER: Scoring Engine - 决策和聚合
FR-TSAR: 接收SECONDARY_TIER的预聚合数据，进行最终决策
FR-RAPTOR: 纯决策逻辑，不做I/O或重复计算
FR-SPARK-PLUG: 明确的函数签名和类型提示
"""

import logging
from typing import Dict, Tuple
from dataclasses import dataclass
from core.video_preprocessor import VideoFeatures

logging.basicConfig(level=logging.INFO)


@dataclass
class ScoringResult:
    """
    TERTIARY_TIER 输出：最终评分结果
    FR-RAPTOR: 自文档化的数据结构
    """
    ai_probability: float  # [0, 100] AI概率
    threat_level: str  # SAFE_ZONE, GRAY_ZONE, KILL_ZONE
    threat_action: str  # 建议的处理动作
    module_scores: Dict[str, float]  # 各模块原始分数
    weighted_scores: Dict[str, float]  # 加权后分数
    decision_rationale: str  # 决策理由（调试用）


class ScoringEngine:
    """
    TERTIARY_TIER Service: 评分引擎

    FR-RAPTOR: 单一职责 - 只做决策和聚合，不做检测
    FR-SPARK-PLUG: 纯函数逻辑，明确的决策规则
    """

    def __init__(self):
        # FR-RAPTOR: 权重配置集中管理
        self.MODULE_NAMES = [
            'metadata_extractor', 'frequency_analyzer', 'texture_noise_detector',
            'model_fingerprint_detector', 'lighting_geometry_checker', 'heartbeat_detector',
            'blink_dynamics_analyzer', 'av_sync_verifier', 'text_fingerprinting',
            'semantic_stylometry', 'sensor_noise_authenticator', 'physics_violation_detector'
        ]

        # FR-TSAR: 加載訓練後參數（若存在），以最小認知成本覆蓋配置
        self.trained = {
            'thresholds': {},
            'weights': {}
        }
        try:
            import json, os
            params_path = os.path.join('config', 'trained_params.json')
            if os.path.exists(params_path):
                with open(params_path, 'r', encoding='utf-8') as f:
                    self.trained = json.load(f)
                logging.info("[TERTIARY_TIER] Loaded trained_params.json")
        except Exception as e:
            logging.warning(f"[TERTIARY_TIER] Failed to load trained_params.json: {e}")

    def _spark_plug_calculate_weights(
        self,
        features: VideoFeatures,
        scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        FR-SPARK-PLUG: 纯函数 - 计算模块权重
        FR-TSAR: 第一性原理 - 根据可靠性动态调整权重
        """
        bitrate = features.metadata.bitrate
        base_name = features.metadata.file_path.lower()

        # 基础权重（第一性原理：根据可靠性）
        weights = {name: 1.0 for name in self.MODULE_NAMES}

        # 降低不可靠模块权重
        weights['metadata_extractor'] = 0.3
        weights['heartbeat_detector'] = 0.5
        weights['blink_dynamics_analyzer'] = 0.5
        weights['lighting_geometry_checker'] = 0.6
        weights['av_sync_verifier'] = 0.6

        # 提高可靠模块权重
        weights['frequency_analyzer'] = 1.5
        weights['texture_noise_detector'] = 1.3
        weights['model_fingerprint_detector'] = 2.2
        weights['text_fingerprinting'] = 1.4
        weights['semantic_stylometry'] = 0.8
        weights['sensor_noise_authenticator'] = 2.0
        weights['physics_violation_detector'] = 1.8

        # 應用訓練後權重覆蓋
        trained_weights = (self.trained or {}).get('weights', {})
        if isinstance(trained_weights, dict) and trained_weights:
            for name, w in trained_weights.items():
                if name in weights:
                    try:
                        weights[name] = float(w)
                    except Exception:
                        pass

        # FR-TSAR: 社交媒体视频动态调整
        is_social_media = (400000 < bitrate < 1500000) or ('download' in base_name)
        if is_social_media:
            weights['frequency_analyzer'] = 1.0
            weights['sensor_noise_authenticator'] = 1.0
            weights['physics_violation_detector'] = 1.2
            logging.info("Social media video detected, adjusted FA/SNA/PVD weights")

        return weights

    def _spark_plug_apply_first_principles(
        self,
        features: VideoFeatures,
        scores: Dict[str, float],
        weighted_scores: Dict[str, float],
        weights: Dict[str, float]
    ) -> Tuple[float, str]:
        """
        FR-SPARK-PLUG: 核心决策函数 - 应用第一性原理规则
        FR-TSAR: 级联放大 - 多层检测逻辑

        Returns:
            (ai_probability, decision_rationale)
        """
        # 提取关键指标
        mfp = scores.get('model_fingerprint_detector', 50.0)
        fa = scores.get('frequency_analyzer', 50.0)
        tn = scores.get('texture_noise_detector', 50.0)
        tf = scores.get('text_fingerprinting', 50.0)
        sna = scores.get('sensor_noise_authenticator', 50.0)
        pvd = scores.get('physics_violation_detector', 50.0)

        bitrate = features.metadata.bitrate
        face_presence = features.face_presence
        static_ratio = features.static_ratio
        base_name = features.metadata.file_path.lower()

        is_phone_video = 800000 < bitrate < 1800000
        is_social_media = (400000 < bitrate < 1500000) or ('download' in base_name)

        # FR-TSAR: 第一性原理 - 绝对判定优先

        thr = (self.trained or {}).get('thresholds', {})
        mfp_abs_ai = float(thr.get('mfp_abs_ai', 88.0))
        # === 阶段1: MFP绝对判定 ===
        if mfp >= mfp_abs_ai:
            return 98.0, f"ABSOLUTE AI: MFP={mfp:.1f} (绝对AI特征)"

        # 真实视频保护条件
        smartphone_real_dance = (
            is_phone_video and fa >= 65 and mfp <= 40 and
            tf <= 15 and static_ratio < 0.3 and tn <= 20
        )
        smartphone_nightclub_real = (
            is_phone_video and mfp <= 12 and tf <= 35 and
            face_presence < 0.90 and static_ratio < 0.15
        )
        tiktok_reedit_real = (
            is_social_media and mfp <= 30 and 15 <= tf <= 50 and
            face_presence < 0.85 and static_ratio < 0.2
        )
        real_guard = smartphone_real_dance or smartphone_nightclub_real or tiktok_reedit_real

        if mfp <= 15:
            # MFP极低 - 可能是真实
            has_strong_ai_signal = False

            # 检测AI信号
            if fa >= 90:
                has_strong_ai_signal = True
            if tf >= 70 and face_presence >= 0.9:
                has_strong_ai_signal = True
            if face_presence >= 0.98:
                has_strong_ai_signal = True

            if has_strong_ai_signal and not real_guard:
                # 覆盖MFP判定
                adjusted_weights = weights.copy()
                adjusted_weights['model_fingerprint_detector'] = 1.0
                adjusted_weighted_scores = {
                    name: scores[name] * adjusted_weights[name]
                    for name in self.MODULE_NAMES
                }
                ai_p = sum(adjusted_weighted_scores.values()) / sum(adjusted_weights.values())

                if fa >= 95:
                    ai_p = max(ai_p, 85.0)

                # 社交媒体保护
                if is_social_media and mfp <= 10 and face_presence < 0.98:
                    ai_p = min(ai_p, 55.0)

                return ai_p, f"MFP覆盖: 其他模块显示强AI信号 (FA={fa:.1f}, face={face_presence:.2f})"
            else:
                return 8.0, f"ABSOLUTE REAL: MFP={mfp:.1f} (绝对真实)"

        # === 阶段2: 加权平均 + 规则调整 ===
        ai_p = sum(weighted_scores.values()) / sum(weights.values())
        rationale = "加权平均计算"

        # TikTok平台政策
        beautify_filter_ai = (
            (is_phone_video or is_social_media) and
            face_presence >= float(thr.get('beautify_face_presence', 0.50)) and
            tn <= float(thr.get('beautify_tn', 20.0)) and
            tf <= float(thr.get('beautify_tf', 35.0)) and
            static_ratio < float(thr.get('beautify_static', 0.20)) and
            ((mfp >= float(thr.get('beautify_mfp_or_pvd', 70.0))) or (pvd >= float(thr.get('beautify_mfp_or_pvd', 70.0))))
        )
        if beautify_filter_ai:
            ai_p = max(ai_p, 85.0)
            rationale += " | 平台政策: 美颜/换脸"

        # 强AI模式检测
        if sna >= float(thr.get('sna_high', 75.0)) and not real_guard:
            boost = 15.0 if is_social_media else 30.0
            ai_p += boost
            rationale += f" | 传感器噪声异常 (+{boost})"

        if pvd >= float(thr.get('pvd_high', 75.0)) and not real_guard:
            boost = 14.0 if is_social_media else 28.0
            ai_p += boost
            rationale += f" | 物理违规 (+{boost})"

        # 真实保护
        if is_phone_video and mfp <= 40 and sna <= 30:
            ai_p -= 20.0
            rationale += " | 真实手机拍摄保护"

        # === 阶段3: 社交媒体保护 ===
        physical_avg = (sna + pvd) / 2.0
        critical_ai = (mfp >= 95 or (pvd >= 90 and mfp >= 85))

        # 静态POV保护
        if (is_social_media and static_ratio >= 0.85 and
            face_presence < 0.3 and not beautify_filter_ai):
            if physical_avg >= 80:
                ai_p = min(ai_p, 25.0)
                rationale += " | 静态POV真实保护"

        # 动态场景保护
        elif (is_social_media and static_ratio < 0.10 and
              (mfp >= 70 or face_presence >= 0.7) and
              not beautify_filter_ai and not critical_ai):
            if physical_avg >= 85:
                ai_p = min(ai_p, 30.0)
                rationale += " | 动态场景真实保护"

        # 限制范围
        ai_p = max(0.0, min(100.0, ai_p))

        return ai_p, rationale

    def calculate_score(
        self,
        features: VideoFeatures,
        module_scores: Dict[str, float]
    ) -> ScoringResult:
        """
        TERTIARY_TIER 主入口：计算最终评分

        FR-RAPTOR: 明确的数据流 - SECONDARY -> TERTIARY
        FR-SPARK-PLUG: 纯函数决策，无副作用

        Args:
            features: PRIMARY_TIER输出
            module_scores: SECONDARY_TIER输出

        Returns:
            ScoringResult: 最终评分结果
        """
        logging.info("[TERTIARY_TIER] Calculating final score...")

        # 1. 计算权重
        weights = self._spark_plug_calculate_weights(features, module_scores)

        # 2. 计算加权分数
        weighted_scores = {
            name: module_scores.get(name, 50.0) * weights.get(name, 1.0)
            for name in self.MODULE_NAMES
        }

        # 3. 应用第一性原理决策
        ai_probability, rationale = self._spark_plug_apply_first_principles(
            features, module_scores, weighted_scores, weights
        )

        # 4. 确定威胁等级
        if ai_probability <= 20:
            threat_level = "SAFE_ZONE"
            threat_action = "PASS - Video cleared for distribution"
        elif ai_probability <= 60:
            threat_level = "GRAY_ZONE"
            threat_action = "FLAGGED - Shadowban/Manual review recommended"
        else:
            threat_level = "KILL_ZONE"
            threat_action = "BLOCKED - Zero playback / Hard limit"

        logging.info(f"[TERTIARY_TIER] Final AI_P={ai_probability:.2f}, Level={threat_level}")
        logging.info(f"[TERTIARY_TIER] Rationale: {rationale}")

        return ScoringResult(
            ai_probability=ai_probability,
            threat_level=threat_level,
            threat_action=threat_action,
            module_scores=module_scores,
            weighted_scores=weighted_scores,
            decision_rationale=rationale
        )

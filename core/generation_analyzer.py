#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第二層：生成機制分析層 (Generation Mechanism Analysis Layer)
==========================================================

職責：
- 調用所有物理檢測模組（頻域、物理違規、傳感器噪聲等）
- 輸出標準化的 GenerationAnalysisResult
- **僅**判定生成機制（AI/Real/Uncertain/Movie-Anime）
- **不**進行商業行為分類（粉片/貨片）

設計原則：
- 第一性原理：物理特徵不可偽造
- 輸出純淨的生成機制標籤（不受商業特徵污染）
- 為第三層提供可靠的物理特徵向量

Author: Claude Sonnet 4.5
Date: 2025-12-20
"""

import os
import logging
import importlib.util
from typing import Dict, List
import cv2
from pymediainfo import MediaInfo

from core.four_layer_system import (
    GenerationAnalysisResult,
    GenerationMechanism,
    VideoMetadata
)

logging.basicConfig(level=logging.INFO)


class GenerationMechanismAnalyzer:
    """
    生成機制分析器（第二層核心）

    核心原則：
    1. 只判定「片怎麼做的」（AI vs Real vs Uncertain vs Movie-Anime）
    2. 不判定「片拿來幹嘛」（粉片 vs 貨片）
    3. 輸出物理特徵向量供第三層仲裁使用
    """

    # 模組列表（物理檢測模組）
    MODULE_NAMES = [
        'frequency_analyzer',
        'texture_noise_detector',
        'model_fingerprint_detector',
        'lighting_geometry_checker',
        'heartbeat_detector',
        'blink_dynamics_analyzer',
        'text_fingerprinting',
        'sensor_noise_authenticator',
        'physics_violation_detector'
    ]

    def __init__(self):
        """初始化分析器，加載所有模組"""
        self.modules = self._load_modules()

    def _load_module(self, module_name: str):
        """動態加載單個模組"""
        try:
            spec = importlib.util.spec_from_file_location(
                module_name,
                f'modules/{module_name}.py'
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            logging.info(f"[第二層] 已加載模組: {module_name}")
            return mod
        except Exception as e:
            logging.error(f"[第二層] 加載模組失敗 {module_name}: {e}")
            raise

    def _load_modules(self) -> List:
        """加載所有檢測模組"""
        logging.info("[第二層] 開始加載物理檢測模組...")
        modules = [self._load_module(name) for name in self.MODULE_NAMES]
        logging.info(f"[第二層] 成功加載 {len(modules)} 個模組")
        return modules

    def _extract_video_metadata(self, file_path: str) -> VideoMetadata:
        """提取視頻元數據"""
        try:
            media_info = MediaInfo.parse(file_path)
            bitrate = 0
            fps = 0.0
            duration = 0.0
            resolution = "unknown"

            for track in media_info.tracks:
                if track.track_type == 'Video':
                    bitrate = track.bit_rate if track.bit_rate else 0
                    fps = float(track.frame_rate) if track.frame_rate else 0.0
                    duration = float(track.duration) / 1000.0 if track.duration else 0.0
                    resolution = f"{track.width}x{track.height}" if track.width and track.height else "unknown"
                    break

            return VideoMetadata(
                file_path=file_path,
                file_name=os.path.basename(file_path),
                bitrate=bitrate,
                fps=fps,
                duration=duration,
                resolution=resolution,
                download_source="local",  # TODO: 從檔名或路徑推斷來源
                timestamp=""
            )

        except Exception as e:
            logging.error(f"[第二層] 提取元數據失敗: {e}")
            # 返回默認值
            return VideoMetadata(
                file_path=file_path,
                file_name=os.path.basename(file_path),
                bitrate=0,
                fps=0.0,
                duration=0.0,
                resolution="unknown",
                download_source="local",
                timestamp=""
            )

    def _calculate_auxiliary_features(self, file_path: str, bitrate: int) -> Dict:
        """計算輔助特徵（face_presence, static_ratio）"""
        try:
            # Face presence
            cap_fp = cv2.VideoCapture(file_path)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            hits = 0
            cnt = 0

            while cap_fp.isOpened() and cnt < 30:
                ret, frame = cap_fp.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.2, 5)
                if len(faces) > 0:
                    hits += 1
                cnt += 1

            cap_fp.release()
            face_presence = hits / max(cnt, 1)

            # Static ratio
            cap_sv = cv2.VideoCapture(file_path)
            prev = None
            diffs = []
            k = 0

            while cap_sv.isOpened() and k < 40:
                ret, frame = cap_sv.read()
                if not ret:
                    break
                g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h, w = g.shape
                scale = 160.0 / max(w, 1)
                g = cv2.resize(g, (int(w * scale), int(h * scale)))

                if prev is not None:
                    d = cv2.absdiff(g, prev)
                    diffs.append(float(d.mean()))

                prev = g
                k += 1

            cap_sv.release()
            static_ratio = float(sum(1.0 for d in diffs if d < 1.5)) / max(len(diffs), 1)

            # is_phone_video
            is_phone_video = 800000 < bitrate < 1800000

            return {
                'face_presence': face_presence,
                'static_ratio': static_ratio,
                'is_phone_video': is_phone_video
            }

        except Exception as e:
            logging.error(f"[第二層] 計算輔助特徵失敗: {e}")
            return {
                'face_presence': 0.0,
                'static_ratio': 0.0,
                'is_phone_video': False
            }

    def _run_all_modules(self, file_path: str) -> Dict[str, float]:
        """運行所有物理檢測模組"""
        scores = {}

        for name, mod in zip(self.MODULE_NAMES, self.modules):
            try:
                score = mod.detect(file_path)
                scores[name] = score
                logging.info(f"[第二層] {name}: {score:.1f}")
            except Exception as e:
                logging.error(f"[第二層] 模組 {name} 執行失敗: {e}")
                scores[name] = 50.0  # 失敗時返回中性分

        return scores

    def _apply_weights(self, scores: Dict[str, float], bitrate: int) -> Dict[str, float]:
        """
        應用模組權重（基於第一性原理優化）

        原理：
        - 降低不可靠模組的權重（AI可以模擬的特徵）
        - 提高可靠模組的權重（物理守恆定律）
        - 低bitrate保護：降低壓縮敏感模組的權重
        """
        # 基礎權重（基於訓練數據優化）
        weights = {
            'frequency_analyzer': 1.3,
            'texture_noise_detector': 1.3,
            'model_fingerprint_detector': 0.9,  # 主要誤報源
            'lighting_geometry_checker': 0.6,
            'heartbeat_detector': 0.465,
            'blink_dynamics_analyzer': 0.5,
            'text_fingerprinting': 1.4,
            'sensor_noise_authenticator': 1.96,
            'physics_violation_detector': 1.8
        }

        # 低bitrate保護
        bitrate_mbps = bitrate / 1_000_000.0
        is_social_media = (bitrate > 0 and bitrate < 2_000_000)

        if bitrate > 0:
            if bitrate < 800_000:  # <0.8 Mbps - 極低bitrate
                weights['model_fingerprint_detector'] *= 0.5
                weights['frequency_analyzer'] *= 0.4
                weights['sensor_noise_authenticator'] *= 0.6
                weights['physics_violation_detector'] *= 0.7
                logging.info(f"[第二層] 極低bitrate保護激活 ({bitrate_mbps:.2f} Mbps)")
            elif bitrate < 1_500_000:  # 0.8-1.5 Mbps - 低bitrate
                weights['model_fingerprint_detector'] *= 0.75
                weights['frequency_analyzer'] *= 0.65
                weights['sensor_noise_authenticator'] *= 0.8
                weights['physics_violation_detector'] *= 0.85
                logging.info(f"[第二層] 低bitrate保護激活 ({bitrate_mbps:.2f} Mbps)")

        # 計算加權分數
        weighted_scores = {name: scores[name] * weights[name] for name in scores}

        return weighted_scores

    def _calculate_ai_probability(self,
                                  scores: Dict[str, float],
                                  weighted_scores: Dict[str, float],
                                  aux_features: Dict) -> float:
        """
        計算AI概率（基於第一性原理）

        原理：使用autotesting.py中優化過的判定邏輯
        這裡簡化實現，使用加權平均
        TODO: 可以完整遷移autotesting.py的複雜判定邏輯
        """
        # 簡化版：加權平均
        weights_sum = sum(w for w in weighted_scores.values())
        if weights_sum == 0:
            return 50.0

        ai_p = sum(weighted_scores.values()) / weights_sum

        # 邊界限制
        ai_p = max(0.0, min(100.0, ai_p))

        logging.info(f"[第二層] AI概率: {ai_p:.1f}%")

        return ai_p

    def _determine_generation_mechanism(self, ai_probability: float) -> GenerationMechanism:
        """
        判定生成機制

        原則：
        - AI_P > 75 → AI
        - AI_P < 30 → REAL
        - 30 <= AI_P <= 75 → UNCERTAIN
        - 電影/動畫特徵 → MOVIE_ANIME（由第三層處理）
        """
        if ai_probability > 75:
            return GenerationMechanism.AI
        elif ai_probability < 30:
            return GenerationMechanism.REAL
        else:
            return GenerationMechanism.UNCERTAIN

    def _calculate_confidence(self, scores: Dict[str, float]) -> float:
        """
        計算置信度（基於模組一致性）

        原理：
        - 所有模組分數接近 → 高置信度
        - 模組分數分散 → 低置信度
        """
        if not scores:
            return 0.0

        score_list = list(scores.values())
        mean_score = sum(score_list) / len(score_list)

        # 計算標準差
        variance = sum((s - mean_score) ** 2 for s in score_list) / len(score_list)
        std_dev = variance ** 0.5

        # 轉換為置信度（標準差越小，置信度越高）
        # 標準差範圍：0-50（0=完全一致，50=完全分散）
        confidence = max(0.0, min(1.0, 1.0 - (std_dev / 50.0)))

        return confidence

    def analyze(self, file_path: str) -> GenerationAnalysisResult:
        """
        分析單個視頻的生成機制（第二層主入口）

        Args:
            file_path: 視頻文件路徑

        Returns:
            GenerationAnalysisResult: 標準化的生成機制分析結果
        """
        logging.info(f"\n{'='*60}")
        logging.info(f"[第二層] 開始分析: {os.path.basename(file_path)}")
        logging.info(f"{'='*60}")

        # Step 1: 提取元數據
        metadata = self._extract_video_metadata(file_path)
        logging.info(f"[第二層] Bitrate: {metadata.bitrate / 1_000_000:.2f} Mbps")

        # Step 2: 計算輔助特徵
        aux_features = self._calculate_auxiliary_features(file_path, metadata.bitrate)
        logging.info(f"[第二層] Face Presence: {aux_features['face_presence']:.2f}")
        logging.info(f"[第二層] Static Ratio: {aux_features['static_ratio']:.2f}")

        # Step 3: 運行所有物理檢測模組
        scores = self._run_all_modules(file_path)

        # Step 4: 應用權重
        weighted_scores = self._apply_weights(scores, metadata.bitrate)

        # Step 5: 計算AI概率
        ai_probability = self._calculate_ai_probability(scores, weighted_scores, aux_features)

        # Step 6: 判定生成機制
        generation_mechanism = self._determine_generation_mechanism(ai_probability)
        logging.info(f"[第二層] 生成機制: {generation_mechanism.value}")

        # Step 7: 計算置信度
        confidence = self._calculate_confidence(scores)
        logging.info(f"[第二層] 置信度: {confidence:.2f}")

        # 構建結果
        result = GenerationAnalysisResult(
            generation_mechanism=generation_mechanism,
            ai_probability=ai_probability,
            module_scores=scores,
            weighted_scores=weighted_scores,
            face_presence=aux_features['face_presence'],
            static_ratio=aux_features['static_ratio'],
            is_phone_video=aux_features['is_phone_video'],
            metadata=metadata,
            confidence=confidence
        )

        logging.info(f"[第二層] 分析完成\n")

        return result


if __name__ == "__main__":
    # 測試代碼
    analyzer = GenerationMechanismAnalyzer()
    print("第二層：生成機制分析器已加載")
    print("職責：僅判定生成機制（AI/Real/Uncertain/Movie-Anime）")
    print("不涉及商業行為分類（粉片/貨片）")

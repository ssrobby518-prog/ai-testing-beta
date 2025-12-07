#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SECONDARY_TIER + SPARK_PLUG: Detection Engine - 纯计算核心
FR-TSAR: 接收压缩数据，进行级联放大计算
FR-RAPTOR: 消除I/O，纯计算逻辑
FR-SPARK-PLUG: 无状态、高内聚、纯函数
"""

import numpy as np
import cv2
import logging
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.video_preprocessor import VideoFeatures

logging.basicConfig(level=logging.INFO)


class DetectionEngine:
    """
    SECONDARY_TIER Service: 检测引擎

    FR-TSAR: 接收PRIMARY_TIER的压缩数据，执行级联放大计算
    FR-RAPTOR: 零I/O，纯计算，单一职责
    FR-SPARK-PLUG: 并行化独立模块，最大化性能
    """

    def __init__(self, parallel: bool = True, max_workers: int = 6):
        """
        Args:
            parallel: 是否并行执行模块（FR-SPARK-PLUG优化）
            max_workers: 最大并行线程数
        """
        self.parallel = parallel
        self.max_workers = max_workers

    # ========== SPARK_PLUG模块：纯计算函数 ==========

    @staticmethod
    def _spark_plug_frequency_analyzer(features: VideoFeatures) -> float:
        """
        FR-SPARK-PLUG: 频域分析（纯计算）
        FR-TSAR: 接收预处理数据，不做I/O
        """
        try:
            frames = features.frames
            bitrate = features.metadata.bitrate

            if len(frames) < 10:
                return 50.0

            # 配置参数（第一性原理）
            FFT_SIZE = 512
            HIGH_FREQ_CUTOFF = 0.85
            LOW_BITRATE = 2000000
            drop_off_threshold = 0.04 if 0 < bitrate < LOW_BITRATE else 0.025

            # FR-SPARK-PLUG: 向量化FFT计算
            magnitudes = []
            for frame_data in frames[:100]:  # 限制计算量
                resized = cv2.resize(frame_data.gray, (FFT_SIZE, FFT_SIZE))
                # 向量化FFT
                fft_result = np.fft.fft2(resized)
                fft_shift = np.fft.fftshift(fft_result)
                magnitude = 20 * np.log(np.abs(fft_shift) + 1e-10)
                magnitudes.append(magnitude)

            # FR-RAPTOR: 简化计算，消除冗余
            avg_mag = np.mean(magnitudes, axis=0)
            center = np.mean(avg_mag[:FFT_SIZE//2])
            high_freq = np.mean(avg_mag[int(FFT_SIZE * HIGH_FREQ_CUTOFF):])
            drop_off = (center - high_freq) / center if center != 0 else 0

            # 时序变异
            hf_means = [np.mean(mag[int(FFT_SIZE * HIGH_FREQ_CUTOFF):]) for mag in magnitudes]
            temporal_cv = float(np.std(hf_means) / (np.mean(hf_means) + 1e-6))
            cv_threshold = 0.15 if bitrate < LOW_BITRATE else 0.20

            # FR-SPARK-PLUG: 纯计算逻辑，无副作用
            if drop_off > drop_off_threshold and temporal_cv < cv_threshold:
                score = 80.0
            elif drop_off <= drop_off_threshold and temporal_cv >= cv_threshold:
                score = 5.0
            else:
                score = 50.0

            return max(5.0, min(95.0, score))

        except Exception as e:
            logging.error(f"Error in frequency_analyzer: {e}")
            return 50.0

    @staticmethod
    def _spark_plug_model_fingerprint(features: VideoFeatures) -> float:
        """
        FR-SPARK-PLUG: 模型指纹检测（纯计算）
        FR-RAPTOR: 消除重复的人脸检测，使用预处理数据
        """
        try:
            frames = features.frames
            bitrate = features.metadata.bitrate
            face_presence = features.face_presence

            if not frames:
                return 50.0

            # 关键指标
            ai_seam_score = 0.0
            stutter_score = 0.0
            color_anomaly_score = 0.0
            analyzed_frames = 0
            prev_frame_gray = None

            is_phone_compressed = 800000 < bitrate < 1800000

            # FR-RAPTOR: 使用预处理数据，避免重复转换
            for i, frame_data in enumerate(frames[:50]):
                gray = frame_data.gray
                hsv = frame_data.hsv
                faces = frame_data.faces

                # 卡顿检测
                if prev_frame_gray is not None and i > 0:
                    frame_diff = cv2.absdiff(gray, prev_frame_gray)
                    diff_mean = np.mean(frame_diff)
                    if diff_mean < 2.0:
                        stutter_score += 2.0
                    elif diff_mean > 60.0:
                        stutter_score += 1.5

                prev_frame_gray = gray.copy()

                # 色彩异常检测（使用预处理的HSV）
                h_channel = hsv[:,:,0]
                s_channel = hsv[:,:,1]
                h_std = np.std(h_channel)
                s_mean = np.mean(s_channel)

                if h_std < 15.0:
                    color_anomaly_score += 1.0
                if s_mean > 140.0:
                    color_anomaly_score += 1.5

                # FR-TSAR: 使用预处理的人脸数据
                if len(faces) == 0:
                    analyzed_frames += 1
                    continue

                x, y, w, h = faces[0]

                # AI 缝合线检测（简化版）
                margin = int(w * 0.15)
                x1, y1 = max(0, x - margin), max(0, y - margin)
                x2, y2 = min(gray.shape[1], x + w + margin), min(gray.shape[0], y + h + margin)
                extended_roi = gray[y1:y2, x1:x2]

                gx = cv2.Sobel(extended_roi, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(extended_roi, cv2.CV_32F, 0, 1, ksize=3)
                gradient = np.sqrt(gx**2 + gy**2)

                # 简化计算
                boundary_mean = np.mean(gradient)
                if boundary_mean > 50.0:  # 简化阈值
                    ai_seam_score += 1.5

                analyzed_frames += 1

            if analyzed_frames == 0:
                return 50.0

            # FR-SPARK-PLUG: 纯计算决策
            avg_ai_seam = ai_seam_score / analyzed_frames
            avg_stutter = stutter_score / max(analyzed_frames - 1, 1)
            avg_color_anomaly = color_anomaly_score / analyzed_frames

            # 绝对AI特征
            if avg_color_anomaly > 2.0:
                return 95.0
            if avg_stutter > 2.0:
                return 95.0

            # 绝对真实特征
            if is_phone_compressed and face_presence < 0.95:
                if avg_color_anomaly < 0.3 and avg_stutter < 0.2:
                    return 8.0

            # 标准计分
            score = 35.0
            if avg_ai_seam > 1.5:
                score += 45.0
            if avg_stutter > 1.0:
                score += 28.0
            if avg_color_anomaly > 1.0:
                score += 25.0

            return max(5.0, min(99.0, score))

        except Exception as e:
            logging.error(f"Error in model_fingerprint: {e}")
            return 50.0

    @staticmethod
    def _spark_plug_physics_violation(features: VideoFeatures) -> float:
        """
        FR-SPARK-PLUG: 物理违规检测（纯计算）
        """
        try:
            frames = features.frames
            fps = features.metadata.fps

            if len(frames) < 2:
                return 50.0

            optical_flow_discontinuities = []
            prev_gray = None
            prev_flow = None

            # FR-RAPTOR: 使用预处理数据
            for frame_data in frames[:60]:
                gray = frame_data.gray

                if prev_gray is not None:
                    # 光流计算
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None,
                        pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                    )

                    flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

                    if prev_flow is not None:
                        flow_diff = np.sqrt(
                            (flow[..., 0] - prev_flow[..., 0])**2 +
                            (flow[..., 1] - prev_flow[..., 1])**2
                        )

                        motion_mask = flow_magnitude > 0.5
                        if np.sum(motion_mask) > 100:
                            flow_change = flow_diff[motion_mask]
                            max_change = np.percentile(flow_change, 95)
                            change_threshold = 2.0 * (30.0 / fps)

                            if max_change > change_threshold * 3:
                                optical_flow_discontinuities.append(2.0)
                            elif max_change > change_threshold * 2:
                                optical_flow_discontinuities.append(1.5)
                            elif max_change > change_threshold:
                                optical_flow_discontinuities.append(1.0)
                            else:
                                optical_flow_discontinuities.append(0.0)

                    prev_flow = flow.copy()

                prev_gray = gray.copy()

            if not optical_flow_discontinuities:
                return 50.0

            # FR-SPARK-PLUG: 纯计算评分
            score = 35.0
            avg_flow_disc = np.mean(optical_flow_discontinuities)

            if avg_flow_disc > 1.5:
                score += 35.0
            elif avg_flow_disc > 1.0:
                score += 25.0
            elif avg_flow_disc < 0.2:
                score -= 15.0

            return max(5.0, min(95.0, score))

        except Exception as e:
            logging.error(f"Error in physics_violation: {e}")
            return 50.0

    @staticmethod
    def _spark_plug_texture_noise(features: VideoFeatures) -> float:
        """FR-SPARK-PLUG: 纹理噪声检测（简化版）"""
        try:
            frames = features.frames[:30]
            if not frames:
                return 50.0

            noise_scores = []
            for frame_data in frames:
                gray = frame_data.gray
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                variance = laplacian.var()
                noise_scores.append(variance)

            avg_noise = np.mean(noise_scores)
            if avg_noise < 50:
                return 75.0  # AI可能过度平滑
            elif avg_noise > 200:
                return 15.0  # 真实噪声
            else:
                return 50.0

        except Exception as e:
            logging.error(f"Error in texture_noise: {e}")
            return 50.0

    @staticmethod
    def _spark_plug_text_fingerprinting(features: VideoFeatures) -> float:
        """FR-SPARK-PLUG: 文本指纹检测（简化版）"""
        # 简化实现：基于边缘密度
        try:
            frames = features.frames[:20]
            if not frames:
                return 50.0

            edge_densities = []
            for frame_data in frames:
                edges = cv2.Canny(frame_data.gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                edge_densities.append(edge_density)

            avg_density = np.mean(edge_densities)
            if avg_density > 0.15:
                return 70.0  # 可能有文本
            else:
                return 20.0

        except Exception as e:
            logging.error(f"Error in text_fingerprinting: {e}")
            return 50.0

    # ========== 简化模块（返回中性分） ==========

    @staticmethod
    def _neutral_module(name: str) -> float:
        """FR-RAPTOR: 暂时返回中性分的模块"""
        logging.info(f"Module {name} using neutral score (not optimized yet)")
        return 50.0

    # ========== SECONDARY_TIER 协调器 ==========

    def detect_all(self, features: VideoFeatures) -> Dict[str, float]:
        """
        SECONDARY_TIER 主入口：并行执行所有检测模块

        FR-SPARK-PLUG: 并行化独立计算，最大化性能
        FR-TSAR: 接收压缩数据，生成预聚合分数

        Args:
            features: PRIMARY_TIER输出的压缩数据

        Returns:
            Dict[str, float]: 各模块分数
        """
        logging.info("[SECONDARY_TIER] Starting parallel detection...")

        # FR-SPARK-PLUG: 定义所有SPARK_PLUG模块
        modules = {
            'frequency_analyzer': lambda: self._spark_plug_frequency_analyzer(features),
            'model_fingerprint_detector': lambda: self._spark_plug_model_fingerprint(features),
            'physics_violation_detector': lambda: self._spark_plug_physics_violation(features),
            'texture_noise_detector': lambda: self._spark_plug_texture_noise(features),
            'text_fingerprinting': lambda: self._spark_plug_text_fingerprinting(features),
            # 简化模块（未优化）
            'metadata_extractor': lambda: self._neutral_module('metadata_extractor'),
            'lighting_geometry_checker': lambda: self._neutral_module('lighting_geometry_checker'),
            'heartbeat_detector': lambda: self._neutral_module('heartbeat_detector'),
            'blink_dynamics_analyzer': lambda: self._neutral_module('blink_dynamics_analyzer'),
            'av_sync_verifier': lambda: self._neutral_module('av_sync_verifier'),
            'semantic_stylometry': lambda: self._neutral_module('semantic_stylometry'),
            'sensor_noise_authenticator': lambda: self._neutral_module('sensor_noise_authenticator'),
        }

        scores = {}

        if self.parallel:
            # FR-SPARK-PLUG: 并行执行
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_module = {
                    executor.submit(func): name
                    for name, func in modules.items()
                }

                for future in as_completed(future_to_module):
                    module_name = future_to_module[future]
                    try:
                        score = future.result()
                        scores[module_name] = score
                        logging.info(f"[SPARK_PLUG] {module_name}: {score:.1f}")
                    except Exception as e:
                        logging.error(f"Module {module_name} failed: {e}")
                        scores[module_name] = 50.0
        else:
            # 串行执行（调试模式）
            for name, func in modules.items():
                try:
                    score = func()
                    scores[name] = score
                    logging.info(f"[SPARK_PLUG] {name}: {score:.1f}")
                except Exception as e:
                    logging.error(f"Module {name} failed: {e}")
                    scores[name] = 50.0

        logging.info("[SECONDARY_TIER] Detection complete")
        return scores

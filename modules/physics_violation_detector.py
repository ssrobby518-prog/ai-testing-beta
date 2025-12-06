#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模組 Physics Violation Detector: 物理規律違反檢測。
第一性原理：真實世界遵守物理定律（運動連續性、因果關係、光學一致性）。
AI生成視頻可能違反物理規律，產生不自然的運動模式或光學異常。
"""

import logging
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)

def detect(file_path):
    """
    第一性原理：物理規律檢測

    真實視頻特徵：
    1. 光流連續（運動平滑，無突變）
    2. 加速度合理（符合慣性定律）
    3. 遮擋關係一致（前景遮擋後景）
    4. 光源與陰影一致（光學物理）

    AI生成視頻缺陷：
    1. 光流突變（幀間運動不連續）
    2. 加速度異常（違反慣性）
    3. 遮擋關係混亂（深度估計錯誤）
    4. 光照方向不一致
    """
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return 50.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # 默認值

        # 關鍵指標
        optical_flow_discontinuities = []  # 光流不連續
        acceleration_anomalies = []  # 加速度異常
        motion_blur_consistency = []  # 運動模糊一致性
        edge_stability_scores = []  # 邊緣穩定性

        prev_gray = None
        prev_flow = None
        prev_velocity = None

        sample_frames = min(60, total_frames)  # 增加採樣以更好檢測運動

        for i in range(sample_frames):
            if total_frames > 0:
                frame_pos = int(i * total_frames / sample_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            if prev_gray is not None:
                # === 1. 光流連續性檢測 ===
                # 第一性原理：真實世界運動是連續的，光流應該平滑
                # AI生成：可能有幀間突變（生成模型的時間一致性問題）

                # 計算稠密光流（Farneback算法）
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0
                )

                # 光流大小（運動速度）
                flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

                # 檢測光流突變
                if prev_flow is not None:
                    # 計算光流的時間變化（加速度）
                    flow_diff = np.sqrt(
                        (flow[..., 0] - prev_flow[..., 0])**2 +
                        (flow[..., 1] - prev_flow[..., 1])**2
                    )

                    # 只考慮有運動的區域
                    motion_mask = (flow_magnitude > 0.5) | (np.sqrt(prev_flow[..., 0]**2 + prev_flow[..., 1]**2) > 0.5)

                    if np.sum(motion_mask) > 100:
                        # 計算運動區域的光流變化
                        flow_change = flow_diff[motion_mask]
                        avg_change = np.mean(flow_change)
                        max_change = np.percentile(flow_change, 95)  # 95百分位

                        # 真實視頻：光流變化平滑（加速度合理）
                        # AI視頻：可能有突變（幀間不連續）
                        # 根據FPS調整閾值
                        change_threshold = 2.0 * (30.0 / fps)  # FPS越低，允許的變化越大

                        if max_change > change_threshold * 3:  # 嚴重突變
                            optical_flow_discontinuities.append(2.0)
                        elif max_change > change_threshold * 2:
                            optical_flow_discontinuities.append(1.5)
                        elif max_change > change_threshold:
                            optical_flow_discontinuities.append(1.0)
                        else:
                            optical_flow_discontinuities.append(0.0)

                # === 2. 運動加速度異常檢測 ===
                # 第一性原理：物體運動遵循慣性定律，加速度不應劇烈變化
                # AI生成：可能有不自然的加速/減速

                # 計算平均速度
                if np.sum(flow_magnitude > 0.5) > 100:
                    current_velocity = np.mean(flow_magnitude[flow_magnitude > 0.5])

                    if prev_velocity is not None:
                        # 加速度 = 速度變化
                        acceleration = abs(current_velocity - prev_velocity) * fps

                        # 真實視頻：加速度合理（< 100 pixels/s²）
                        # AI視頻：可能有異常加速度
                        if acceleration > 150:
                            acceleration_anomalies.append(2.0)
                        elif acceleration > 80:
                            acceleration_anomalies.append(1.0)
                        elif acceleration > 40:
                            acceleration_anomalies.append(0.5)
                        else:
                            acceleration_anomalies.append(0.0)

                    prev_velocity = current_velocity

                # === 3. 運動模糊一致性 ===
                # 第一性原理：快速運動應該產生運動模糊（相機曝光時間內的軌跡積分）
                # AI生成：可能缺少運動模糊或模糊方向不一致

                # 檢測邊緣銳度
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (h * w)

                # 在有運動的區域，檢查邊緣銳度
                if np.sum(flow_magnitude > 2.0) > 500:  # 有顯著運動
                    high_motion_mask = flow_magnitude > 2.0
                    edge_in_motion = edges[high_motion_mask]

                    if len(edge_in_motion) > 0:
                        edge_sharpness = np.sum(edge_in_motion > 0) / len(edge_in_motion)

                        # 真實視頻：快速運動區域邊緣應該模糊（低銳度）
                        # AI視頻：可能邊緣過於銳利（缺少運動模糊）
                        if edge_sharpness > 0.15:  # 運動區域邊緣過於銳利
                            motion_blur_consistency.append(1.5)
                        elif edge_sharpness > 0.10:
                            motion_blur_consistency.append(1.0)
                        else:
                            motion_blur_consistency.append(0.0)

                # === 4. 邊緣穩定性（抗鋸齒一致性）===
                # 第一性原理：真實相機有光學抗鋸齒（lens MTF）
                # AI生成：邊緣可能有數字化鋸齒或過度銳化

                # 計算邊緣梯度的方差
                gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(gx**2 + gy**2)

                # 在邊緣區域計算梯度統計
                edge_mask = gradient_magnitude > np.percentile(gradient_magnitude, 90)
                if np.sum(edge_mask) > 100:
                    edge_gradients = gradient_magnitude[edge_mask]
                    gradient_std = np.std(edge_gradients)
                    gradient_mean = np.mean(edge_gradients)

                    # 真實視頻：邊緣梯度變化平滑（低方差）
                    # AI視頻：邊緣可能有不自然的銳化（高方差）
                    if gradient_mean > 0:
                        cv_gradient = gradient_std / (gradient_mean + 1e-6)
                        if cv_gradient > 0.8:  # 變異係數過高
                            edge_stability_scores.append(1.5)
                        elif cv_gradient > 0.6:
                            edge_stability_scores.append(1.0)
                        else:
                            edge_stability_scores.append(0.0)

                prev_flow = flow.copy()

            prev_gray = gray.copy()

        cap.release()

        if len(optical_flow_discontinuities) == 0:
            return 50.0

        # === 綜合評分 ===
        score = 35.0  # 基礎分

        # 1. 光流不連續（AI特徵）
        if len(optical_flow_discontinuities) > 0:
            avg_flow_disc = np.mean(optical_flow_discontinuities)
            if avg_flow_disc > 1.5:
                score += 35.0
                logging.info(f"PVD: High optical flow discontinuity {avg_flow_disc:.3f} - AI feature")
            elif avg_flow_disc > 1.0:
                score += 25.0
            elif avg_flow_disc > 0.5:
                score += 15.0
            elif avg_flow_disc < 0.2:
                score -= 15.0
                logging.info(f"PVD: Smooth optical flow {avg_flow_disc:.3f} - Real feature")

        # 2. 加速度異常
        if len(acceleration_anomalies) > 0:
            avg_accel = np.mean(acceleration_anomalies)
            if avg_accel > 1.5:
                score += 28.0
                logging.info(f"PVD: Abnormal acceleration {avg_accel:.3f} - AI feature")
            elif avg_accel > 0.8:
                score += 18.0
            elif avg_accel > 0.4:
                score += 10.0
            elif avg_accel < 0.2:
                score -= 12.0

        # 3. 運動模糊缺失
        if len(motion_blur_consistency) > 0:
            avg_blur_issue = np.mean(motion_blur_consistency)
            if avg_blur_issue > 1.2:
                score += 22.0
                logging.info(f"PVD: Missing motion blur {avg_blur_issue:.3f} - AI feature")
            elif avg_blur_issue > 0.7:
                score += 12.0
            elif avg_blur_issue < 0.3:
                score -= 8.0

        # 4. 邊緣穩定性異常
        if len(edge_stability_scores) > 0:
            avg_edge_issue = np.mean(edge_stability_scores)
            if avg_edge_issue > 1.2:
                score += 18.0
                logging.info(f"PVD: Edge stability issue {avg_edge_issue:.3f} - AI artifact")
            elif avg_edge_issue > 0.7:
                score += 10.0
            elif avg_edge_issue < 0.3:
                score -= 10.0

        # 限制分數範圍
        score = max(5.0, min(95.0, score))

        logging.info(f"PVD: flow_disc={np.mean(optical_flow_discontinuities) if len(optical_flow_discontinuities) > 0 else 0:.3f}, "
                    f"accel={np.mean(acceleration_anomalies) if len(acceleration_anomalies) > 0 else 0:.3f}, "
                    f"blur={np.mean(motion_blur_consistency) if len(motion_blur_consistency) > 0 else 0:.3f}, "
                    f"edge={np.mean(edge_stability_scores) if len(edge_stability_scores) > 0 else 0:.3f}, score={score:.1f}")

        return score

    except Exception as e:
        logging.error(f"Error in physics_violation_detector: {e}")
        return 50.0

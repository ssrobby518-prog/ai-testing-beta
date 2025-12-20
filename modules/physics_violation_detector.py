#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模組 Physics Violation Detector v2.0 - TSAR-RAPTOR Phase I
第一性原理：真實世界遵守牛頓運動定律（加加速度連續性、景深一致性）
AI生成視頻違反物理規律：運動不連續、景深矛盾、慣性缺失

關鍵差異:
- 真實: Jerk連續(<3違規/s) + 景深一致 + 慣性守恆
- AI: Jerk突變(>5違規/s) + 景深矛盾 + 慣性違反

優化記錄:
- v1.0: 差距-0.5 (檢測運動幅度，真實手持會觸發)
- v2.0: 預期+30 (檢測運動連續性，AI的幀間不一致)
"""

import logging
import cv2
import numpy as np
from scipy import ndimage

logging.basicConfig(level=logging.INFO)

def detect(file_path):
    """
    第一性原理v2.0：牛頓運動定律檢測（Jerk + 景深一致性）

    真實物理運動（第一性原理）：
    1. Jerk連續性（加加速度）: 力的變化是平滑的，不會瞬間跳變
       - 位置 x(t)
       - 速度 v(t) = dx/dt (一階導數)
       - 加速度 a(t) = dv/dt (二階導數)
       - Jerk j(t) = da/dt (三階導數) ← 關鍵檢測點

    2. 景深一致性（Depth of Field）: 近處物體遮擋遠處物體，不會穿透

    3. 慣性守恆: 物體不會無緣無故改變運動狀態

    AI生成視頻缺陷（非物理）：
    1. Jerk突變: 幀間生成不一致，運動"跳躍"
    2. 景深矛盾: 深度估計錯誤，前後景關係混亂
    3. 慣性違反: 物體運動突然加速/減速（沒有外力）
    """
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return 50.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps > 120:
            fps = 30  # 默認值

        # 關鍵指標（v2.0重新設計）
        jerk_violations = []           # Jerk突變次數（物理核心）
        depth_inconsistencies = []     # 景深矛盾次數
        inertia_violations = []        # 慣性違反次數
        motion_smoothness_scores = []  # 運動平滑度

        prev_gray = None
        prev_flow = None
        prev_acceleration = None
        frame_history = []  # 保存最近5幀用於景深分析

        sample_frames = min(90, total_frames)  # 增加採樣以檢測Jerk

        for i in range(sample_frames):
            if total_frames > 0:
                frame_pos = int(i * total_frames / sample_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # 保存幀歷史（用於景深分析）
            frame_history.append(gray.copy())
            if len(frame_history) > 5:
                frame_history.pop(0)

            if prev_gray is not None:
                # === 1. Jerk檢測（第一性原理核心）===
                # 真實物理：Jerk（加加速度）是平滑的
                # AI生成：幀間不一致導致Jerk突變

                # 計算光流（速度場）
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

                # 光流大小（速度）
                flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

                if prev_flow is not None:
                    # 加速度 = 速度變化 / 時間
                    # 使用numpy廣播計算整個場的加速度
                    dt = 1.0 / fps
                    acceleration_x = (flow[..., 0] - prev_flow[..., 0]) / dt
                    acceleration_y = (flow[..., 1] - prev_flow[..., 1]) / dt
                    acceleration = np.sqrt(acceleration_x**2 + acceleration_y**2)

                    if prev_acceleration is not None:
                        # Jerk = 加速度變化 / 時間（三階導數）
                        jerk_x = (acceleration_x - prev_acceleration[0]) / dt
                        jerk_y = (acceleration_y - prev_acceleration[1]) / dt
                        jerk_magnitude = np.sqrt(jerk_x**2 + jerk_y**2)

                        # 只分析有運動的區域（避免靜態區域噪聲）
                        motion_mask = (flow_magnitude > 0.5) | \
                                     (np.sqrt(prev_flow[..., 0]**2 + prev_flow[..., 1]**2) > 0.5)

                        if np.sum(motion_mask) > 500:
                            # 計算運動區域的Jerk統計
                            jerk_in_motion = jerk_magnitude[motion_mask]

                            # 第一性原理：真實物理Jerk應該小（<500 pixels/s³）
                            # AI生成：Jerk大（>2000 pixels/s³）表示運動不連續
                            jerk_threshold = 1500  # 調整後的閾值
                            jerk_violation_count = np.sum(jerk_in_motion > jerk_threshold)
                            jerk_violation_ratio = jerk_violation_count / len(jerk_in_motion)

                            # 記錄違規比例
                            jerk_violations.append(jerk_violation_ratio)

                            # 額外檢測：極端Jerk
                            extreme_jerk_threshold = 3000
                            extreme_jerk_count = np.sum(jerk_in_motion > extreme_jerk_threshold)
                            if extreme_jerk_count > 10:
                                jerk_violations.append(jerk_violation_ratio * 1.5)  # 加權

                    # 保存當前加速度
                    prev_acceleration = (acceleration_x.copy(), acceleration_y.copy())

                # === 2. 慣性守恆檢測 ===
                # 第一性原理：物體在無外力作用下，速度保持不變（牛頓第一定律）
                # AI生成：可能突然加速/減速（沒有物理原因）

                if prev_flow is not None:
                    # 計算速度變化
                    velocity_change = np.sqrt(
                        (flow[..., 0] - prev_flow[..., 0])**2 +
                        (flow[..., 1] - prev_flow[..., 1])**2
                    )

                    # 在運動區域檢測突然的速度變化
                    motion_mask = flow_magnitude > 1.0
                    if np.sum(motion_mask) > 500:
                        velocity_changes = velocity_change[motion_mask]

                        # 真實視頻：速度變化平滑（慣性）
                        # AI視頻：速度突變（無慣性）
                        sudden_change_threshold = 3.0 * (30.0 / fps)
                        sudden_changes = np.sum(velocity_changes > sudden_change_threshold)
                        inertia_violation_ratio = sudden_changes / len(velocity_changes)

                        inertia_violations.append(inertia_violation_ratio)

                # === 3. 運動平滑度（輔助指標）===
                # 計算運動場的空間一致性
                if np.sum(flow_magnitude > 0.5) > 500:
                    # 使用Sobel檢測運動場的梯度
                    flow_grad_x = cv2.Sobel(flow[..., 0], cv2.CV_64F, 1, 0, ksize=3)
                    flow_grad_y = cv2.Sobel(flow[..., 1], cv2.CV_64F, 0, 1, ksize=3)
                    flow_gradient = np.sqrt(flow_grad_x**2 + flow_grad_y**2)

                    # 在運動區域計算梯度
                    motion_mask = flow_magnitude > 0.5
                    flow_gradients = flow_gradient[motion_mask]

                    # 真實運動：運動場平滑（低梯度）
                    # AI運動：運動場不連續（高梯度）
                    avg_flow_gradient = np.mean(flow_gradients)
                    if avg_flow_gradient > 2.0:
                        motion_smoothness_scores.append(1.5)  # 不平滑
                    elif avg_flow_gradient > 1.0:
                        motion_smoothness_scores.append(1.0)
                    else:
                        motion_smoothness_scores.append(0.0)  # 平滑

                prev_flow = flow.copy()

            # === 4. 景深一致性檢測 ===
            # 第一性原理：近處物體遮擋遠處物體（光學物理）
            # AI生成：深度估計錯誤，可能出現穿透、遮擋關係混亂

            if len(frame_history) >= 3:
                # 使用幀間差異檢測運動物體
                frame_diff1 = cv2.absdiff(frame_history[-1], frame_history[-2])
                frame_diff2 = cv2.absdiff(frame_history[-2], frame_history[-3])

                # 檢測邊緣（潛在的遮擋邊界）
                edges = cv2.Canny(gray, 50, 150)

                # 在運動邊緣處檢測深度矛盾
                # 如果邊緣處有運動，但前後幀的差異模式不一致，可能是景深錯誤
                edge_motion_mask = (edges > 0) & ((frame_diff1 > 10) | (frame_diff2 > 10))

                if np.sum(edge_motion_mask) > 200:
                    # 檢測遮擋關係的一致性
                    # 簡化方法：比較邊緣兩側的亮度變化
                    kernel = np.ones((3, 3), np.uint8)
                    edge_dilated = cv2.dilate(edges, kernel, iterations=1)

                    # 計算邊緣區域的亮度變化
                    edge_region = gray[edge_dilated > 0]
                    if len(edge_region) > 100:
                        edge_brightness_std = np.std(edge_region)

                        # AI生成：邊緣區域亮度變化異常（景深矛盾）
                        if edge_brightness_std > 60:
                            depth_inconsistencies.append(1.5)
                        elif edge_brightness_std > 45:
                            depth_inconsistencies.append(1.0)
                        else:
                            depth_inconsistencies.append(0.0)

            prev_gray = gray.copy()

        cap.release()

        if len(jerk_violations) == 0:
            return 50.0

        # === v2.0 評分邏輯（第一性原理驅動）===
        score = 50.0  # 中性基礎分

        # 1. Jerk違規（核心指標 - 權重最高）
        avg_jerk_violation = np.mean(jerk_violations)
        if avg_jerk_violation > 0.15:  # AI特徵（>15%運動區域有Jerk突變）
            score += 40.0
            logging.info(f"PVD v2: High jerk violations {avg_jerk_violation:.3f} - AI physics violation")
        elif avg_jerk_violation > 0.08:
            score += 25.0
        elif avg_jerk_violation > 0.04:
            score += 12.0
        elif avg_jerk_violation < 0.02:  # 真實特徵（Jerk平滑）
            score -= 25.0
            logging.info(f"PVD v2: Smooth jerk {avg_jerk_violation:.3f} - Real physics")

        # 2. 慣性違規
        if len(inertia_violations) > 0:
            avg_inertia_violation = np.mean(inertia_violations)
            if avg_inertia_violation > 0.12:  # AI特徵（慣性違反）
                score += 25.0
                logging.info(f"PVD v2: Inertia violations {avg_inertia_violation:.3f} - AI feature")
            elif avg_inertia_violation > 0.06:
                score += 15.0
            elif avg_inertia_violation < 0.03:  # 真實特徵（遵守慣性）
                score -= 15.0

        # 3. 運動平滑度
        if len(motion_smoothness_scores) > 0:
            avg_smoothness = np.mean(motion_smoothness_scores)
            if avg_smoothness > 1.2:  # AI特徵（運動場不連續）
                score += 18.0
            elif avg_smoothness > 0.7:
                score += 10.0
            elif avg_smoothness < 0.3:  # 真實特徵（運動場平滑）
                score -= 12.0

        # 4. 景深一致性
        if len(depth_inconsistencies) > 0:
            avg_depth_issue = np.mean(depth_inconsistencies)
            if avg_depth_issue > 1.2:  # AI特徵（景深矛盾）
                score += 20.0
                logging.info(f"PVD v2: Depth inconsistencies {avg_depth_issue:.3f} - AI depth error")
            elif avg_depth_issue > 0.7:
                score += 12.0
            elif avg_depth_issue < 0.3:  # 真實特徵（景深一致）
                score -= 10.0

        # 限制分數範圍
        score = max(5.0, min(95.0, score))

        logging.info(f"PVD v2.0: jerk={avg_jerk_violation:.3f}, inertia={np.mean(inertia_violations) if len(inertia_violations) > 0 else 0:.3f}, "
                    f"smoothness={np.mean(motion_smoothness_scores) if len(motion_smoothness_scores) > 0 else 0:.3f}, "
                    f"depth={np.mean(depth_inconsistencies) if len(depth_inconsistencies) > 0 else 0:.3f}, score={score:.1f}")

        return score

    except Exception as e:
        logging.error(f"Error in physics_violation_detector v2.0: {e}")
        return 50.0

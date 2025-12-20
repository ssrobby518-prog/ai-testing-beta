#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模組 Blink Dynamics Analyzer v2.0 - TSAR-RAPTOR Phase II
第一性原理：真實人類眨眼有「快閉慢開」不對稱特徵，AI無法模擬眼瞼肌肉的生物力學
使用EAR（Eye Aspect Ratio）檢測眨眼動態，關鍵是閉眼/睜眼速度比

關鍵差異:
- 真實: 閉眼快(100-150ms) + 睜眼慢(150-250ms) + 不對稱比 1.3-2.5
- AI: 對稱運動(<1.1比例) + 速度曲線線性 + 缺少生理細節

優化記錄:
- v1.0: 差距0.0 (只檢測眨眼頻率，AI也可以有正常頻率)
- v2.0: 預期+30 (檢測快閉慢開不對稱性，AI無法模擬眼瞼肌肉動力學)
"""

import logging
import cv2
import numpy as np
from scipy import signal

logging.basicConfig(level=logging.INFO)

def detect(file_path):
    """
    第一性原理v2.0：EAR + 眨眼不對稱性檢測

    真實人類眨眼特徵（第一性原理 - 生物力學）：
    1. 快閉慢開（Fast Close, Slow Open）:
       - 閉眼階段：100-150ms（提上瞼肌快速收縮）
       - 睜眼階段：150-250ms（重力+肌肉緩慢放鬆）
       - 不對稱比：開眼時間/閉眼時間 ∈ [1.3, 2.5]

    2. EAR曲線形狀:
       - 下降階段：陡峭（快速閉合）
       - 上升階段：平緩（緩慢打開）
       - 峰值銳利（完全睜開）

    3. 眨眼完整性:
       - 真實眨眼：EAR降到 < 0.2（完全閉合）
       - 部分眨眼：EAR僅降到 0.2-0.3（AI可能產生的不完整眨眼）

    AI生成視頻缺陷：
    1. 對稱運動：閉眼和睜眼速度相同（比例 < 1.1）
    2. 線性曲線：缺少生物力學的非線性特徵
    3. 不完整眨眼：EAR未降到閾值以下
    """
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return 50.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps > 120:
            fps = 30.0

        # 關鍵指標（v2.0重新設計）
        blink_asymmetries = []        # 眨眼不對稱比（開眼/閉眼時間）
        close_speeds = []             # 閉眼速度
        open_speeds = []              # 睜眼速度
        blink_completeness = []       # 眨眼完整性（最小EAR值）
        ear_curve_shapes = []         # EAR曲線形狀特徵

        # EAR時間序列
        ear_values = []

        # 臉部和眼睛檢測器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        sample_frames = min(180, total_frames)

        for i in range(sample_frames):
            if total_frames > 0:
                frame_pos = int(i * total_frames / sample_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 檢測臉部
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            if len(faces) > 0:
                # 選擇最大的臉
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face_roi = gray[y:y+h, x:x+w]

                # 在臉部ROI中檢測眼睛
                eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5)

                if len(eyes) >= 2:
                    # 計算EAR（Eye Aspect Ratio）
                    # 簡化版本：使用眼睛區域的高寬比作為EAR的近似

                    # 選擇兩隻眼睛（通常檢測到左右眼）
                    eyes_sorted = sorted(eyes, key=lambda e: e[0])  # 按x坐標排序

                    left_eye = eyes_sorted[0]
                    right_eye = eyes_sorted[-1]

                    # 計算每隻眼睛的EAR近似值
                    # EAR = 眼睛高度 / 眼睛寬度
                    # 真實定義需要6個landmark點，這裡用簡化版本

                    def compute_simple_ear(eye_rect):
                        ex, ey, ew, eh = eye_rect
                        # 提取眼睛ROI
                        eye_roi = face_roi[ey:ey+eh, ex:ex+ew]

                        if eye_roi.size == 0:
                            return 0.3  # 默認值

                        # 使用垂直邊緣密度作為"睜眼程度"的指標
                        # 睜眼：上下眼瞼分離，垂直邊緣多
                        # 閉眼：上下眼瞼合併，垂直邊緣少

                        # 計算垂直梯度
                        sobel_y = cv2.Sobel(eye_roi, cv2.CV_64F, 0, 1, ksize=3)
                        vertical_edges = np.abs(sobel_y)

                        # 計算垂直邊緣密度
                        edge_density = np.mean(vertical_edges)

                        # 歸一化到0-1範圍（經驗值）
                        # 完全閉眼：edge_density ≈ 5-10
                        # 完全睜眼：edge_density ≈ 20-40
                        ear_approx = np.clip((edge_density - 5) / 35, 0.0, 1.0)

                        return ear_approx

                    left_ear = compute_simple_ear(left_eye)
                    right_ear = compute_simple_ear(right_eye)

                    # 平均兩隻眼睛的EAR
                    ear = (left_ear + right_ear) / 2.0
                    ear_values.append(ear)

        cap.release()

        if len(ear_values) < 30:
            return 50.0

        ear_values = np.array(ear_values)

        # === 1. 眨眼事件檢測 ===
        # 使用EAR閾值檢測眨眼
        # 真實眨眼：EAR降到 < 0.2
        # AI可能：EAR僅降到 0.2-0.3（不完整）

        # 平滑EAR信號（移除高頻噪聲）
        window_size = max(3, int(fps / 10))  # 100ms窗口
        ear_smooth = signal.savgol_filter(ear_values, window_size if window_size % 2 == 1 else window_size + 1, 2)

        # 檢測眨眼（EAR下降事件）
        ear_threshold = 0.25  # 低於此值視為眨眼

        # 找到所有低於閾值的區域
        below_threshold = ear_smooth < ear_threshold

        # 找到眨眼事件的起始和結束
        blink_events = []
        in_blink = False
        blink_start = 0

        for i in range(len(below_threshold)):
            if below_threshold[i] and not in_blink:
                # 眨眼開始
                blink_start = i
                in_blink = True
            elif not below_threshold[i] and in_blink:
                # 眨眼結束
                blink_end = i
                in_blink = False

                # 記錄眨眼事件
                if blink_end - blink_start >= 2:  # 至少2幀
                    blink_events.append((blink_start, blink_end))

        if len(blink_events) == 0:
            # 沒有檢測到眨眼，可能是AI或無臉視頻
            return 50.0

        # === 2. 分析每個眨眼事件的動態特徵 ===
        for blink_start, blink_end in blink_events:
            # 擴展窗口以包含完整的眨眼動作（前後各加30%）
            window_size = blink_end - blink_start
            window_start = max(0, blink_start - int(window_size * 0.5))
            window_end = min(len(ear_smooth), blink_end + int(window_size * 0.5))

            if window_end - window_start < 5:
                continue

            # 提取眨眼窗口的EAR曲線
            blink_ear = ear_smooth[window_start:window_end]

            # 找到EAR的最小值（完全閉眼時刻）
            min_idx = np.argmin(blink_ear)
            min_ear = blink_ear[min_idx]

            # 2.1 眨眼完整性
            # 真實：min_ear < 0.2（完全閉眼）
            # AI：min_ear > 0.2（不完整）
            blink_completeness.append(min_ear)

            # 2.2 快閉慢開分析
            # 找到閉眼階段（從峰值到谷底）和睜眼階段（從谷底到峰值）

            # 閉眼階段：從開始到最小值
            close_phase = blink_ear[:min_idx + 1]
            # 睜眼階段：從最小值到結束
            open_phase = blink_ear[min_idx:]

            if len(close_phase) >= 3 and len(open_phase) >= 3:
                # 計算閉眼時間和睜眼時間（幀數）
                close_duration = len(close_phase)
                open_duration = len(open_phase)

                # 轉換為時間（毫秒）
                close_time_ms = close_duration / fps * 1000
                open_time_ms = open_duration / fps * 1000

                # 計算不對稱比（開眼時間/閉眼時間）
                # 第一性原理：真實 ∈ [1.3, 2.5]，AI < 1.1
                if close_time_ms > 0:
                    asymmetry = open_time_ms / close_time_ms
                    blink_asymmetries.append(asymmetry)

                # 計算閉眼和睜眼速度（EAR變化率）
                # 速度 = ΔEAR / Δt

                # 閉眼速度（EAR下降速度，取絕對值）
                if len(close_phase) > 1:
                    close_speed = np.abs(np.diff(close_phase)).mean() * fps
                    close_speeds.append(close_speed)

                # 睜眼速度（EAR上升速度）
                if len(open_phase) > 1:
                    open_speed = np.abs(np.diff(open_phase)).mean() * fps
                    open_speeds.append(open_speed)

            # 2.3 EAR曲線形狀分析
            # 真實：下降陡峭，上升平緩
            # AI：對稱或線性

            # 計算曲線的偏斜度（Skewness）
            # 負偏斜：快閉慢開（真實特徵）
            # 接近0：對稱（AI特徵）
            if len(blink_ear) > 5:
                from scipy.stats import skew
                curve_skewness = skew(blink_ear)
                ear_curve_shapes.append(curve_skewness)

        # === v2.0 評分邏輯（第一性原理驅動）===
        score = 50.0  # 中性基礎分

        # 1. 眨眼不對稱比（核心指標 - 權重最高）
        if len(blink_asymmetries) > 0:
            avg_asymmetry = np.mean(blink_asymmetries)

            # 第一性原理：真實 ∈ [1.3, 2.5]，AI < 1.1
            if avg_asymmetry < 1.1:  # AI特徵（對稱運動）
                score += 40.0
                logging.info(f"BDA v2: Symmetric blink {avg_asymmetry:.2f} - AI symmetric motion")
            elif avg_asymmetry < 1.2:
                score += 25.0
            elif avg_asymmetry >= 1.3 and avg_asymmetry <= 2.5:  # 真實特徵（快閉慢開）
                score -= 30.0
                logging.info(f"BDA v2: Asymmetric blink {avg_asymmetry:.2f} - Real fast-close slow-open")
            elif avg_asymmetry > 1.2 and avg_asymmetry < 1.3:
                score -= 15.0
            elif avg_asymmetry > 2.5:  # 過度不對稱（異常）
                score += 20.0

        # 2. 閉眼/睜眼速度比
        if len(close_speeds) > 0 and len(open_speeds) > 0:
            avg_close_speed = np.mean(close_speeds)
            avg_open_speed = np.mean(open_speeds)

            if avg_open_speed > 0:
                speed_ratio = avg_close_speed / avg_open_speed

                # 真實：閉眼快，睜眼慢（比例 > 1.2）
                # AI：速度相似（比例 ≈ 1.0）
                if speed_ratio < 1.1:  # AI特徵（速度對稱）
                    score += 25.0
                elif speed_ratio > 1.5:  # 真實特徵（閉眼明顯快）
                    score -= 20.0

        # 3. 眨眼完整性
        if len(blink_completeness) > 0:
            avg_completeness = np.mean(blink_completeness)

            # 真實：完全閉眼（min EAR < 0.2）
            # AI：不完整（min EAR > 0.2）
            if avg_completeness > 0.25:  # AI特徵（不完整眨眼）
                score += 22.0
                logging.info(f"BDA v2: Incomplete blink {avg_completeness:.3f} - AI incomplete closure")
            elif avg_completeness < 0.15:  # 真實特徵（完全閉眼）
                score -= 18.0

        # 4. EAR曲線形狀（偏斜度）
        if len(ear_curve_shapes) > 0:
            avg_skewness = np.mean(ear_curve_shapes)

            # 真實：負偏斜（快閉慢開導致的非對稱曲線）
            # AI：接近0（對稱曲線）
            if avg_skewness > -0.2:  # AI特徵（對稱或正偏斜）
                score += 18.0
            elif avg_skewness < -0.8:  # 真實特徵（明顯負偏斜）
                score -= 15.0

        # 5. 眨眼頻率檢查（輔助指標）
        duration_sec = len(ear_values) / fps
        blink_rate_per_min = len(blink_events) / (duration_sec / 60.0)

        # 正常範圍：10-30次/分鐘
        if blink_rate_per_min < 5 or blink_rate_per_min > 40:
            score += 12.0  # 異常頻率（可能是AI）
        elif 10 <= blink_rate_per_min <= 30:
            score -= 8.0  # 正常頻率（真實特徵）

        # 限制分數範圍
        score = max(5.0, min(95.0, score))

        logging.info(f"BDA v2.0: asymmetry={np.mean(blink_asymmetries) if len(blink_asymmetries) > 0 else 0:.2f}, "
                    f"completeness={np.mean(blink_completeness) if len(blink_completeness) > 0 else 0:.3f}, "
                    f"skewness={np.mean(ear_curve_shapes) if len(ear_curve_shapes) > 0 else 0:.2f}, "
                    f"blinks={len(blink_events)}, rate={blink_rate_per_min:.1f}/min, score={score:.1f}")

        return score

    except Exception as e:
        logging.error(f"Error in blink_dynamics_analyzer v2.0: {e}")
        return 50.0

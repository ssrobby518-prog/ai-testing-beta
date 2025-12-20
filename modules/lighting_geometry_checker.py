#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模組 Lighting Geometry Checker v2.0 - TSAR-RAPTOR Phase II
第一性原理：真實手持視頻有特徵性的相機抖動（0.5-2度/秒旋轉）
AI生成視頻和三腳架視頻極度穩定（<0.1度/秒）

關鍵差異:
- 真實手持: 旋轉抖動 0.5-2 度/秒 + 不規則抖動模式
- AI/三腳架: 旋轉抖動 < 0.1 度/秒 + 完美穩定或線性移動

優化記錄:
- v1.0: 差距-2.1 (檢測運動幅度，誤判手持為真實)
- v2.0: 預期+20 (檢測相機旋轉抖動，AI無法模擬手持微運動)
"""

import logging
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)

def detect(file_path):
    """
    第一性原理v2.0：相機旋轉抖動檢測（Hand-held Jitter Detection）

    真實手持視頻特徵（第一性原理）：
    1. 相機旋轉抖動:
       - 平均旋轉速度: 0.5-2.0 度/秒
       - 不規則抖動模式（非線性）
       - 多軸微運動（pitch, yaw, roll）

    2. 自然手部振動:
       - 頻率約 8-12 Hz（生理顫抖）
       - 振幅隨時間變化

    3. 光照變化:
       - 隨運動產生動態陰影
       - 反射點位置變化

    AI生成視頻缺陷：
    1. 相機極度穩定: 旋轉 < 0.1 度/秒（完美穩定）
    2. 線性運動: 只有平移，缺少旋轉（運鏡過於完美）
    3. 不自然的穩定: 即使有運動，也是完美的線性軌跡
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
        rotation_angles = []          # 幀間旋轉角度（度）
        rotation_speeds = []          # 旋轉速度（度/秒）
        translation_magnitudes = []   # 平移幅度
        jitter_irregularities = []    # 抖動不規則性
        light_variations = []         # 光照變化

        sample_frames = min(120, total_frames)
        prev_gray = None
        prev_features = None

        for i in range(sample_frames):
            if total_frames > 0:
                frame_pos = int(i * total_frames / sample_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # 縮放以加速處理
            scale = 320.0 / max(w, h)
            if scale < 1.0:
                gray_scaled = cv2.resize(gray, (int(w * scale), int(h * scale)))
            else:
                gray_scaled = gray

            if prev_gray is not None:
                # === 1. 估計相機旋轉和平移（使用特徵匹配）===
                # 使用ORB特徵點匹配來估計相機運動

                # 檢測角點（用於光流估計）
                corners = cv2.goodFeaturesToTrack(
                    prev_gray,
                    maxCorners=100,
                    qualityLevel=0.01,
                    minDistance=10
                )

                if corners is not None and len(corners) >= 4:
                    # 計算光流
                    next_corners, status, _ = cv2.calcOpticalFlowPyrLK(
                        prev_gray,
                        gray_scaled,
                        corners,
                        None
                    )

                    # 過濾有效的匹配點
                    good_old = corners[status == 1]
                    good_new = next_corners[status == 1]

                    if len(good_old) >= 4:
                        # 使用仿射變換估計相機運動
                        # 仿射矩陣包含旋轉、縮放、平移
                        M, inliers = cv2.estimateAffinePartial2D(
                            good_old,
                            good_new,
                            method=cv2.RANSAC
                        )

                        if M is not None:
                            # 從仿射矩陣提取旋轉角度
                            # M = [[cos(θ) -sin(θ) tx]
                            #      [sin(θ)  cos(θ) ty]]
                            rotation_rad = np.arctan2(M[1, 0], M[0, 0])
                            rotation_deg = np.abs(np.degrees(rotation_rad))

                            rotation_angles.append(rotation_deg)

                            # 計算旋轉速度（度/秒）
                            rotation_speed = rotation_deg * fps / (sample_frames / total_frames if total_frames > 0 else 1)
                            rotation_speeds.append(rotation_speed)

                            # 計算平移幅度
                            tx = M[0, 2]
                            ty = M[1, 2]
                            translation = np.sqrt(tx**2 + ty**2)
                            translation_magnitudes.append(translation)

                # === 2. 光照變化檢測 ===
                # 真實手持視頻：光照隨運動產生動態變化
                # AI視頻：光照過於穩定

                # 計算全局亮度變化
                brightness_curr = np.mean(gray_scaled)
                brightness_prev = np.mean(prev_gray)
                brightness_change = np.abs(brightness_curr - brightness_prev)
                light_variations.append(brightness_change)

            prev_gray = gray_scaled

        cap.release()

        if len(rotation_speeds) < 10:
            return 50.0

        # === v2.0 評分邏輯（第一性原理驅動）===
        score = 50.0  # 中性基礎分

        # 1. 相機旋轉速度（核心指標）
        avg_rotation_speed = np.mean(rotation_speeds)
        median_rotation_speed = np.median(rotation_speeds)

        # 第一性原理：真實手持 0.5-2 度/秒，AI/三腳架 < 0.1 度/秒
        if avg_rotation_speed < 0.1:  # AI特徵（極度穩定）
            score += 40.0
            logging.info(f"LGC v2: Ultra-stable camera {avg_rotation_speed:.3f}°/s - AI or tripod")
        elif avg_rotation_speed < 0.3:
            score += 28.0
        elif avg_rotation_speed < 0.5:
            score += 15.0
        elif 0.5 <= avg_rotation_speed <= 2.5:  # 真實手持範圍
            score -= 30.0
            logging.info(f"LGC v2: Natural handheld jitter {avg_rotation_speed:.3f}°/s - Real video")
        elif avg_rotation_speed > 3.0:  # 過度抖動（可能是故意的晃動）
            score += 10.0

        # 2. 旋轉抖動不規則性
        # 真實手持：抖動不規則（高標準差）
        # AI：抖動規則或無抖動（低標準差）
        if len(rotation_speeds) > 5:
            rotation_std = np.std(rotation_speeds)

            if rotation_std < 0.05:  # AI特徵（抖動規則或無抖動）
                score += 25.0
                logging.info(f"LGC v2: Regular rotation pattern std={rotation_std:.3f} - AI synthetic")
            elif rotation_std > 0.3:  # 真實特徵（抖動不規則）
                score -= 18.0

        # 3. 旋轉vs平移比例
        # 真實手持：有旋轉+平移
        # AI線性運鏡：只有平移，缺少旋轉
        if len(translation_magnitudes) > 0:
            avg_translation = np.mean(translation_magnitudes)

            # 計算旋轉/平移比例
            if avg_translation > 0.1:
                rotation_translation_ratio = avg_rotation_speed / (avg_translation + 1e-6)

                # AI特徵：高平移但無旋轉（完美線性運鏡）
                if rotation_translation_ratio < 0.05 and avg_translation > 1.0:
                    score += 22.0
                    logging.info(f"LGC v2: Linear motion without rotation - AI camera path")

        # 4. 抖動頻率分析
        # 真實手部顫抖：8-12 Hz
        # 這需要足夠的時間分辨率，如果幀數不足則跳過
        if len(rotation_angles) >= 30:
            # 對旋轉角度序列進行FFT
            rotation_signal = np.array(rotation_angles) - np.mean(rotation_angles)

            # FFT
            fft_result = np.fft.rfft(rotation_signal)
            fft_freqs = np.fft.rfftfreq(len(rotation_signal), d=1.0/fps)
            fft_power = np.abs(fft_result) ** 2

            # 檢測是否在8-12 Hz有峰值（手部生理顫抖）
            tremor_band_mask = (fft_freqs >= 8) & (fft_freqs <= 12)

            if np.any(tremor_band_mask):
                tremor_power = np.sum(fft_power[tremor_band_mask])
                total_power = np.sum(fft_power)

                tremor_ratio = tremor_power / (total_power + 1e-10)

                # 真實手持：手部顫抖頻段有能量
                if tremor_ratio > 0.1:
                    score -= 15.0
                    logging.info(f"LGC v2: Hand tremor detected (8-12Hz) - Real handheld")

        # 5. 光照變化動態性
        # 真實手持：光照隨運動變化
        # AI：光照過於穩定
        if len(light_variations) > 5:
            avg_light_change = np.mean(light_variations)

            # AI特徵：極低光照變化
            if avg_light_change < 0.5:
                score += 15.0
            elif avg_light_change > 2.0:  # 真實特徵：動態光照
                score -= 10.0

        # 6. 零運動檢測（完全靜止）
        # 如果視頻幾乎完全靜止（三腳架或AI靜態生成）
        zero_motion_ratio = np.sum(np.array(rotation_speeds) < 0.05) / len(rotation_speeds)

        if zero_motion_ratio > 0.8:  # 80%以上的幀幾乎無運動
            score += 30.0
            logging.info(f"LGC v2: Nearly static video {zero_motion_ratio*100:.1f}% - AI or tripod")
        elif zero_motion_ratio < 0.2:  # 持續運動
            score -= 12.0

        # 限制分數範圍
        score = max(5.0, min(95.0, score))

        logging.info(f"LGC v2.0: rotation_speed={avg_rotation_speed:.3f}°/s, "
                    f"std={np.std(rotation_speeds):.3f}, "
                    f"zero_motion={zero_motion_ratio*100:.1f}%, "
                    f"light_change={np.mean(light_variations) if len(light_variations) > 0 else 0:.2f}, "
                    f"score={score:.1f}")

        return score

    except Exception as e:
        logging.error(f"Error in lighting_geometry_checker v2.0: {e}")
        return 50.0

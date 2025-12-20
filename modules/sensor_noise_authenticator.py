#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模組 Sensor Noise Authenticator v2.0 - TSAR-RAPTOR Phase I
第一性原理：真實相機傳感器產生量子散粒噪聲（Shot Noise）+ 暗電流噪聲（Dark Current）
AI生成視頻的噪聲是算法噪聲，非物理噪聲，具有固定模式和空間相關性

關鍵差異:
- 真實: 白噪聲高(>0.7) + 固定模式低(<0.3) + 暗區噪聲強
- AI: 白噪聲低(<0.4) + 固定模式高(>0.6) + 暗區噪聲弱/無

優化記錄:
- v1.0: 差距-6.0 (檢測了壓縮噪聲，非傳感器噪聲)
- v2.0: 預期+40 (只分析暗區，檢測量子噪聲特徵)
"""

import logging
import cv2
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO)

def detect(file_path):
    """
    第一性原理v2.0：量子散粒噪聲檢測（只分析暗區）

    真實傳感器噪聲（物理特性）：
    1. 白噪聲（White Noise）: 頻譜平坦，所有頻率能量相等
    2. 隨機性（Randomness）: 空間上無相關性（自相關係數低）
    3. 暗電流噪聲（Dark Current）: 暗區有固定偏置
    4. 讀出噪聲（Readout Noise）: 每幀不同的隨機值

    AI算法噪聲（非物理）：
    1. 有色噪聲（Colored Noise）: 頻譜不平坦
    2. 固定模式（Fixed Pattern）: 空間上有相關性
    3. 暗區過度平滑: 缺少暗電流噪聲
    4. 時間相關性: 噪聲在幀間有模式
    """
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return 50.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 關鍵指標（v2.0重新設計）
        white_noise_ratios = []      # 白噪聲比例（頻譜平坦度）
        fixed_pattern_scores = []    # 固定模式噪聲分數（空間自相關）
        dark_noise_intensities = []  # 暗區噪聲強度
        readout_noise_variance = []  # 讀出噪聲方差（幀間變化）

        sample_frames = min(50, total_frames)

        for i in range(sample_frames):
            if total_frames > 0:
                frame_pos = int(i * total_frames / sample_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                break

            # 轉換為灰度
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            h, w = gray.shape

            # === 第一性原理：只分析暗區（亮度<50）===
            # 原因：壓縮不會破壞暗區的傳感器噪聲
            dark_mask = gray < 50

            if np.sum(dark_mask) < 1000:  # 暗區太小，跳過
                continue

            # 提取暗區像素
            dark_pixels = gray[dark_mask]

            # === 1. 白噪聲比例檢測（頻譜平坦度）===
            # 真實傳感器：白噪聲（所有頻率能量相等）
            # AI算法：有色噪聲（特定頻率能量高）

            # 計算暗區的頻譜
            dark_region = np.zeros_like(gray)
            dark_region[dark_mask] = gray[dark_mask]

            fft = np.fft.fft2(dark_region)
            magnitude = np.abs(np.fft.fftshift(fft))

            # 計算頻譜平坦度（Spectral Flatness）
            # 白噪聲 = 幾何平均 / 算術平均 ≈ 1
            # 有色噪聲 < 0.5
            magnitude_flat = magnitude.flatten()
            magnitude_flat = magnitude_flat[magnitude_flat > 0]

            if len(magnitude_flat) > 100:
                geometric_mean = stats.gmean(magnitude_flat)
                arithmetic_mean = np.mean(magnitude_flat)
                spectral_flatness = geometric_mean / (arithmetic_mean + 1e-6)
                white_noise_ratios.append(spectral_flatness)

            # === 2. 固定模式噪聲檢測（空間自相關）===
            # 真實噪聲：隨機，自相關係數低
            # AI噪聲：有固定模式，自相關係數高

            if np.sum(dark_mask) > 2000:
                # 提取暗區的2D噪聲patch
                dark_coords = np.argwhere(dark_mask)
                if len(dark_coords) > 100:
                    # 隨機選擇一個暗區patch
                    center_idx = np.random.choice(len(dark_coords))
                    cy, cx = dark_coords[center_idx]

                    patch_size = 32
                    if cy > patch_size and cx > patch_size and \
                       cy < h - patch_size and cx < w - patch_size:
                        patch = gray[cy-patch_size:cy+patch_size, cx-patch_size:cx+patch_size]

                        # 計算2D自相關
                        patch_mean = np.mean(patch)
                        patch_centered = patch - patch_mean

                        # 簡化：只計算中心點的自相關
                        autocorr = np.sum(patch_centered * patch_centered) / (np.std(patch) ** 2 * patch.size + 1e-6)

                        # 固定模式分數（歸一化到0-1）
                        fixed_pattern_score = min(1.0, autocorr / 100.0)
                        fixed_pattern_scores.append(fixed_pattern_score)

            # === 3. 暗區噪聲強度===
            # 真實：暗區有明顯噪聲（暗電流噪聲）
            # AI：暗區過度平滑
            dark_noise_std = np.std(dark_pixels)
            dark_noise_intensities.append(dark_noise_std)

            # === 4. 讀出噪聲變異性（幀間變化）===
            # 真實：每幀的讀出噪聲不同（隨機）
            # AI：噪聲模式固定
            if len(dark_noise_intensities) > 1:
                noise_variance = np.var(dark_noise_intensities[-10:]) if len(dark_noise_intensities) >= 10 else 0
                readout_noise_variance.append(noise_variance)


        cap.release()

        if len(white_noise_ratios) == 0:
            return 50.0

        # === v2.0 評分邏輯（第一性原理驅動）===
        score = 50.0  # 中性基礎分

        # 1. 白噪聲比例（關鍵指標）
        avg_white_noise = np.mean(white_noise_ratios)
        if avg_white_noise > 0.7:  # 真實傳感器特徵
            score -= 30.0
            logging.info(f"SNA v2: High white noise ratio {avg_white_noise:.3f} - REAL sensor")
        elif avg_white_noise < 0.4:  # AI算法噪聲
            score += 35.0
            logging.info(f"SNA v2: Low white noise ratio {avg_white_noise:.3f} - AI algorithm")
        elif avg_white_noise < 0.5:
            score += 20.0

        # 2. 固定模式噪聲（關鍵指標）
        if len(fixed_pattern_scores) > 0:
            avg_fixed_pattern = np.mean(fixed_pattern_scores)
            if avg_fixed_pattern > 0.6:  # AI特徵（固定模式明顯）
                score += 30.0
                logging.info(f"SNA v2: High fixed pattern {avg_fixed_pattern:.3f} - AI feature")
            elif avg_fixed_pattern < 0.3:  # 真實特徵（隨機性高）
                score -= 25.0

        # 3. 暗區噪聲強度
        if len(dark_noise_intensities) > 0:
            avg_dark_noise = np.mean(dark_noise_intensities)
            if avg_dark_noise < 2.0:  # 暗區過度平滑（AI特徵）
                score += 20.0
                logging.info(f"SNA v2: Low dark noise {avg_dark_noise:.2f} - AI smoothing")
            elif avg_dark_noise > 5.0:  # 暗區有明顯噪聲（真實特徵）
                score -= 15.0

        # 4. 讀出噪聲變異性
        if len(readout_noise_variance) > 0:
            avg_variance = np.mean(readout_noise_variance)
            if avg_variance < 0.5:  # 噪聲模式固定（AI特徵）
                score += 15.0
            elif avg_variance > 2.0:  # 噪聲隨機變化（真實特徵）
                score -= 10.0

        # 限制分數範圍
        score = max(5.0, min(95.0, score))

        logging.info(f"SNA v2.0: white_noise={avg_white_noise:.3f}, fixed_pattern={np.mean(fixed_pattern_scores) if len(fixed_pattern_scores) > 0 else 0:.3f}, "
                    f"dark_noise={np.mean(dark_noise_intensities):.2f}, score={score:.1f}")

        return score

    except Exception as e:
        logging.error(f"Error in sensor_noise_authenticator v2.0: {e}")
        return 50.0

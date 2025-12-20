#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模組 Frequency Analyzer v2.0 - TSAR-RAPTOR Phase I
第一性原理：真實相機的頻譜是連續的、熵高；AI生成的頻譜有周期性結構（GAN棋盤格）、熵低
GAN上採樣層（Transpose Convolution）會產生特定頻率的周期性峰值

關鍵差異:
- 真實: 頻譜連續 + 高熵(>6.5) + 無周期性峰值
- AI: 棋盤格模式 + 低熵(<5.5) + 周期性峰值

優化記錄:
- v1.0: 差距+4.1 (檢測高頻截斷，壓縮也會導致)
- v2.0: 預期+25 (檢測GAN棋盤格，AI特有的上採樣痕跡)
"""

import logging
import cv2
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO)

def detect(file_path):
    """
    第一性原理v2.0：GAN棋盤格檢測 + 頻譜熵分析

    真實相機頻譜特性：
    1. 頻譜連續性: 所有頻率都有能量，無突兀的峰值
    2. 高頻譜熵: 頻域能量分佈隨機（Entropy > 6.5）
    3. 方位角積分平滑: 徑向能量分佈平滑（無周期性）

    AI生成視頻缺陷（GAN上採樣痕跡）：
    1. 棋盤格模式（Checkerboard Pattern）:
       - GAN的轉置卷積（Transpose Convolution）會產生棋盤格偽影
       - 在FFT中表現為特定頻率的周期性峰值
       - 通常出現在 k=π/2, π, 3π/2 等位置

    2. 低頻譜熵:
       - AI生成的頻譜過於"整齊"（Entropy < 5.5）
       - 真實視頻的頻譜更混亂/隨機

    3. 方位角積分異常:
       - 計算每個半徑的方位角積分（Azimuthal Integration）
       - AI會有周期性振盪，真實是平滑下降
    """
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return 50.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 關鍵指標（v2.0重新設計）
        checkerboard_scores = []     # 棋盤格偽影分數
        spectral_entropies = []      # 頻譜熵
        azimuthal_smoothness = []    # 方位角積分平滑度
        periodic_peak_counts = []    # 周期性峰值計數

        sample_frames = min(60, total_frames)
        FFT_SIZE = 256  # 使用較小的FFT以專注於棋盤格頻率

        for i in range(sample_frames):
            if total_frames > 0:
                frame_pos = int(i * total_frames / sample_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Resize到FFT大小
            gray_resized = cv2.resize(gray, (FFT_SIZE, FFT_SIZE))

            # === 1. 2D FFT（空間頻域分析）===
            # 第一性原理：在空間頻域檢測GAN上採樣層的周期性痕跡

            # 計算2D FFT
            fft_2d = np.fft.fft2(gray_resized)
            fft_shifted = np.fft.fftshift(fft_2d)
            magnitude = np.abs(fft_shifted)

            # 對數尺度（更容易看到弱信號）
            magnitude_db = 20 * np.log10(magnitude + 1e-10)

            # === 1.1 棋盤格模式檢測 ===
            # GAN的轉置卷積會在特定頻率產生峰值
            # 檢測方法：在高頻區域尋找周期性峰值

            center_y, center_x = FFT_SIZE // 2, FFT_SIZE // 2

            # 提取高頻區域（距離中心 > 40%）
            y_coords, x_coords = np.ogrid[:FFT_SIZE, :FFT_SIZE]
            distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)

            high_freq_mask = (distances > FFT_SIZE * 0.4) & (distances < FFT_SIZE * 0.7)

            if np.sum(high_freq_mask) > 100:
                high_freq_region = magnitude_db[high_freq_mask]
                high_freq_mean = np.mean(high_freq_region)
                high_freq_std = np.std(high_freq_region)

                # 檢測異常峰值（比平均值高 2σ 以上）
                peak_threshold = high_freq_mean + 2 * high_freq_std
                peak_count = np.sum(high_freq_region > peak_threshold)
                peak_ratio = peak_count / len(high_freq_region)

                # AI特徵：高頻區域有異常多的峰值（棋盤格）
                # 真實視頻：高頻區域能量分佈較均勻
                if peak_ratio > 0.15:  # >15%的點是峰值
                    checkerboard_scores.append(2.0)
                elif peak_ratio > 0.10:
                    checkerboard_scores.append(1.5)
                elif peak_ratio > 0.06:
                    checkerboard_scores.append(1.0)
                else:
                    checkerboard_scores.append(0.0)

            # === 1.2 周期性峰值檢測（更精確）===
            # 檢測特定頻率（π/2, π, 3π/2）的周期性峰值
            # 這些頻率對應GAN上採樣的棋盤格模式

            periodic_peaks = 0
            # 檢測4個方向（0°, 90°, 180°, 270°）的對稱性
            # 棋盤格會在這些方向產生對稱的峰值

            # 在4個象限檢測峰值
            quadrants = [
                magnitude_db[center_y:, center_x:],      # 右下
                magnitude_db[center_y:, :center_x],      # 左下
                magnitude_db[:center_y, center_x:],      # 右上
                magnitude_db[:center_y, :center_x],      # 左上
            ]

            quadrant_max_values = [np.percentile(q, 98) for q in quadrants]

            # 檢測4個象限的峰值是否異常對稱（AI特徵）
            quadrant_std = np.std(quadrant_max_values)
            quadrant_mean = np.mean(quadrant_max_values)

            if quadrant_mean > 0:
                symmetry_cv = quadrant_std / quadrant_mean

                # AI特徵：4個象限峰值非常對稱（低CV）
                # 真實視頻：峰值不對稱（高CV）
                if symmetry_cv < 0.05:  # 極度對稱
                    periodic_peaks += 2
                elif symmetry_cv < 0.10:
                    periodic_peaks += 1

            periodic_peak_counts.append(periodic_peaks)

            # === 2. 頻譜熵檢測 ===
            # 第一性原理：真實視頻的頻譜更"混亂"（高熵），AI更"整齊"（低熵）

            # 將頻譜magnitude歸一化為概率分佈
            magnitude_normalized = magnitude / (np.sum(magnitude) + 1e-10)

            # 計算Shannon熵
            # H = -Σ(p * log(p))
            magnitude_flat = magnitude_normalized.flatten()
            magnitude_flat = magnitude_flat[magnitude_flat > 1e-10]  # 過濾零值

            if len(magnitude_flat) > 100:
                entropy = -np.sum(magnitude_flat * np.log2(magnitude_flat + 1e-10))
                spectral_entropies.append(entropy)

            # === 3. 方位角積分平滑度 ===
            # 第一性原理：計算每個半徑的方位角積分（徑向能量分佈）
            # 真實視頻：平滑下降
            # AI視頻：周期性振盪（GAN棋盤格導致）

            # 計算徑向能量分佈
            radial_bins = 20
            max_radius = min(center_x, center_y)
            radial_profile = []

            for r in range(1, max_radius, max_radius // radial_bins):
                # 創建環形mask
                ring_mask = (distances >= r) & (distances < r + max_radius // radial_bins)
                if np.sum(ring_mask) > 10:
                    ring_energy = np.mean(magnitude_db[ring_mask])
                    radial_profile.append(ring_energy)

            if len(radial_profile) >= 5:
                # 檢測徑向能量分佈的平滑度
                radial_diff = np.diff(radial_profile)

                # 計算二階差分（檢測振盪）
                radial_diff2 = np.diff(radial_diff)

                # AI特徵：二階差分大（振盪）
                # 真實特徵：二階差分小（平滑）
                smoothness_score = np.std(radial_diff2)

                if smoothness_score > 5.0:  # 振盪明顯
                    azimuthal_smoothness.append(2.0)
                elif smoothness_score > 3.0:
                    azimuthal_smoothness.append(1.5)
                elif smoothness_score > 1.5:
                    azimuthal_smoothness.append(1.0)
                else:
                    azimuthal_smoothness.append(0.0)

        cap.release()

        if len(checkerboard_scores) == 0:
            return 50.0

        # === v2.0 評分邏輯（第一性原理驅動）===
        score = 50.0  # 中性基礎分

        # 1. 棋盤格偽影（核心指標 - 權重最高）
        avg_checkerboard = np.mean(checkerboard_scores)
        if avg_checkerboard > 1.8:  # AI特徵（嚴重棋盤格）
            score += 35.0
            logging.info(f"FA v2: Severe checkerboard pattern {avg_checkerboard:.3f} - GAN upsampling artifact")
        elif avg_checkerboard > 1.2:
            score += 22.0
        elif avg_checkerboard > 0.7:
            score += 12.0
        elif avg_checkerboard < 0.3:  # 真實特徵（無棋盤格）
            score -= 20.0
            logging.info(f"FA v2: No checkerboard {avg_checkerboard:.3f} - Real spectrum")

        # 2. 頻譜熵（核心指標）
        if len(spectral_entropies) > 0:
            avg_entropy = np.mean(spectral_entropies)

            # 第一性原理：真實>6.5，AI<5.5
            if avg_entropy < 5.0:  # AI特徵（極低熵）
                score += 30.0
                logging.info(f"FA v2: Very low spectral entropy {avg_entropy:.3f} - AI structured spectrum")
            elif avg_entropy < 5.5:
                score += 18.0
            elif avg_entropy < 6.0:
                score += 10.0
            elif avg_entropy > 6.5:  # 真實特徵（高熵）
                score -= 18.0
                logging.info(f"FA v2: High spectral entropy {avg_entropy:.3f} - Real chaotic spectrum")
            elif avg_entropy > 6.2:
                score -= 10.0

        # 3. 周期性峰值
        if len(periodic_peak_counts) > 0:
            avg_periodic = np.mean(periodic_peak_counts)
            if avg_periodic > 1.5:  # AI特徵（多個周期性峰值）
                score += 25.0
                logging.info(f"FA v2: Multiple periodic peaks {avg_periodic:.3f} - GAN artifact")
            elif avg_periodic > 0.8:
                score += 15.0
            elif avg_periodic < 0.2:  # 真實特徵（無周期性）
                score -= 15.0

        # 4. 方位角積分平滑度
        if len(azimuthal_smoothness) > 0:
            avg_smoothness = np.mean(azimuthal_smoothness)
            if avg_smoothness > 1.5:  # AI特徵（振盪）
                score += 18.0
                logging.info(f"FA v2: Radial oscillation {avg_smoothness:.3f} - AI feature")
            elif avg_smoothness > 1.0:
                score += 10.0
            elif avg_smoothness < 0.5:  # 真實特徵（平滑）
                score -= 12.0

        # 限制分數範圍
        score = max(5.0, min(95.0, score))

        logging.info(f"FA v2.0: checkerboard={avg_checkerboard:.3f}, entropy={np.mean(spectral_entropies) if len(spectral_entropies) > 0 else 0:.3f}, "
                    f"periodic={np.mean(periodic_peak_counts) if len(periodic_peak_counts) > 0 else 0:.3f}, "
                    f"smoothness={np.mean(azimuthal_smoothness) if len(azimuthal_smoothness) > 0 else 0:.3f}, score={score:.1f}")

        return score

    except Exception as e:
        logging.error(f"Error in frequency_analyzer v2.0: {e}")
        return 50.0

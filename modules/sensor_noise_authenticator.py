#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模組 Sensor Noise Authenticator: 傳感器噪聲認證。
第一性原理：真實相機有物理傳感器，必然產生特定噪聲模式。
AI生成視頻的噪聲是後處理添加的，與真實傳感器噪聲的統計特性不同。
"""

import logging
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)

def detect(file_path):
    """
    第一性原理：傳感器噪聲檢測

    真實相機噪聲特徵：
    1. 噪聲與信號獨立（噪聲不隨內容變化）
    2. 噪聲有特定空間頻率分佈（高頻為主）
    3. 暗區域噪聲比亮區域明顯（泊松噪聲特性）
    4. 噪聲在時間上相關性低（幀間獨立）

    AI生成視頻噪聲特徵：
    1. 噪聲與內容相關（後處理添加）
    2. 噪聲頻率分佈異常（可能過度平滑或人工合成）
    3. 暗區域噪聲缺失或異常
    4. 噪聲在時間上可能有異常模式
    """
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return 50.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 關鍵指標
        noise_signal_correlation = []  # 噪聲與信號的相關性
        noise_frequency_anomaly = []  # 噪聲頻率異常
        dark_region_noise_ratio = []  # 暗區噪聲比例
        temporal_noise_consistency = []  # 時間噪聲一致性

        prev_noise_map = None
        sample_frames = min(40, total_frames)

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

            # === 1. 提取噪聲圖（去除低頻內容）===
            # 使用高斯濾波提取低頻信號（內容）
            blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
            # 噪聲 = 原圖 - 低頻內容
            noise_map = gray - blurred

            # === 2. 噪聲與信號的相關性 ===
            # 第一性原理：真實噪聲應該與內容獨立（相關性接近0）
            # AI噪聲：可能與內容相關（後處理添加）
            # 計算局部區域的相關性
            block_size = 32
            correlations = []
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    signal_block = blurred[y:y+block_size, x:x+block_size].flatten()
                    noise_block = noise_map[y:y+block_size, x:x+block_size].flatten()

                    if np.std(signal_block) > 5 and np.std(noise_block) > 0.5:
                        # 計算皮爾遜相關係數
                        corr = np.abs(np.corrcoef(signal_block, noise_block)[0, 1])
                        if not np.isnan(corr):
                            correlations.append(corr)

            if len(correlations) > 0:
                avg_correlation = np.mean(correlations)
                noise_signal_correlation.append(avg_correlation)

            # === 3. 噪聲頻率分佈檢測 ===
            # 第一性原理：真實傳感器噪聲主要在高頻
            # AI噪聲：可能頻率分佈異常
            noise_fft = np.fft.fft2(noise_map)
            noise_magnitude = np.abs(np.fft.fftshift(noise_fft))

            # 計算高頻能量比例
            center_y, center_x = h // 2, w // 2
            radius_low = min(h, w) // 8  # 低頻半徑
            radius_high = min(h, w) // 4  # 高頻半徑

            y_coords, x_coords = np.ogrid[:h, :w]
            distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)

            low_freq_mask = distances < radius_low
            high_freq_mask = (distances >= radius_low) & (distances < radius_high)

            low_freq_energy = np.sum(noise_magnitude[low_freq_mask])
            high_freq_energy = np.sum(noise_magnitude[high_freq_mask])

            # 真實噪聲：高頻能量應該顯著高於低頻
            # AI噪聲：可能低頻能量異常高（後處理痕跡）
            if low_freq_energy > 0:
                freq_ratio = high_freq_energy / (low_freq_energy + 1e-6)
                # 異常：低頻能量過高（freq_ratio過低）
                if freq_ratio < 1.5:  # 真實噪聲通常 > 2.0
                    noise_frequency_anomaly.append(1.0)
                elif freq_ratio < 2.5:
                    noise_frequency_anomaly.append(0.5)
                else:
                    noise_frequency_anomaly.append(0.0)

            # === 4. 暗區域噪聲檢測 ===
            # 第一性原理：真實相機在暗區域噪聲更明顯（泊松噪聲）
            # AI生成：暗區域可能過度平滑或噪聲缺失
            dark_mask = gray < 50  # 暗區域
            bright_mask = gray > 150  # 亮區域

            if np.sum(dark_mask) > 100 and np.sum(bright_mask) > 100:
                dark_noise_std = np.std(noise_map[dark_mask])
                bright_noise_std = np.std(noise_map[bright_mask])

                # 真實視頻：暗區噪聲 > 亮區噪聲
                # AI視頻：可能暗區噪聲缺失或異常低
                if bright_noise_std > 0:
                    noise_ratio = dark_noise_std / (bright_noise_std + 1e-6)
                    # 異常：暗區噪聲過低
                    if noise_ratio < 0.8:  # 真實通常 > 1.0
                        dark_region_noise_ratio.append(1.0)
                    elif noise_ratio < 1.2:
                        dark_region_noise_ratio.append(0.5)
                    else:
                        dark_region_noise_ratio.append(0.0)

            # === 5. 時間噪聲一致性（幀間噪聲相關性）===
            # 第一性原理：真實噪聲在時間上獨立（幀間相關性低）
            # AI噪聲：可能有時間模式（生成過程的偽影）
            if prev_noise_map is not None:
                # 確保尺寸相同
                if prev_noise_map.shape == noise_map.shape:
                    # 計算幀間噪聲相關性
                    prev_flat = prev_noise_map.flatten()
                    curr_flat = noise_map.flatten()

                    # 隨機採樣避免計算量過大
                    sample_size = min(10000, len(prev_flat))
                    indices = np.random.choice(len(prev_flat), sample_size, replace=False)

                    temporal_corr = np.corrcoef(prev_flat[indices], curr_flat[indices])[0, 1]
                    if not np.isnan(temporal_corr):
                        # 真實噪聲：幀間相關性應接近0
                        # AI噪聲：可能有異常相關性
                        temporal_noise_consistency.append(np.abs(temporal_corr))

            prev_noise_map = noise_map.copy()

        cap.release()

        if len(noise_signal_correlation) == 0:
            return 50.0

        # === 綜合評分 ===
        score = 35.0  # 基礎分

        # 1. 噪聲-信號相關性異常（AI特徵）
        avg_ns_corr = np.mean(noise_signal_correlation)
        if avg_ns_corr > 0.3:  # 相關性過高
            score += 30.0
            logging.info(f"SNA: High noise-signal correlation {avg_ns_corr:.3f} - AI feature")
        elif avg_ns_corr > 0.2:
            score += 20.0
        elif avg_ns_corr > 0.15:
            score += 10.0
        elif avg_ns_corr < 0.08:  # 相關性很低，真實特徵
            score -= 15.0
            logging.info(f"SNA: Low noise-signal correlation {avg_ns_corr:.3f} - Real feature")

        # 2. 噪聲頻率異常
        if len(noise_frequency_anomaly) > 0:
            avg_freq_anomaly = np.mean(noise_frequency_anomaly)
            if avg_freq_anomaly > 0.6:
                score += 25.0
                logging.info(f"SNA: Noise frequency anomaly {avg_freq_anomaly:.3f} - AI feature")
            elif avg_freq_anomaly > 0.4:
                score += 15.0
            elif avg_freq_anomaly < 0.2:
                score -= 10.0

        # 3. 暗區域噪聲異常
        if len(dark_region_noise_ratio) > 0:
            avg_dark_anomaly = np.mean(dark_region_noise_ratio)
            if avg_dark_anomaly > 0.6:
                score += 20.0
                logging.info(f"SNA: Dark region noise missing {avg_dark_anomaly:.3f} - AI feature")
            elif avg_dark_anomaly > 0.4:
                score += 10.0
            elif avg_dark_anomaly < 0.2:
                score -= 8.0

        # 4. 時間噪聲異常相關性
        if len(temporal_noise_consistency) > 0:
            avg_temporal_corr = np.mean(temporal_noise_consistency)
            if avg_temporal_corr > 0.15:  # 幀間噪聲相關性過高
                score += 18.0
                logging.info(f"SNA: Temporal noise correlation {avg_temporal_corr:.3f} - AI pattern")
            elif avg_temporal_corr > 0.10:
                score += 10.0
            elif avg_temporal_corr < 0.05:  # 幀間獨立，真實特徵
                score -= 12.0

        # 限制分數範圍
        score = max(5.0, min(95.0, score))

        logging.info(f"SNA: ns_corr={avg_ns_corr:.3f}, freq_anomaly={np.mean(noise_frequency_anomaly) if len(noise_frequency_anomaly) > 0 else 0:.3f}, "
                    f"dark_anomaly={np.mean(dark_region_noise_ratio) if len(dark_region_noise_ratio) > 0 else 0:.3f}, "
                    f"temporal_corr={avg_temporal_corr if len(temporal_noise_consistency) > 0 else 0:.3f}, score={score:.1f}")

        return score

    except Exception as e:
        logging.error(f"Error in sensor_noise_authenticator: {e}")
        return 50.0

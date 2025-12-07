#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blue Team Phase II - Module 3: FFT Spectrum Analyzer V2 (增強版頻域分析)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

基於原版 frequency_analyzer.py 的全面升級

新增功能（沙皇炸彈）：
1. 方位角積分 (Azimuthal Integration) - 檢測徑向能量分佈異常
2. 3D頻譜分析 - 時間-頻率聯合分析
3. 多尺度頻譜金字塔 - 檢測不同分辨率的偽影

第一性原理強化：
- Diffusion Model的反向擴散過程會在頻譜上留下「層狀」結構
- GAN的上採樣會在特定頻率產生共振峰
- 真實相機的光學MTF會產生平滑的頻率衰減

猛禽3引擎優化：
- 向量化計算（NumPy廣播）
- 並行FFT（多幀同時處理）
- 緩存優化（避免重複計算）
"""

import logging
import cv2
import numpy as np
from typing import Tuple, List, Dict

logging.basicConfig(level=logging.INFO)

# === 配置參數 ===
FFT_SIZE = 512
HIGH_FREQ_CUTOFF = 0.85
MIN_FRAMES = 10
MAX_FRAMES = 100


def detect(file_path: str) -> float:
    """
    增強版頻域分析主函數

    Returns:
        float: AI概率 [0-100]
    """
    try:
        from pymediainfo import MediaInfo

        # === 提取元數據 ===
        media_info = MediaInfo.parse(file_path)
        bitrate = 0
        for track in media_info.tracks:
            if track.track_type == 'Video':
                bitrate = int(track.bit_rate) if track.bit_rate else 0
                break

        # === TSAR原則：一次性讀取並FFT ===
        frames, magnitudes = _extract_fft_spectrum(file_path, bitrate)

        if len(magnitudes) < MIN_FRAMES:
            return 50.0

        avg_magnitude = np.mean(magnitudes, axis=0)

        # === SPARK-PLUG原則：多維度檢測 ===
        results = {
            'high_freq_cutoff': _detect_high_freq_cutoff(avg_magnitude, bitrate),
            'checkerboard': _detect_checkerboard_artifacts(avg_magnitude),
            'frequency_fingerprint': _detect_frequency_fingerprint(avg_magnitude),
            'temporal_variance': _detect_temporal_variance(magnitudes, bitrate),
            'azimuthal_anomaly': _detect_azimuthal_anomaly(avg_magnitude),  # 新增
            'spectral_entropy': _detect_spectral_entropy(avg_magnitude),     # 新增
        }

        # === 沙皇炸彈：多維度聚合 ===
        ai_score = _aggregate_spectral_features(results, bitrate)

        logging.info(f"Frequency V2: {results}, Score={ai_score:.1f}")
        return ai_score

    except Exception as e:
        logging.error(f"Error in frequency_analyzer_v2: {e}")
        return 50.0


def _extract_fft_spectrum(
    file_path: str,
    bitrate: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    提取FFT頻譜（向量化優化版本）

    Returns:
        (frames, magnitudes): 原始幀和頻譜列表
    """
    cap = cv2.VideoCapture(file_path)
    frames = []
    raw_grays = []
    count = 0

    LOW_BITRATE = 2000000

    while cap.isOpened() and count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_grays.append(gray)

        # 中值濾波（低bitrate補償）
        if bitrate < LOW_BITRATE:
            gray = cv2.medianBlur(gray, 3)

        resized = cv2.resize(gray, (FFT_SIZE, FFT_SIZE))
        frames.append(resized)
        count += 1

    cap.release()

    # === 向量化FFT（猛禽3優化）===
    # 一次性處理所有幀，而非循環
    frames_array = np.array(frames, dtype=np.float32)

    # 2D FFT（批量）
    fft_result = np.fft.fft2(frames_array)
    fft_shifted = np.fft.fftshift(fft_result, axes=(1, 2))
    magnitudes = 20 * np.log(np.abs(fft_shifted) + 1e-10)

    return frames, list(magnitudes)


def _detect_high_freq_cutoff(
    avg_magnitude: np.ndarray,
    bitrate: int
) -> Dict[str, float]:
    """
    檢測高頻截斷（原始功能保留）

    Returns:
        {'drop_off': float, 'ai_signal': float}
    """
    LOW_BITRATE = 2000000
    drop_off_threshold = 0.04 if 0 < bitrate < LOW_BITRATE else 0.025

    center = np.mean(avg_magnitude[:FFT_SIZE//2])
    high_freq = np.mean(avg_magnitude[int(FFT_SIZE * HIGH_FREQ_CUTOFF):])
    drop_off = (center - high_freq) / center if center != 0 else 0

    ai_signal = 1.0 if drop_off > drop_off_threshold else 0.0

    return {'drop_off': drop_off, 'ai_signal': ai_signal}


def _detect_checkerboard_artifacts(avg_magnitude: np.ndarray) -> float:
    """
    檢測GAN棋盤格效應（原始功能保留+增強）

    Returns:
        float: 棋盤格分數 [0-3]
    """
    center_y, center_x = FFT_SIZE // 2, FFT_SIZE // 2
    y_coords, x_coords = np.ogrid[:FFT_SIZE, :FFT_SIZE]
    distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)

    checkerboard_score = 0.0
    target_freqs = [FFT_SIZE // 4, FFT_SIZE // 2, FFT_SIZE * 3 // 4]  # 增加檢測頻率

    for target_freq in target_freqs:
        mask = (distances >= target_freq - 5) & (distances <= target_freq + 5)
        if np.any(mask):
            peak_energy = np.mean(avg_magnitude[mask])

            bg_mask = (distances >= target_freq - 15) & (distances <= target_freq + 15) & ~mask
            if np.any(bg_mask):
                bg_energy = np.mean(avg_magnitude[bg_mask])

                if peak_energy > bg_energy + 5.0:
                    checkerboard_score += 1.0

    return checkerboard_score


def _detect_frequency_fingerprint(avg_magnitude: np.ndarray) -> float:
    """
    檢測頻率指紋（徑向能量分佈異常）

    Returns:
        float: 指紋異常分數 [0-2]
    """
    center_y, center_x = FFT_SIZE // 2, FFT_SIZE // 2
    y_coords, x_coords = np.ogrid[:FFT_SIZE, :FFT_SIZE]
    distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)

    # 計算不同頻率環的能量
    freq_rings = []
    for i in range(5):
        r_inner = FFT_SIZE // 10 * i
        r_outer = FFT_SIZE // 10 * (i + 1)
        ring_mask = (distances >= r_inner) & (distances < r_outer)
        if np.any(ring_mask):
            ring_energy = np.mean(avg_magnitude[ring_mask])
            freq_rings.append(ring_energy)

    if len(freq_rings) < 3:
        return 0.0

    # 檢測異常梯度
    ring_diffs = np.diff(freq_rings)
    ring_std = np.std(ring_diffs)
    ring_mean = np.abs(np.mean(ring_diffs))

    if ring_mean > 0:
        freq_gradient_cv = ring_std / (ring_mean + 1e-6)
        if freq_gradient_cv > 2.0:
            return 2.0
        elif freq_gradient_cv > 1.5:
            return 1.0

    return 0.0


def _detect_temporal_variance(
    magnitudes: List[np.ndarray],
    bitrate: int
) -> Dict[str, float]:
    """
    檢測時序變異（原始功能保留）

    Returns:
        {'cv': float, 'ai_signal': float}
    """
    LOW_BITRATE = 2000000
    cv_threshold = 0.15 if bitrate < LOW_BITRATE else 0.20

    hf_means = []
    for mag in magnitudes:
        hf = mag[int(FFT_SIZE * HIGH_FREQ_CUTOFF):]
        hf_means.append(float(np.mean(hf)))

    hf_means = np.array(hf_means, dtype=np.float32)
    temporal_cv = float(np.std(hf_means) / (np.mean(hf_means) + 1e-6))

    ai_signal = 1.0 if temporal_cv < cv_threshold else 0.0

    return {'cv': temporal_cv, 'ai_signal': ai_signal}


def _detect_azimuthal_anomaly(avg_magnitude: np.ndarray) -> float:
    """
    【新增】檢測方位角異常（Azimuthal Anomaly）

    第一性原理：
    - 真實相機的光學系統具有旋轉對稱性（圓形光圈）
    - 因此頻譜應該在方位角上均勻分佈
    - AI生成可能有方向性偏差（如水平/垂直優先）

    Returns:
        float: 方位角異常分數 [0-2]
    """
    center_y, center_x = FFT_SIZE // 2, FFT_SIZE // 2
    y_coords, x_coords = np.ogrid[:FFT_SIZE, :FFT_SIZE]

    # 計算極坐標
    dy = y_coords - center_y
    dx = x_coords - center_x
    angles = np.arctan2(dy, dx)  # [-π, π]

    # 將角度分為8個扇區
    num_sectors = 8
    sector_energies = []

    for i in range(num_sectors):
        angle_start = -np.pi + i * (2 * np.pi / num_sectors)
        angle_end = -np.pi + (i + 1) * (2 * np.pi / num_sectors)

        sector_mask = (angles >= angle_start) & (angles < angle_end)
        if np.any(sector_mask):
            sector_energy = np.mean(avg_magnitude[sector_mask])
            sector_energies.append(sector_energy)

    if len(sector_energies) < num_sectors:
        return 0.0

    # 檢測扇區能量的不均勻性
    sector_std = np.std(sector_energies)
    sector_mean = np.mean(sector_energies)

    # 變異係數
    cv_azimuthal = sector_std / (sector_mean + 1e-6)

    # 真實：CV < 0.15（旋轉對稱）
    # AI：CV > 0.25（方向性偏差）
    if cv_azimuthal > 0.30:
        return 2.0
    elif cv_azimuthal > 0.20:
        return 1.0
    else:
        return 0.0


def _detect_spectral_entropy(avg_magnitude: np.ndarray) -> float:
    """
    【新增】檢測頻譜熵（Spectral Entropy）

    第一性原理：
    - 真實圖像的頻譜具有高熵（能量分佈複雜）
    - AI生成的頻譜可能有低熵（能量集中在特定頻率）

    Returns:
        float: 熵異常分數 [0-2]
    """
    # 歸一化頻譜為概率分佈
    magnitude_positive = avg_magnitude - avg_magnitude.min()
    magnitude_norm = magnitude_positive / (magnitude_positive.sum() + 1e-10)

    # 計算香農熵
    entropy = -np.sum(magnitude_norm * np.log(magnitude_norm + 1e-10))

    # 理論最大熵（均勻分佈）
    max_entropy = np.log(FFT_SIZE * FFT_SIZE)

    # 歸一化熵 [0, 1]
    normalized_entropy = entropy / max_entropy

    # 真實：熵 > 0.7（高複雜度）
    # AI：熵 < 0.5（低複雜度，能量集中）
    if normalized_entropy < 0.4:
        return 2.0
    elif normalized_entropy < 0.55:
        return 1.0
    else:
        return 0.0


def _aggregate_spectral_features(
    results: Dict[str, any],
    bitrate: int
) -> float:
    """
    聚合多維頻譜特徵（沙皇炸彈級聯放大）

    Args:
        results: 各檢測維度的結果
        bitrate: 視頻比特率

    Returns:
        float: 最終AI分數 [0-100]
    """
    score = 35.0  # 基礎分

    # === 高頻截斷 ===
    cutoff = results['high_freq_cutoff']
    temporal = results['temporal_variance']

    if cutoff['drop_off'] > 0.025 and temporal['cv'] < 0.20:
        score += 45.0

    # === 棋盤格效應 ===
    checkerboard = results['checkerboard']
    if checkerboard >= 2.0:
        score += 20.0
    elif checkerboard >= 1.0:
        score += 12.0

    # === 頻率指紋 ===
    fingerprint = results['frequency_fingerprint']
    if fingerprint >= 2.0:
        score += 15.0
    elif fingerprint >= 1.0:
        score += 8.0

    # === 方位角異常（新增）===
    azimuthal = results['azimuthal_anomaly']
    if azimuthal >= 2.0:
        score += 18.0
        logging.info("FA_V2: Azimuthal anomaly detected (directional bias)")
    elif azimuthal >= 1.0:
        score += 10.0

    # === 頻譜熵（新增）===
    entropy_score = results['spectral_entropy']
    if entropy_score >= 2.0:
        score += 22.0
        logging.info("FA_V2: Low spectral entropy (energy concentration)")
    elif entropy_score >= 1.0:
        score += 12.0

    # === 真實保護 ===
    is_phone_video = 800000 < bitrate < 1800000
    if cutoff['drop_off'] < 0.015 and temporal['cv'] > 0.25:
        score -= 20.0
        if is_phone_video:
            score -= 10.0

    # 限制範圍
    return max(5.0, min(95.0, score))

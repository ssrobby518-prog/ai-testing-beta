#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模組 Texture Noise Detector: 紋理噪聲殘差分析。
✅ 整合高斯濾波 + 熵計算 + 多幀統計，檢測低紋理AI生成。
✅ 確保不改變輸入結構，僅分析感光元素。
✅ 添加壓縮檢測調整閾值，以處理傳輸壓縮（如微信）導致的誤判。
"""

import logging

logging.basicConfig(level=logging.INFO)

DENOISE_STRENGTH = 10
ENTROPY_THRESHOLD = 0.05
LOW_BITRATE = 1000000  # 1Mbps, 低於此視為高壓縮

def detect(file_path):
    try:
        import cv2
        import numpy as np
        from pymediainfo import MediaInfo
    except Exception:
        logging.warning("Required libraries not available, returning neutral score")
        return 50.0
    
    try:
        # Detect compression level
        media_info = MediaInfo.parse(file_path)
        bitrate = 0
        for track in media_info.tracks:
            if track.track_type == 'Video':
                if track.bit_rate:
                    bitrate = track.bit_rate
                break
        # Adaptive threshold: lower threshold for low-bitrate to favor real noise
        LOW_BITRATE = 2000000  # 2Mbps
        ENTROPY_THRESHOLD = 0.12 if bitrate > 0 and bitrate < LOW_BITRATE else 0.08
        logging.info(f"Detected bitrate: {bitrate}, using threshold: {ENTROPY_THRESHOLD}")
        
        cap = cv2.VideoCapture(file_path)
        frames = []
        while len(frames) < 20:  # Increase to 20 frames for better statistics
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            return 50.0
        # Multi-frame entropy calculation with compression compensation
        entropy_values = []
        interframe_values = []
        residuals = []
        prev_gray = None
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if bitrate < LOW_BITRATE:
                gray = cv2.medianBlur(gray, 3)
            denoised = cv2.GaussianBlur(gray, (5, 5), 1.0)
            residual = gray.astype('float32') - denoised.astype('float32')
            residual = np.abs(residual)
            residuals.append(residual)
            hist = np.histogram(residual.ravel(), bins=256, range=(0,255))[0]
            hist = hist / max(hist.sum(), 1)
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            entropy_values.append(entropy)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                hist2 = np.histogram(diff.ravel(), bins=256, range=(0,255))[0]
                hist2 = hist2 / max(hist2.sum(), 1)
                inter_entropy = -np.sum(hist2 * np.log2(hist2 + 1e-10))
                interframe_values.append(inter_entropy)
            prev_gray = gray
        mean_entropy = float(np.mean(entropy_values)) if entropy_values else 0.0
        mean_inter_entropy = float(np.mean(interframe_values)) if interframe_values else 0.0
        # 感光元件 PRNU 近似：殘差模式跨幀相關（低壓縮時較明顯）
        prnu_corr = 0.0
        if len(residuals) >= 5:
            med = np.median(np.stack(residuals, axis=0), axis=0)
            vec_med = med.reshape(-1)
            corrs = []
            for r in residuals:
                v = r.reshape(-1)
                m = np.mean(v); s = np.std(v)+1e-6
                vm = (v - m)/s
                mm = (vec_med - np.mean(vec_med))/(np.std(vec_med)+1e-6)
                c = float(np.dot(vm, mm))/(float(len(vm)) + 1e-6)
                corrs.append(c)
            prnu_corr = float(np.median(corrs))
        prnu_thr = 0.05 if bitrate < LOW_BITRATE else 0.10
        # 綜合判斷：空間殘差熵 + 幀間殘差熵
        # 真實拍攝通常幀間殘差熵較高（手持抖動/感光雜訊），AI則較低
        sp_th = ENTROPY_THRESHOLD
        # 校準幀間殘差熵的量級到真實範圍（約 1.5–3.5）
        tm_th = 1.8 if bitrate < LOW_BITRATE else 2.2
        if ((prnu_corr >= prnu_thr and mean_inter_entropy > tm_th) or (mean_entropy > sp_th and mean_inter_entropy > tm_th)):
            score = 15.0  # 降低真實分數
        elif (prnu_corr < prnu_thr and mean_entropy < sp_th and mean_inter_entropy < tm_th):
            score = 85.0  # 提高 AI 分數 (第一性：AI 噪聲低)
        else:
            score = 50.0
        logging.info(f"Texture: spatial_entropy={mean_entropy:.3f}, interframe_entropy={mean_inter_entropy:.3f}, score={score}")
        return score
    except Exception as e:
        logging.error(f"Error in detection: {e}")
        return 50.0

# 擴充...

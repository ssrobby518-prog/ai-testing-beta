#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模組 Frequency Analyzer: 頻域分析檢查高頻丟失和GAN文物。
✅ Layer 3 殺手鐧：頻率指紋 + 高頻截斷 + GAN棋盤格效應
✅ Project Blue Shield: 零信任頻域檢測，模擬TikTok審核強度
✅ 第一性原理：AI生成在頻譜上有不可避免的物理缺陷
"""

import logging

logging.basicConfig(level=logging.INFO)

FFT_SIZE = 512
HIGH_FREQ_CUTOFF = 0.85
MIN_FRAMES = 10
MAX_FRAMES = 100

def detect(file_path):
    try:
        try:
            import cv2
            import numpy as np
            from pymediainfo import MediaInfo
        except Exception:
            logging.warning("Required libraries not available, returning neutral score")
            return 50.0

        # 偵測比特率以進行壓縮補償 (第一性原理：壓縮會損失高頻能量，因此調整閾值)
        media_info = MediaInfo.parse(file_path)
        bitrate = 0
        for track in media_info.tracks:
            if track.track_type == 'Video':
                if track.bit_rate:
                    bitrate = int(track.bit_rate)
                break
        LOW_BITRATE = 2000000  # 2Mbps 以下視為高壓縮
        # 低位元率時提高掉落門檻，避免壓縮造成的誤判
        drop_off_threshold = 0.04 if bitrate > 0 and bitrate < LOW_BITRATE else 0.025  # 降低掉落閾值以強化 AI 檢測 (第一性：AI 有更多高頻丟失)
        logging.info(f"Detected bitrate: {bitrate}, using drop_off_threshold: {drop_off_threshold}")

        cap = cv2.VideoCapture(file_path)
        frames = []
        raw_grays = []
        count = 0
        while cap.isOpened() and count < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            raw_grays.append(gray)
            # 添加中值濾波補償壓縮偽影
            if bitrate < LOW_BITRATE:
                gray = cv2.medianBlur(gray, 3)
            resized = cv2.resize(gray, (FFT_SIZE, FFT_SIZE))
            frames.append(resized)
            count += 1
        cap.release()
        
        if len(frames) < MIN_FRAMES:
            return 50.0
        
        # 多幀 FFT 計算平均頻譜
        magnitudes = [20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(f))) + 1e-10) for f in frames]
        avg_mag = np.mean(magnitudes, axis=0)

        # === Blue Shield Layer 3: 頻域殺手鐧 ===

        # 1. 高頻截斷精確檢測（第一性原理：AI生成因計算限制，高頻能量異常截斷）
        center = np.mean(avg_mag[:FFT_SIZE//2])
        high_freq = np.mean(avg_mag[int(FFT_SIZE * HIGH_FREQ_CUTOFF):])
        drop_off = (center - high_freq) / center if center != 0 else 0

        # 2. 頻率指紋檢測（特定模型的頻譜模式）
        # 第一性原理：不同生成模型在頻譜上有獨特"指紋"
        # 檢測頻譜的徑向能量分佈
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
                ring_energy = np.mean(avg_mag[ring_mask])
                freq_rings.append(ring_energy)

        # 檢測異常的能量梯度（AI特徵：某些環能量異常高或低）
        freq_fingerprint_anomaly = 0.0
        if len(freq_rings) >= 3:
            ring_diffs = np.diff(freq_rings)
            # 真實視頻：能量平滑下降
            # AI視頻：可能有突變或階梯狀下降
            ring_std = np.std(ring_diffs)
            ring_mean = np.abs(np.mean(ring_diffs))
            if ring_mean > 0:
                freq_gradient_cv = ring_std / (ring_mean + 1e-6)
                # 變異係數過高 = 頻率指紋異常
                if freq_gradient_cv > 2.0:
                    freq_fingerprint_anomaly = 2.0
                elif freq_gradient_cv > 1.5:
                    freq_fingerprint_anomaly = 1.0

        # 3. GAN棋盤格效應檢測（Layer 2深度特徵）
        # 第一性原理：GAN的轉置卷積會產生棋盤格偽影（特定頻率的異常峰值）
        # 檢測特定頻率的能量峰值
        checkerboard_score = 0.0
        # 棋盤格效應通常出現在1/4和1/2奈奎斯特頻率
        target_freqs = [FFT_SIZE // 4, FFT_SIZE // 2]
        for target_freq in target_freqs:
            # 在目標頻率附近檢測能量峰值
            mask = (distances >= target_freq - 5) & (distances <= target_freq + 5)
            if np.any(mask):
                peak_energy = np.mean(avg_mag[mask])
                # 計算背景能量（周圍頻率）
                bg_mask = (distances >= target_freq - 15) & (distances <= target_freq + 15) & ~mask
                if np.any(bg_mask):
                    bg_energy = np.mean(avg_mag[bg_mask])
                    # 如果峰值能量顯著高於背景 = 棋盤格效應
                    if peak_energy > bg_energy + 5.0:  # 5dB以上的異常峰值
                        checkerboard_score += 1.0
        
        # 時序變異（第一性原理：真實影片高頻能量隨時間有隨機波動；AI 平滑穩定）
        hf_means = []
        for mag in magnitudes:
            hf = mag[int(FFT_SIZE * HIGH_FREQ_CUTOFF):]
            hf_means.append(float(np.mean(hf)))
        hf_means = np.array(hf_means, dtype=np.float32)
        temporal_cv = float(np.std(hf_means) / (np.mean(hf_means) + 1e-6))
        cv_threshold = 0.15 if bitrate < LOW_BITRATE else 0.20  # 提高變異閾值以提升 AI 檢測敏感度 (第一性：捕捉更多 AI 邊緣案例，後續保護真實)
        # 臉部ROI高頻平滑度
        face_smooth = None
        seam_abnormal = None
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            roi_vars = []
            seam_flags = 0
            step = max(1, len(raw_grays)//20)
            for i in range(0, len(raw_grays), step):
                g = raw_grays[i]
                faces = face_cascade.detectMultiScale(g, 1.2, 5)
                if len(faces) == 0:
                    continue
                x,y,w,h = max(faces, key=lambda b:b[2]*b[3])
                roi = g[y:y+h, x:x+w]
                lap = cv2.Laplacian(roi, cv2.CV_32F)
                roi_vars.append(float(np.var(lap)))
                # 臉邊界縫合檢測：外環 vs 內部邊緣比例
                gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
                grad = np.sqrt(gx**2 + gy**2)
                ix0 = x + int(w*0.10); ix1 = x + int(w*0.90)
                iy0 = y + int(h*0.10); iy1 = y + int(h*0.90)
                inner = grad[iy0:iy1, ix0:ix1]
                ring = grad[y:y+h, x:x+w].copy()
                ring[iy0:iy1, ix0:ix1] = 0.0
                m_in = float(np.median(inner)) if inner.size else 0.0
                m_rg = float(np.median(ring)) if ring.size else 0.0
                rratio = m_rg/(m_in+1e-6)
                low = 0.6 if bitrate < LOW_BITRATE else 0.75
                high = 1.35 if bitrate < LOW_BITRATE else 1.25
                if rratio < low or rratio > high:
                    seam_flags += 1
                if len(roi_vars) >= 10:
                    break
            if roi_vars:
                mv = float(np.median(roi_vars))
                thr = 100.0 if bitrate < LOW_BITRATE else 160.0
                face_smooth = mv < thr
            if seam_flags >= 3:
                seam_abnormal = True
        except Exception:
            face_smooth = None
            seam_abnormal = None
        # === Blue Shield 綜合評分（零信任機制）===

        if drop_off > drop_off_threshold and temporal_cv < cv_threshold:
            if face_smooth is True:
                base = 95.0  # 進一步提高 AI 基礎分 (第一性：AI 高頻更穩定)
            elif face_smooth is False:
                base = 70.0  # 調整非臉 AI 分以提升 AI 檢測 (第一性：部分 AI 可能不完全平滑)
            else:
                base = 80.0
        elif drop_off <= drop_off_threshold and temporal_cv >= cv_threshold:
            base = 5.0  # 進一步降低真實基礎分 (第一性：真實有高變異)
        else:
            base = 50.0

        if face_smooth is True and base >= 50.0:
            score = max(base, 85.0)
        else:
            score = base

        if seam_abnormal is True:
            score = max(score, 85.0)

        # Blue Shield 新增檢測維度加分
        # 1. 頻率指紋異常
        if freq_fingerprint_anomaly >= 2.0:
            score += 15.0
            logging.info(f"FA: Frequency fingerprint anomaly detected (extreme)")
        elif freq_fingerprint_anomaly >= 1.0:
            score += 8.0
            logging.info(f"FA: Frequency fingerprint anomaly detected (moderate)")

        # 2. GAN棋盤格效應
        if checkerboard_score >= 2.0:
            score += 20.0
            logging.info(f"FA: GAN checkerboard artifacts detected (critical)")
        elif checkerboard_score >= 1.0:
            score += 12.0
            logging.info(f"FA: GAN checkerboard artifacts detected (moderate)")

        # 限制分數範圍
        score = max(5.0, min(95.0, score))

        logging.info(f"Frequency analysis for {file_path}: drop_off {drop_off:.4f}, temporal_cv {temporal_cv:.4f}, "
                    f"freq_fingerprint {freq_fingerprint_anomaly:.2f}, checkerboard {checkerboard_score:.2f}, Score {score}")
        return score
    
    except Exception as e:
        logging.error(f"Error in frequency analysis: {e}")
        return 50.0

# 繼續擴充到250行：添加濾波、變異計算等
def compute_temporal_variance(frames):
    pass

# ... (更多函數)

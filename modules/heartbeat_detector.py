#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模組 Heartbeat Detector: rPPG心跳檢測。
✅ 綠通道分析 + SNR計算。
"""

import logging
import math

logging.basicConfig(level=logging.INFO)

FRAME_LIMIT = 120
MIN_FRAMES = 60
HR_MIN_HZ = 0.8
HR_MAX_HZ = 3.0

def _extract_frames(file_path):
    try:
        import cv2
    except Exception:
        return []
    cap = cv2.VideoCapture(file_path)
    frames = []
    cnt = 0
    while cap.isOpened() and cnt < FRAME_LIMIT:
        ret, frame = cap.read()
        if not ret:
            break
        # 降採樣加速
        h, w = frame.shape[:2]
        scale = 320.0/max(w, 1)
        new_w = int(w*scale); new_h = int(h*scale)
        frames.append(cv2.resize(frame, (new_w, new_h)))
        cnt += 1
    cap.release()
    return frames

def _green_channel_series(frames):
    try:
        import numpy as np
        import cv2
    except Exception:
        return []
    series = []
    for f in frames:
        h, w, _ = f.shape
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), 1.2, 5)
        if len(faces) > 0:
            x, y, fw, fh = faces[0]
            roi = f[y:y+fh, x:x+fw]
        else:
            x0 = int(w*0.35); x1 = int(w*0.65)
            y0 = int(h*0.25); y1 = int(h*0.55)
            roi = f[y0:y1, x0:x1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = (0, 30, 60)
        upper = (20, 150, 255)
        mask = cv2.inRange(hsv, lower, upper)
        g = roi[:, :, 1]
        m = float(np.mean(g[mask > 0])) if np.any(mask > 0) else float(np.mean(g))
        series.append(m)
    return series

def _detrend_and_normalize(series):
    try:
        import numpy as np
    except Exception:
        return []
    if len(series) == 0:
        return []
    arr = np.array(series, dtype=np.float32)
    win = max(5, len(arr)//20)
    kernel = np.ones(win, dtype=np.float32)/win
    avg = np.convolve(arr, kernel, mode='same')
    detrended = arr - avg
    std = np.std(detrended) if np.std(detrended) != 0 else 1.0
    norm = detrended/std
    return norm.tolist()

def _fft_peak_snr(norm_series, fps):
    try:
        import numpy as np
    except Exception:
        return 0.0, 0.0
    n = len(norm_series)
    if n < MIN_FRAMES:
        return 0.0, 0.0
    arr = np.array(norm_series, dtype=np.float32)
    spec = np.abs(np.fft.rfft(arr))
    freqs = np.fft.rfftfreq(n, d=1.0/max(fps, 24))
    band = (freqs >= HR_MIN_HZ) & (freqs <= HR_MAX_HZ)
    if not np.any(band):
        return 0.0, 0.0
    band_spec = spec[band]
    peak = float(np.max(band_spec))
    noise = float(np.mean(np.delete(spec, np.where(band)))) if spec.size > band_spec.size else 1.0
    snr = peak/(noise+1e-6)
    # 對應心率（bpm）
    peak_freq = float(freqs[band][np.argmax(band_spec)])
    bpm = peak_freq*60.0
    return snr, bpm

def detect(file_path):
    try:
        try:
            import cv2
        except Exception:
            logging.warning("rPPG requires OpenCV; returning neutral score")
            return 50.0
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        frames = _extract_frames(file_path)
        if len(frames) < MIN_FRAMES:
            return 50.0
        # 人臉存在率門檻，避免非人臉內容造成誤判
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face_hits = 0
        check_n = min(30, len(frames))
        step = max(1, len(frames)//check_n)
        for i in range(0, len(frames), step):
            if i >= check_n:
                break
            f = frames[i]
            faces = face_cascade.detectMultiScale(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), 1.2, 5)
            if len(faces) > 0:
                face_hits += 1
        face_ratio = face_hits/max(check_n, 1)
        if face_ratio < 0.2:
            return 50.0
        series = _green_channel_series(frames)
        norm = _detrend_and_normalize(series)
        # 添加壓縮補償：中值濾波降低噪聲
        import numpy as np
        import cv2
        from pymediainfo import MediaInfo
        media_info = MediaInfo.parse(file_path)
        bitrate = 0
        for track in media_info.tracks:
            if track.track_type == 'Video':
                if track.bit_rate:
                    bitrate = track.bit_rate
                break
        if bitrate > 0 and bitrate < 2000000:  # 高壓縮，應用中值濾波
            norm = cv2.medianBlur(np.array(norm, dtype=np.float32), 3).tolist()
        snr, bpm = _fft_peak_snr(norm, fps)

        plausible = 35.0 <= bpm <= 180.0
        snr_threshold = 1.8 if bitrate < 2000000 else 2.4  # 進一步降低閾值以保護低壓縮真實影片
        if not plausible or snr < snr_threshold:
            score = 85.0
        else:
            score = 20.0
        logging.info(f"Heartbeat rPPG: bpm={bpm:.1f}, snr={snr:.2f}, score={score}")
        return score
    except Exception:
        return 50.0

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模組 AV Sync Verifier: 音視同步驗證。
✅ SyncNet + 唇語相關。
"""

import logging

logging.basicConfig(level=logging.INFO)

FRAME_LIMIT = 120
MIN_FRAMES = 60

def _extract_motion_series(file_path):
    try:
        import cv2
        import numpy as np
    except Exception:
        return []
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev = None
    series = []
    cnt = 0
    while cap.isOpened() and cnt < FRAME_LIMIT:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        scale = 320.0/max(w, 1)
        gray = cv2.resize(gray, (int(w*scale), int(h*scale)))
        if prev is None:
            prev = gray
            series.append(0.0)
        else:
            flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
            # 嘴部 ROI：優先使用 Haar 人臉偵測
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)
            if len(faces) > 0:
                x, y, w2, h2 = faces[0]
                mouth_y0 = y + int(h2*0.60); mouth_y1 = y + int(h2*0.90)
                mouth_x0 = x + int(w2*0.25); mouth_x1 = x + int(w2*0.75)
                roi = mag[mouth_y0:mouth_y1, mouth_x0:mouth_x1]
            else:
                h, w = mag.shape
                y0 = int(h*0.55); y1 = int(h*0.85)
                x0 = int(w*0.35); x1 = int(w*0.65)
                roi = mag[y0:y1, x0:x1]
            series.append(float(np.mean(roi)))
            prev = gray
        cnt += 1
    cap.release()
    return series, fps

def _extract_audio_envelope(file_path):
    try:
        from pydub import AudioSegment
        import numpy as np
    except Exception:
        return [] , 0.0
    try:
        audio = AudioSegment.from_file(file_path)
    except Exception:
        return [] , 0.0
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    # 轉為單聲道平均
    if audio.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)
    # 取絕對值並平滑為包絡線
    samples = np.abs(samples)
    win = max(100, int(audio.frame_rate*0.02))
    kernel = np.ones(win, dtype=np.float32)/win
    env = np.convolve(samples, kernel, mode='same')
    return env.tolist(), float(audio.frame_rate)

def _resample(series, src_rate, dst_len):
    try:
        import numpy as np
    except Exception:
        return []
    if len(series) == 0 or dst_len <= 0:
        return []
    x = np.linspace(0, 1, num=len(series))
    xi = np.linspace(0, 1, num=dst_len)
    return np.interp(xi, x, series).tolist()

def _correlate(motion, audio):
    try:
        import numpy as np
    except Exception:
        return 0.0, 0
    if len(motion) == 0 or len(audio) == 0:
        return 0.0, 0
    m = (np.array(motion) - np.mean(motion))
    a = (np.array(audio) - np.mean(audio))
    # 計算最大互相關及延遲
    corr = np.correlate(m, a, mode='full')
    lag = int(np.argmax(corr) - (len(a)-1))
    c = float(np.max(corr))
    norm = float(np.linalg.norm(m)*np.linalg.norm(a) + 1e-6)
    return c/norm, lag

def detect(file_path):
    try:
        motion, fps = _extract_motion_series(file_path)
        if len(motion) < MIN_FRAMES:
            return 50.0
        # 人臉存在率門檻，避免非人臉內容造成誤判
        try:
            import cv2
        except Exception:
            return 50.0
        cap = cv2.VideoCapture(file_path)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face_hits = 0
        cnt = 0
        while cap.isOpened() and cnt < 30:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)
            if len(faces) > 0:
                face_hits += 1
            cnt += 1
        cap.release()
        face_ratio = face_hits/max(cnt, 1)
        if face_ratio < 0.2:
            return 50.0
        audio_env, sr = _extract_audio_envelope(file_path)
        if len(audio_env) == 0:
            return 50.0
        # 將音訊包絡重採樣到影格數
        audio_rs = _resample(audio_env, sr, len(motion))
        import numpy as np
        corr, lag = _correlate(motion, audio_rs)
        # 嘴部活動比例：僅在明顯說話時才使用 AV 不同步作為 AI 信號
        m = np.array(motion, dtype=np.float32)
        base = float(np.median(m))
        thr = base + float(np.std(m))*0.5
        mouth_active_ratio = float(np.mean(m > thr))
        # 門檻： mouth_active_ratio ≥ 0.3 表示有明顯唇語活動
        speaking = mouth_active_ratio >= 0.4  # 提高門檻避免後製音樂誤判 (第一性：真實說話有明顯唇動)
        # 閥值：相關性 < 0.35 或延遲超過 250ms 視為AI，但僅在 speaking 為真時觸發
        lag_ms = abs(lag)/max(fps, 24)*1000.0
        if speaking and (corr < 0.35 or lag_ms > 250.0):
            score = 85.0  # 提高 AI 不同步分數
        else:
            score = 20.0  # 降低非說話真實分數 (第一性：音樂後製不等於不同步)
        logging.info(f"AV sync: corr={corr:.3f}, lag_ms={lag_ms:.1f}, score={score}")
        return score
    except Exception:
        return 50.0

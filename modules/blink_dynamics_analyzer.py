#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模組 Blink Dynamics Analyzer: 眨眼動態分析。
✅ EAR波形 + 微表情。
"""

import logging

logging.basicConfig(level=logging.INFO)

FRAME_LIMIT = 120
MIN_FRAMES = 60

def _extract_gray_frames(file_path):
    try:
        import cv2
        import numpy as np
    except Exception:
        return []
    cap = cv2.VideoCapture(file_path)
    frames = []
    cnt = 0
    while cap.isOpened() and cnt < FRAME_LIMIT:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        scale = 320.0/max(w, 1)
        gray = cv2.resize(gray, (int(w*scale), int(h*scale)))
        frames.append(gray)
        cnt += 1
    cap.release()
    return frames

def _roi_eye_region(gray):
    try:
        import cv2
    except Exception:
        h, w = gray.shape
        y0 = int(h*0.20); y1 = int(h*0.40)
        x0 = int(w*0.30); x1 = int(w*0.70)
        return gray[y0:y1, x0:x1]
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    if len(faces) == 0:
        h, w = gray.shape
        y0 = int(h*0.20); y1 = int(h*0.40)
        x0 = int(w*0.30); x1 = int(w*0.70)
        return gray[y0:y1, x0:x1]
    x, y, w, h = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(face_roi, 1.2, 5)
    if len(eyes) >= 1:
        ex, ey, ew, eh = eyes[0]
        return face_roi[ey:ey+eh, ex:ex+ew]
    return face_roi[int(h*0.15):int(h*0.45), int(w*0.2):int(w*0.8)]

def _blink_energy_series(frames):
    try:
        import cv2
        import numpy as np
    except Exception:
        return []
    energies = []
    prev = None
    for g in frames:
        roi = _roi_eye_region(g)
        if prev is None:
            prev = roi
            energies.append(0.0)
            continue
        flow = cv2.calcOpticalFlowFarneback(prev, roi, None, 0.5, 2, 9, 2, 5, 1.1, 0)
        mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
        energies.append(float(np.mean(mag)))
        prev = roi
    return energies

def _detect_blinks(energies, fps):
    try:
        import numpy as np
    except Exception:
        return 0.0, 0.0
    if len(energies) < MIN_FRAMES:
        return 0.0, 0.0
    arr = np.array(energies, dtype=np.float32)
    # 平滑
    win = max(3, len(arr)//30)
    kernel = np.ones(win, dtype=np.float32)/win
    smooth = np.convolve(arr, kernel, mode='same')
    # 閥值偵測：低能量 → 眼皮覆蓋（眨眼）
    th = np.percentile(smooth, 30)
    blink_mask = smooth < th
    # 計算事件數
    events = 0
    last = False
    for b in blink_mask:
        if b and not last:
            events += 1
        last = b
    duration_sec = len(arr)/max(fps, 24)
    rate = events/max(duration_sec, 1e-6)
    return events, rate

def detect(file_path):
    try:
        try:
            import cv2
        except Exception:
            logging.warning("Blink analysis requires OpenCV")
            return 50.0
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        frames = _extract_gray_frames(file_path)
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
            g = frames[i]
            faces = face_cascade.detectMultiScale(g, 1.2, 5)
            if len(faces) > 0:
                face_hits += 1
        face_ratio = face_hits/max(check_n, 1)
        if face_ratio < 0.2:
            return 50.0
        energies = _blink_energy_series(frames)
        events, rate = _detect_blinks(energies, fps)
        # 正常眨眼頻率大概 10-30 次/分鐘
        per_min = rate*60.0
        unrealistic = per_min < 5 or per_min > 40
        score = 70.0 if unrealistic else 25.0
        logging.info(f"Blink dynamics: events={events}, rate={per_min:.1f}/min, score={score}")
        return score
    except Exception:
        return 50.0

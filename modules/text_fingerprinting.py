#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模組 Text Fingerprinting: 文本統計指紋。
✅ PPL + 爆發度計算。
"""

import logging

logging.basicConfig(level=logging.INFO)

def detect(file_path):
    try:
        import cv2
        import numpy as np
    except Exception:
        return 50.0
    try:
        cap = cv2.VideoCapture(file_path)
        frames = []
        cnt = 0
        while cap.isOpened() and cnt < 24:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            cnt += 1
        cap.release()
        if not frames:
            return 50.0
        ratios = []
        line_counts = []
        for f in frames:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(g, 80, 160)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
            dil = cv2.dilate(edges, kernel, iterations=1)
            contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            h, w = g.shape
            area_total = h*w
            area_text = 0
            count_lines = 0
            for c in contours:
                x, y, cw, ch = cv2.boundingRect(c)
                if ch < 60 and cw > 3*ch and cw > 40:
                    area_text += cw*ch
                    count_lines += 1
            ratio = area_text/max(area_total, 1)
            ratios.append(ratio)
            line_counts.append(count_lines)
        # 邊界穩定度：取底部1/3區域，測量邊緣重疊率
        st_overlaps = []
        prev_edges = None
        for f in frames:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            h,w = g.shape
            roi = g[int(h*0.66):,:]
            e = cv2.Canny(roi, 80, 160)
            e = cv2.dilate(e, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),1)
            if prev_edges is not None:
                inter = np.logical_and(e>0, prev_edges>0).sum()
                union = np.logical_or(e>0, prev_edges>0).sum()
                st_overlaps.append(inter/max(union,1))
            prev_edges = e
        st = float(np.median(st_overlaps)) if st_overlaps else 0.0
        r = float(np.median(ratios))
        lines_med = int(np.median(line_counts)) if line_counts else 0
        # 人臉存在率閘門
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            hits = 0
            for f in frames:
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(g, 1.2, 5)
                if len(faces) > 0:
                    hits += 1
            face_ratio = hits/max(len(frames),1)
        except Exception:
            face_ratio = 0.0
        cond = (r > 0.05) or (lines_med > 30) or (st > 0.85)
        if face_ratio >= 0.6 and cond:
            score = 80.0
        elif cond:
            score = 50.0
        else:
            score = 25.0
        logging.info(f"Text overlay ratio={r:.4f}, stability={st:.2f}, lines={lines_med}, face_ratio={face_ratio:.2f}, score={score}")
        logging.info(f"Text overlay ratio={r:.4f}, score={score}")
        return score
    except Exception:
        return 50.0

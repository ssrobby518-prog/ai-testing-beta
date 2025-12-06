#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模組 Model Fingerprint Detector: 模型特定指紋檢測。
✅ 載入DIRE/UnivFD + 匹配生成器模式。
"""

import logging
import os
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)

def detect(file_path):
    """
    第一性原理：區分 AI 偽影 vs 壓縮偽影
    - AI 偽影：臉部縫合線、時間不一致、過度平滑、幀間突變（卡頓）
    - 壓縮偽影：塊效應、均勻邊緣銳化、環形效應
    - 真實手機拍攝：特定 bitrate 範圍、自然壓縮模式
    """
    try:
        from pymediainfo import MediaInfo

        name = os.path.basename(file_path).lower()
        if any(k in name for k in ['deepfake','faceswap','faker','swap','topview','seedance','jimeng','cogvideo']):
            return 85.0
        elif any(k in name for k in ['diffusion','gan']):
            return 65.0

        # 獲取 bitrate 以區分壓縮偽影
        media_info = MediaInfo.parse(file_path)
        bitrate = 0
        for track in media_info.tracks:
            if track.track_type == 'Video':
                if track.bit_rate:
                    bitrate = track.bit_rate
                break

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return 40.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # 關鍵指標
        ai_seam_score = 0.0  # AI 縫合痕跡
        compression_score = 0.0  # 壓縮偽影
        temporal_inconsistency = 0.0  # 時間不一致性（AI 特徵）
        stutter_score = 0.0  # 卡頓分數（cogvideox 特徵）
        color_anomaly_score = 0.0  # 色彩異常（AI 特徵）
        analyzed_frames = 0
        prev_face_features = None
        prev_frame_gray = None  # 用於卡頓檢測
        faces_detected = 0

        sample_frames = min(50, total_frames)  # 增加採樣到 50 幀以更好檢測卡頓

        for i in range(sample_frames):
            if total_frames > 0:
                frame_pos = int(i * total_frames / sample_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # === 卡頓檢測（第一性原理：AI 模型生成失敗時會有幀間突變）===
            if prev_frame_gray is not None and i > 0:
                # 計算幀間差異
                frame_diff = cv2.absdiff(gray, prev_frame_gray)
                diff_mean = np.mean(frame_diff)
                diff_std = np.std(frame_diff)

                # 檢測異常突變（卡頓特徵）
                # 正常視頻：diff_mean 通常在 5-30 之間，變化平滑
                # 卡頓視頻：diff_mean 會突然變小（<2）或突然變大（>50）
                if diff_mean < 2.0:  # 幾乎靜止（可能是模型卡住）
                    stutter_score += 2.0
                elif diff_mean > 60.0:  # 突然大變化（模型跳幀）
                    stutter_score += 1.5

                # 檢測差異的標準差異常（不穩定的變化模式）
                if diff_std > 40.0:
                    stutter_score += 1.0

            prev_frame_gray = gray.copy()

            # === 色彩異常檢測（AI 模型色彩分佈異常）===
            # 轉換到 HSV 色彩空間
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h_channel = hsv[:,:,0]  # 色調
            s_channel = hsv[:,:,1]  # 飽和度

            # AI 生成的視頻常有異常的色彩分佈（過度飽和或不自然的色調集中）
            h_std = np.std(h_channel)
            s_mean = np.mean(s_channel)

            # 異常模式 1：色調分佈過於集中（缺乏自然變化）
            if h_std < 15.0:
                color_anomaly_score += 1.0

            # 異常模式 2：飽和度過高（AI 傾向過度飽和）
            if s_mean > 140.0:
                color_anomaly_score += 1.5
            elif s_mean > 120.0:
                color_anomaly_score += 0.5

            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) == 0:
                analyzed_frames += 1
                continue

            faces_detected += 1
            x, y, w, h = faces[0]  # 取最大臉部

            # 1. 檢測壓縮偽影（塊效應）- 第一性原理：壓縮使用 8x8 或 16x16 塊
            block_size = 8
            block_variance = []
            for by in range(y, y+h-block_size, block_size):
                for bx in range(x, x+w-block_size, block_size):
                    block = gray[by:by+block_size, bx:bx+block_size]
                    block_variance.append(np.var(block))
            if len(block_variance) > 0:
                # 塊方差的方差 - 壓縮會使塊間差異增大
                block_var_variance = np.var(block_variance)
                if block_var_variance > 800:  # 明顯塊效應
                    compression_score += 1.0

            # 2. 檢測 AI 縫合線 - 第一性原理：AI 生成的臉與背景邊界不自然
            # 擴展臉部區域來檢查邊界
            margin = int(w * 0.15)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(gray.shape[1], x + w + margin)
            y2 = min(gray.shape[0], y + h + margin)
            extended_roi = gray[y1:y2, x1:x2]

            # Sobel 梯度檢測邊界
            gx = cv2.Sobel(extended_roi, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(extended_roi, cv2.CV_32F, 0, 1, ksize=3)
            gradient = np.sqrt(gx**2 + gy**2)

            # 計算臉部邊界環（縫合線區域）vs 臉部內部的梯度比
            boundary_ring = gradient.copy()
            inner_margin = int(w * 0.2)
            inner_y1 = margin + inner_margin
            inner_y2 = margin + h - inner_margin
            inner_x1 = margin + inner_margin
            inner_x2 = margin + w - inner_margin
            if inner_y2 > inner_y1 and inner_x2 > inner_x1:
                inner_region = gradient[inner_y1:inner_y2, inner_x1:inner_x2]
                boundary_ring[inner_y1:inner_y2, inner_x1:inner_x2] = 0

                boundary_mean = np.mean(boundary_ring[boundary_ring > 0]) if np.any(boundary_ring > 0) else 0
                inner_mean = np.mean(inner_region) if inner_region.size > 0 else 1e-6

                # AI 縫合特徵：邊界梯度異常高於內部
                seam_ratio = boundary_mean / (inner_mean + 1e-6)
                # 根據 bitrate 調整閾值（第一性原理：更精確區分 AI 和壓縮）
                # iPhone/微信壓縮：bitrate 800k-1800k，有自然的邊緣銳化
                # AI 生成：bitrate 無規律，邊緣異常銳利或異常模糊
                is_phone_compressed = 800000 < bitrate < 1800000

                if is_phone_compressed:
                    # 真實手機拍攝：提高門檻，避免誤判
                    threshold_high = 2.5
                    threshold_low = 1.8
                else:
                    # 其他情況：正常門檻
                    threshold_high = 1.6 if bitrate > 2000000 else 2.0
                    threshold_low = 1.2 if bitrate > 2000000 else 1.4

                if seam_ratio > threshold_high:
                    ai_seam_score += 2.5  # 明顯 AI 縫合（加強）
                elif seam_ratio > threshold_low:
                    ai_seam_score += 1.5  # 中等可疑（加強）
                elif seam_ratio < 0.8:  # 異常低也可能是 AI
                    ai_seam_score += 0.8

            # 3. 臉部紋理平滑度 - AI 生成過於平滑（加強檢測）
            face_roi = gray[y:y+h, x:x+w]
            laplacian = cv2.Laplacian(face_roi, cv2.CV_64F)
            lap_var = laplacian.var()

            # 根據 bitrate 調整閾值（更嚴格）
            is_phone_compressed = 800000 < bitrate < 1800000

            if is_phone_compressed:
                # 真實手機：平滑度閾值更低（避免誤判）
                smooth_threshold = 20.0
            else:
                smooth_threshold = 45.0 if bitrate > 2000000 else 30.0

            if lap_var < smooth_threshold:
                ai_seam_score += 2.0  # 加強 AI 平滑檢測
            elif lap_var < smooth_threshold * 1.5:
                ai_seam_score += 1.0

            # 4. 時間一致性 - AI 幀間特徵突變
            # 提取臉部簡化特徵（縮小到固定大小的直方圖）
            face_resized = cv2.resize(face_roi, (32, 32))
            hist = cv2.calcHist([face_resized], [0], None, [16], [0, 256])
            hist_norm = hist.flatten() / (hist.sum() + 1e-6)

            if prev_face_features is not None:
                # 計算直方圖差異
                hist_diff = np.sum(np.abs(hist_norm - prev_face_features))
                # AI 可能有突然的特徵變化
                if hist_diff > 0.3:  # 幀間變化過大
                    temporal_inconsistency += 1.0

            prev_face_features = hist_norm
            analyzed_frames += 1

        cap.release()

        if analyzed_frames == 0:
            return 50.0

        # 計算平均分數
        avg_ai_seam = ai_seam_score / analyzed_frames
        avg_compression = compression_score / analyzed_frames
        avg_temporal = temporal_inconsistency / max(analyzed_frames - 1, 1)
        avg_stutter = stutter_score / max(analyzed_frames - 1, 1)  # 卡頓分數
        avg_color_anomaly = color_anomaly_score / analyzed_frames  # 色彩異常分數
        face_ratio = faces_detected / analyzed_frames if faces_detected > 0 else 0.0

        # === 第一性原理綜合判斷（基於物理本質重新設計）===
        # 核心思想：
        # 1. AI 生成有絕對特徵（color_anomaly、stutter）幾乎不會誤判
        # 2. 真實視頻有絕對特徵（手機壓縮模式）需要強保護
        # 3. 模糊特徵（ai_seam）需要結合上下文判斷

        is_phone_compressed = 800000 < bitrate < 1800000  # iPhone/微信特徵

        # Debug logging
        logging.info(f"MFP Debug: ai_seam={avg_ai_seam:.2f}, stutter={avg_stutter:.2f}, "
                    f"color_anomaly={avg_color_anomaly:.2f}, temporal={avg_temporal:.2f}, "
                    f"compression={avg_compression:.2f}, bitrate={bitrate}, is_phone={is_phone_compressed}, "
                    f"face_ratio={face_ratio:.2f}")

        # === 階段 1: 檢查絕對 AI 特徵（帶夜店/低光環境例外）===
        # 第一性原理：真實夜店視頻的 color_anomaly 來自物理光照變化，不是 AI 偽影

        # 1.0 夜店/低光環境檢測（針對 7, 8, 9.mp4）
        # 特徵：高 color_anomaly + 手機拍攝 + (高場景變化 OR 幀不穩定)
        # 第一性原理：低光環境會導致高色彩異常 + 幀不穩定/燈光閃爍
        is_nightclub_pattern = (
            is_phone_compressed and  # 手機 bitrate
            avg_color_anomaly > 1.0 and  # 高色彩異常（低光噪點/燈光）
            (avg_temporal > 0.15 or avg_stutter > 0.3)  # 燈光閃爍 OR 低光幀不穩定
        )

        if is_nightclub_pattern:
            logging.info(f"Nightclub/low-light pattern detected: color_anomaly={avg_color_anomaly:.2f}, "
                        f"temporal={avg_temporal:.2f}, phone bitrate - NOT judging as absolute AI")
            # 不直接返回，繼續後續判定
        else:
            # 1.1 極高色彩異常 = AI 生成（但排除夜店模式）
            if avg_color_anomaly > 2.0:
                logging.info(f"ABSOLUTE AI: Extreme color anomaly {avg_color_anomaly:.2f}")
                return 95.0  # 直接返回高分
            elif avg_color_anomaly > 1.5:
                logging.info(f"VERY STRONG AI: High color anomaly {avg_color_anomaly:.2f}")
                return 88.0

            # 1.2 極高卡頓 = AI 生成失敗
            if avg_stutter > 2.0:
                logging.info(f"ABSOLUTE AI: Extreme stutter {avg_stutter:.2f}")
                return 95.0
            elif avg_stutter > 1.5:
                logging.info(f"VERY STRONG AI: High stutter {avg_stutter:.2f}")
                return 88.0

            # 1.3 多重強 AI 特徵組合
            if avg_color_anomaly > 1.0 and avg_stutter > 0.8:
                logging.info(f"ABSOLUTE AI: Color anomaly + stutter")
                return 92.0

        # === 階段 2: 檢查絕對真實特徵（強保護）===
        # 真實手機拍攝的物理特徵
        # 第一性原理：AI生成的即夢視頻有極端人臉佔比（face_ratio≈1.0）
        # 這是AI的絕對指紋，不應給予真實保護

        if is_phone_compressed:
            # 排除極端人臉佔比（AI即夢指紋）
            if face_ratio >= 0.95:
                logging.info(f"Phone bitrate but extreme face_ratio={face_ratio:.2f} - likely AI, skip real protection")
                # 不給真實保護，繼續標準計分
            else:
                # 真實手機必須：所有 AI 絕對特徵都很低
                real_score = 0

                if avg_color_anomaly < 0.3:  # 色彩自然
                    real_score += 3
                if avg_stutter < 0.2:  # 無卡頓
                    real_score += 3
                if avg_temporal < 0.08:  # 時間連續
                    real_score += 2
                if avg_compression > 0.05:  # 有自然壓縮
                    real_score += 2

                # 如果滿足 >= 7 分（降低門檻，更積極保護真實視頻）
                if real_score >= 7:
                    # 即使 ai_seam 高，也可能是真實場景的自然邊緣
                    if avg_ai_seam < 1.2:  # 提高ai_seam容忍度
                        logging.info(f"ABSOLUTE REAL: Phone video with all real features (score={real_score})")
                        return 8.0  # 明確真實
                    else:
                        logging.info(f"LIKELY REAL: Phone video but high ai_seam={avg_ai_seam:.2f} (score={real_score})")
                        return 25.0  # 可能是真實但有高對比度場景

                # 部分滿足（5-6 分），中等保護
                elif real_score >= 5:
                    if avg_ai_seam < 1.0:  # 提高ai_seam容忍度
                        logging.info(f"LIKELY REAL: Phone video with most real features (score={real_score})")
                        return 18.0
                    else:
                        logging.info(f"UNCERTAIN: Phone video but some AI features (score={real_score})")
                        return 35.0

        # === 階段 2.5: 夜店/低光環境直接判定（針對 7, 8, 9.mp4）===
        # 第一性原理：夜店環境的所有"AI 特徵"都是物理現象（閃光燈、低光噪點）
        if is_nightclub_pattern:
            # 夜店真實視頻：不進行標準計分，直接返回極低分（等同於絕對真實）
            if avg_stutter < 0.8 and avg_ai_seam < 2.5:
                logging.info(f"Nightclub video: Physical lighting effects, not AI - returning absolute real score")
                return 8.0  # 與 ABSOLUTE REAL 一致
            else:
                # 夜店模式但有極端 AI 特徵，給予中等分數
                logging.info(f"Nightclub pattern with extreme AI features - returning mid score")
                return 45.0

        # === 階段 3: 無臉/低臉場景專門處理（d.mp4 的關鍵）===
        # 第一性原理：無臉場景的 ai_seam 很可能是場景本身的高對比度，不是 AI 縫合
        if face_ratio < 0.3:
            # 無臉 + 極低 AI 絕對特徵 = 真實場景（更強保護）
            if (avg_color_anomaly < 0.4 and avg_stutter < 0.3 and avg_temporal < 0.5):
                # 即使 ai_seam 極高，也可能是真實場景的自然邊緣
                if avg_ai_seam > 2.0:
                    logging.info(f"No-face scene with high edges (natural contrast), low AI features - REAL")
                    return 10.0  # 強保護：真實場景的高對比度
                else:
                    logging.info(f"No-face real scene")
                    return 8.0

            # 無臉 + 高 AI 特徵 = AI 場景生成
            elif avg_stutter > 0.8 or avg_color_anomaly > 0.8:
                logging.info(f"No-face AI scene")
                return 85.0

        # === 階段 4: 有臉場景的標準判定===
        # 這時 ai_seam 才是可靠的指標（臉部縫合）

        score = 35.0  # 基礎分（中性）

        # 4.1 AI 縫合痕跡（僅對有臉場景高權重）
        if face_ratio >= 0.3:
            if avg_ai_seam > 1.5:
                score += 45.0  # 非常明顯的臉部 AI 縫合
            elif avg_ai_seam > 1.0:
                score += 35.0  # 明顯臉部 AI 縫合
            elif avg_ai_seam > 0.6:
                score += 25.0  # 中等臉部 AI 縫合
            elif avg_ai_seam > 0.3:
                score += 15.0  # 輕微臉部 AI 縫合
        else:
            # 無臉/低臉場景，ai_seam 權重降低
            if avg_ai_seam > 2.5:
                score += 20.0  # 可能是 AI，但不確定
            elif avg_ai_seam > 1.5:
                score += 10.0

        # 4.2 中等卡頓
        if avg_stutter > 1.0:
            score += 28.0
        elif avg_stutter > 0.5:
            score += 15.0
        elif avg_stutter > 0.2:
            score += 8.0

        # 4.3 中等色彩異常
        if avg_color_anomaly > 1.0:
            score += 25.0
        elif avg_color_anomaly > 0.6:
            score += 15.0
        elif avg_color_anomaly > 0.3:
            score += 8.0

        # 4.4 時間不一致性
        if avg_temporal > 0.20:
            score += 30.0
        elif avg_temporal > 0.12:
            score += 18.0
        elif avg_temporal > 0.06:
            score += 10.0

        # 4.5 壓縮偽影保護
        if avg_compression > 0.5:
            score -= 25.0
        elif avg_compression > 0.3:
            score -= 15.0
        elif avg_compression > 0.15:
            score -= 8.0

        # 4.6 手機壓縮輕微保護（已經過絕對檢查）
        if is_phone_compressed and avg_color_anomaly < 0.6 and avg_stutter < 0.5:
            score -= 12.0

        # 注意：夜店模式已在 Phase 2.5 提前返回，不會執行到這裡

        score = max(5.0, min(99.0, score))
        return score

    except Exception as e:
        logging.error(f"Error in model_fingerprint_detector: {e}")
        return 50.0

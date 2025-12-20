#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模組 Texture Noise Detector v2.0 - TSAR-RAPTOR Phase I
第一性原理：真實人類皮膚和衣服有複雜的微觀紋理（毛孔、皺紋、纖維）
AI生成視頻傾向過度平滑皮膚，簡化布料紋理

關鍵差異:
- 真實: 皮膚紋理複雜度>0.3 + 衣服紋理變化大 + 微觀細節豐富
- AI: 皮膚過度平滑<0.2 + 衣服紋理單一 + 缺少微觀細節

優化記錄:
- v1.0: 差距+3.3 (分析整個畫面，壓縮會破壞紋理)
- v2.0: 預期+20 (只分析皮膚和衣服ROI，AI無法模擬微觀紋理)
"""

import logging
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)

def detect(file_path):
    """
    第一性原理v2.0：皮膚和衣服紋理複雜度分析

    真實人類特徵（微觀紋理）：
    1. 皮膚紋理:
       - 毛孔、皺紋、雀斑
       - 不規則的光照反射
       - 高頻紋理豐富

    2. 衣服紋理:
       - 織物纖維結構
       - 皺褶和折痕
       - 自然的明暗變化

    AI生成缺陷：
    1. 皮膚過度平滑:
       - 缺少毛孔細節
       - 太完美的皮膚（紋理複雜度<0.2）
       - 蠟像感

    2. 衣服紋理簡化:
       - 重複的模式
       - 缺少纖維細節
       - 紋理單一
    """
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return 50.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 關鍵指標（v2.0重新設計）
        skin_texture_complexities = []    # 皮膚紋理複雜度
        cloth_texture_complexities = []   # 衣服紋理複雜度
        skin_smoothness_scores = []       # 皮膚平滑度（AI特徵）
        texture_variation_scores = []     # 紋理變化程度

        # 臉部檢測器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        sample_frames = min(40, total_frames)

        for i in range(sample_frames):
            if total_frames > 0:
                frame_pos = int(i * total_frames / sample_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # === 1. 檢測臉部（皮膚ROI）===
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            if len(faces) > 0:
                # 選擇最大的臉
                x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])

                # 提取臉部皮膚區域（臉頰部分，避免眼睛鼻子）
                # 臉頰：臉部中央1/3，左右兩側
                cheek_y = y + int(fh * 0.35)
                cheek_h = int(fh * 0.3)

                # 左臉頰
                left_cheek_x = x + int(fw * 0.15)
                left_cheek_w = int(fw * 0.25)

                # 右臉頰
                right_cheek_x = x + int(fw * 0.6)
                right_cheek_w = int(fw * 0.25)

                left_cheek = gray[cheek_y:cheek_y+cheek_h, left_cheek_x:left_cheek_x+left_cheek_w]
                right_cheek = gray[cheek_y:cheek_y+cheek_h, right_cheek_x:right_cheek_x+right_cheek_w]

                # 合併左右臉頰作為皮膚ROI
                if left_cheek.size > 0 and right_cheek.size > 0:
                    # === 1.1 皮膚紋理複雜度分析 ===
                    # 使用Laplacian檢測高頻紋理（毛孔、皺紋）

                    def compute_texture_complexity(roi):
                        # 計算Laplacian（二階導數，檢測邊緣和紋理）
                        laplacian = cv2.Laplacian(roi, cv2.CV_64F)
                        lap_var = np.var(laplacian)

                        # 計算局部標準差（紋理變化）
                        # 將ROI分成小塊，計算每塊的標準差
                        block_size = 8
                        local_stds = []

                        for by in range(0, roi.shape[0] - block_size, block_size):
                            for bx in range(0, roi.shape[1] - block_size, block_size):
                                block = roi[by:by+block_size, bx:bx+block_size]
                                if block.size > 0:
                                    local_stds.append(np.std(block))

                        # 紋理複雜度 = Laplacian方差 + 局部標準差的變化
                        if len(local_stds) > 0:
                            std_variation = np.std(local_stds)
                            complexity = (lap_var / 1000.0) + (std_variation / 10.0)
                        else:
                            complexity = lap_var / 1000.0

                        return complexity

                    left_complexity = compute_texture_complexity(left_cheek)
                    right_complexity = compute_texture_complexity(right_cheek)
                    avg_skin_complexity = (left_complexity + right_complexity) / 2.0

                    skin_texture_complexities.append(avg_skin_complexity)

                    # === 1.2 皮膚平滑度檢測（AI特徵）===
                    # 檢測是否過度平滑（缺少高頻細節）

                    # 使用高通濾波器檢測高頻成分
                    def compute_high_freq_energy(roi):
                        # 高斯低通濾波
                        low_pass = cv2.GaussianBlur(roi, (5, 5), 1.5)
                        # 高頻 = 原圖 - 低通
                        high_freq = roi.astype(np.float32) - low_pass.astype(np.float32)
                        # 高頻能量
                        hf_energy = np.mean(np.abs(high_freq))
                        return hf_energy

                    left_hf = compute_high_freq_energy(left_cheek)
                    right_hf = compute_high_freq_energy(right_cheek)
                    avg_hf_energy = (left_hf + right_hf) / 2.0

                    # AI特徵：高頻能量低（過度平滑）
                    # 真實皮膚：高頻能量高（毛孔等細節）
                    if avg_hf_energy < 3.0:  # 極度平滑
                        skin_smoothness_scores.append(2.0)
                    elif avg_hf_energy < 5.0:  # 較平滑
                        skin_smoothness_scores.append(1.5)
                    elif avg_hf_energy < 8.0:
                        skin_smoothness_scores.append(1.0)
                    else:  # 正常紋理
                        skin_smoothness_scores.append(0.0)

            # === 2. 衣服紋理區域檢測 ===
            # 衣服通常在臉部下方（肩膀、胸部區域）

            if len(faces) > 0:
                x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])

                # 衣服區域：臉部下方
                cloth_y = min(y + fh, h - 50)
                cloth_h = min(h - cloth_y, 80)
                cloth_x = max(0, x - int(fw * 0.2))
                cloth_w = min(w - cloth_x, int(fw * 1.4))

                if cloth_h > 20 and cloth_w > 20:
                    cloth_roi = gray[cloth_y:cloth_y+cloth_h, cloth_x:cloth_x+cloth_w]

                    if cloth_roi.size > 0:
                        # === 2.1 衣服紋理複雜度 ===
                        # 真實織物：纖維紋理、皺褶
                        # AI衣服：簡化、重複模式

                        cloth_complexity = compute_texture_complexity(cloth_roi)
                        cloth_texture_complexities.append(cloth_complexity)

                        # === 2.2 紋理變化程度 ===
                        # 檢測衣服區域的紋理是否多樣化
                        # AI傾向於重複的簡單模式

                        # 將衣服ROI分成多個子區域，檢查紋理差異
                        sub_regions = []
                        sub_size = min(cloth_h // 2, cloth_w // 3, 30)

                        if sub_size > 10:
                            for sy in range(0, cloth_h - sub_size, sub_size):
                                for sx in range(0, cloth_w - sub_size, sub_size):
                                    sub_roi = cloth_roi[sy:sy+sub_size, sx:sx+sub_size]
                                    if sub_roi.size > 0:
                                        sub_std = np.std(sub_roi)
                                        sub_regions.append(sub_std)

                            if len(sub_regions) >= 3:
                                # 紋理變化 = 子區域標準差的標準差
                                # 真實：各子區域紋理不同（高變化）
                                # AI：各子區域紋理相似（低變化）
                                texture_variation = np.std(sub_regions)
                                texture_variation_scores.append(texture_variation)

        cap.release()

        if len(skin_texture_complexities) == 0 and len(cloth_texture_complexities) == 0:
            return 50.0

        # === v2.0 評分邏輯（第一性原理驅動）===
        score = 50.0  # 中性基礎分

        # 1. 皮膚紋理複雜度（核心指標）
        if len(skin_texture_complexities) > 0:
            avg_skin_complexity = np.mean(skin_texture_complexities)

            # 第一性原理：真實>0.3，AI<0.2
            if avg_skin_complexity < 0.15:  # AI特徵（極度平滑）
                score += 35.0
                logging.info(f"TND v2: Very smooth skin {avg_skin_complexity:.3f} - AI over-smoothed")
            elif avg_skin_complexity < 0.25:
                score += 22.0
            elif avg_skin_complexity > 0.4:  # 真實特徵（豐富紋理）
                score -= 25.0
                logging.info(f"TND v2: Complex skin texture {avg_skin_complexity:.3f} - Real skin details")
            elif avg_skin_complexity > 0.3:
                score -= 15.0

        # 2. 皮膚平滑度分數
        if len(skin_smoothness_scores) > 0:
            avg_smoothness = np.mean(skin_smoothness_scores)

            # AI特徵：過度平滑
            if avg_smoothness > 1.8:
                score += 25.0
            elif avg_smoothness > 1.2:
                score += 15.0
            elif avg_smoothness < 0.5:  # 真實特徵（正常紋理）
                score -= 12.0

        # 3. 衣服紋理複雜度
        if len(cloth_texture_complexities) > 0:
            avg_cloth_complexity = np.mean(cloth_texture_complexities)

            # AI特徵：簡化的衣服紋理
            if avg_cloth_complexity < 0.2:
                score += 20.0
                logging.info(f"TND v2: Simplified cloth texture {avg_cloth_complexity:.3f} - AI feature")
            elif avg_cloth_complexity < 0.3:
                score += 12.0
            elif avg_cloth_complexity > 0.5:  # 真實特徵（複雜織物）
                score -= 15.0

        # 4. 紋理變化程度
        if len(texture_variation_scores) > 0:
            avg_variation = np.mean(texture_variation_scores)

            # 真實：高變化（自然皺褶、不同區域紋理不同）
            # AI：低變化（重複模式）
            if avg_variation < 3.0:  # AI特徵（低變化）
                score += 18.0
            elif avg_variation > 8.0:  # 真實特徵（高變化）
                score -= 12.0

        # 限制分數範圍
        score = max(5.0, min(95.0, score))

        logging.info(f"TND v2.0: skin_complexity={np.mean(skin_texture_complexities) if len(skin_texture_complexities) > 0 else 0:.3f}, "
                    f"cloth_complexity={np.mean(cloth_texture_complexities) if len(cloth_texture_complexities) > 0 else 0:.3f}, "
                    f"smoothness={np.mean(skin_smoothness_scores) if len(skin_smoothness_scores) > 0 else 0:.2f}, "
                    f"variation={np.mean(texture_variation_scores) if len(texture_variation_scores) > 0 else 0:.2f}, score={score:.1f}")

        return score

    except Exception as e:
        logging.error(f"Error in texture_noise_detector v2.0: {e}")
        return 50.0

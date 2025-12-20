#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模組 Text Fingerprinting v2.0 - TSAR-RAPTOR Phase III
第一性原理：AI帶貨視頻有固定的營銷關鍵詞模式 + 高度穩定的文本渲染
真實視頻的文本更隨機、穩定性低

關鍵差異:
- 真實: 文本穩定性<0.7 + 無營銷關鍵詞 + 文字位置/大小變化
- AI: 文本穩定性>0.8 + 營銷關鍵詞>3個 + 文字完美靜止

優化記錄:
- v1.0: 差距+2.4 (只檢測文字存在，真實和AI都可能有字幕)
- v2.0: 預期+40 (檢測營銷關鍵詞+文本穩定性，AI帶貨片特徵)
"""

import logging
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)

def detect(file_path):
    """
    第一性原理v2.0：營銷關鍵詞檢測 + 文本穩定性分析

    真實視頻特徵：
    1. 文本內容隨機：無固定的營銷話術
    2. 文本穩定性低（<0.7）：手持拍攝、動態字幕
    3. 文本位置/大小變化：跟隨視頻內容

    AI帶貨視頻特徵（第一性原理）：
    1. 營銷關鍵詞密集：
       - 時間緊迫：「限時」「搶購」「秒殺」「今日」
       - 優惠促銷：「優惠」「折扣」「特價」「免費」
       - 行動指令：「立即」「馬上」「快速」「點擊」
       - 數字誘導：「XX折」「$XX」「僅需」

    2. 文本穩定性極高（>0.8）：
       - AI渲染的文字完美靜止，幀間位置/內容不變
       - 真實視頻字幕會隨相機運動輕微抖動

    3. 文本密度高：
       - 畫面底部或頂部有大量文字
       - 多行文本同時出現
    """
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return 50.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 關鍵指標（v2.0重新設計）
        text_stability_scores = []    # 文本穩定性（幀間一致性）
        marketing_keyword_counts = []  # 營銷關鍵詞計數
        text_density_scores = []      # 文本密度
        text_position_variance = []   # 文本位置方差

        # 營銷關鍵詞庫（中英文）
        marketing_keywords = {
            # 時間緊迫類
            'urgency': ['限時', '搶購', '秒殺', '今日', '倒計時', '最後', '僅剩',
                       'limited', 'hurry', 'now', 'today', 'flash'],
            # 優惠促銷類
            'discount': ['優惠', '折扣', '特價', '免費', '贈送', '送', '省', '便宜',
                        'sale', 'off', 'free', 'discount', 'deal', 'save'],
            # 行動指令類
            'action': ['立即', '馬上', '快速', '點擊', '購買', '下單', '搶', '買',
                      'buy', 'click', 'order', 'get', 'shop'],
            # 數字誘導類（正則模式）
            'numbers': ['折', '%', '$', '元', '塊', '￥', '¥']
        }

        # 展平所有關鍵詞
        all_keywords = []
        for category in marketing_keywords.values():
            all_keywords.extend(category)

        sample_frames = min(60, total_frames)
        frames = []

        for i in range(sample_frames):
            if total_frames > 0:
                frame_pos = int(i * total_frames / sample_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        if len(frames) < 10:
            return 50.0

        # === 1. 文本區域檢測（使用邊緣檢測）===
        text_masks = []
        text_regions = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # 檢測文本區域（通常在畫面頂部或底部）
            # 使用Canny邊緣檢測 + 形態學操作

            edges = cv2.Canny(gray, 80, 160)

            # 膨脹以連接文本字符
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
            dilated = cv2.dilate(edges, kernel, iterations=1)

            # 找到輪廓
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 創建文本mask
            text_mask = np.zeros_like(gray)
            text_boxes = []

            for contour in contours:
                x, y, cw, ch = cv2.boundingRect(contour)

                # 文本特徵：寬度 > 3倍高度，高度適中
                if ch < 60 and ch > 10 and cw > 3 * ch and cw > 40:
                    cv2.rectangle(text_mask, (x, y), (x + cw, y + ch), 255, -1)
                    text_boxes.append((x, y, cw, ch))

            text_masks.append(text_mask)
            text_regions.append(text_boxes)

            # 計算文本密度
            text_area = np.sum(text_mask > 0)
            frame_area = h * w
            density = text_area / frame_area
            text_density_scores.append(density)

        # === 2. 文本穩定性分析 ===
        # 計算相鄰幀的文本mask重疊率
        # AI特徵：重疊率>0.8（文字完美靜止）
        # 真實特徵：重疊率<0.7（字幕抖動或動態）

        for i in range(len(text_masks) - 1):
            mask1 = text_masks[i]
            mask2 = text_masks[i + 1]

            # 計算IoU（Intersection over Union）
            intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
            union = np.logical_or(mask1 > 0, mask2 > 0).sum()

            if union > 0:
                stability = intersection / union
                text_stability_scores.append(stability)

        # === 3. 文本位置方差 ===
        # AI特徵：文本位置固定（方差低）
        # 真實特徵：文本位置變化（方差高）

        if len(text_regions) > 5:
            # 計算文本框中心位置
            centers_x = []
            centers_y = []

            for boxes in text_regions:
                if len(boxes) > 0:
                    # 計算所有文本框的平均中心
                    avg_x = np.mean([x + cw / 2 for x, y, cw, ch in boxes])
                    avg_y = np.mean([y + ch / 2 for x, y, cw, ch in boxes])
                    centers_x.append(avg_x)
                    centers_y.append(avg_y)

            if len(centers_x) > 3:
                # 計算位置方差
                var_x = np.var(centers_x)
                var_y = np.var(centers_y)
                total_variance = var_x + var_y
                text_position_variance.append(total_variance)

        # === 4. 營銷關鍵詞檢測（簡化版OCR）===
        # 注意：完整OCR需要pytesseract，這裡使用簡化方法
        # 檢測已知的營銷詞語模板（基於視覺特徵）

        # 簡化方法：分析文本密度+位置模式
        # AI帶貨片特徵：
        # 1. 底部有大量文字（>5%畫面）
        # 2. 文字高度穩定
        # 3. 多行並排（檢測text_boxes的y坐標分佈）

        marketing_pattern_score = 0

        # 檢查是否有密集底部文字（常見於AI帶貨）
        if len(text_regions) > 0:
            h, w = frames[0].shape[:2]

            for boxes in text_regions:
                if len(boxes) >= 3:  # 至少3個文本框
                    # 檢查是否在底部1/3
                    bottom_boxes = [box for box in boxes if box[1] > h * 0.6]

                    if len(bottom_boxes) >= 2:
                        marketing_pattern_score += 1

                    # 檢查是否有並排文本（多行文字）
                    y_coords = [box[1] for box in boxes]
                    if len(y_coords) > 2:
                        y_unique = len(set([int(y / 20) * 20 for y in y_coords]))  # 聚類到20px
                        if y_unique >= 2:  # 至少2行
                            marketing_pattern_score += 1

        # 由於沒有真正的OCR，使用模式匹配作為替代
        # 如果檢測到高穩定性 + 密集文字 + 底部位置，視為營銷特徵
        estimated_keyword_count = 0
        if len(text_stability_scores) > 0:
            avg_stability = np.mean(text_stability_scores)
            avg_density = np.mean(text_density_scores)

            # 啟發式規則：
            # 高穩定性(>0.8) + 高密度(>0.05) + 營銷模式 → 推測有營銷關鍵詞
            if avg_stability > 0.8 and avg_density > 0.05 and marketing_pattern_score >= 2:
                estimated_keyword_count = 5  # 推測有多個營銷關鍵詞
            elif avg_stability > 0.75 and avg_density > 0.03:
                estimated_keyword_count = 3
            elif avg_stability > 0.7 and marketing_pattern_score >= 1:
                estimated_keyword_count = 1

        marketing_keyword_counts.append(estimated_keyword_count)

        # === v2.0 評分邏輯（第一性原理驅動）===
        score = 50.0  # 中性基礎分

        # 1. 文本穩定性（核心指標）
        if len(text_stability_scores) > 0:
            avg_stability = np.mean(text_stability_scores)

            # 第一性原理：AI>0.8，真實<0.7
            if avg_stability > 0.85:  # AI特徵（文字完美靜止）
                score += 35.0
                logging.info(f"TFP v2: Very high text stability {avg_stability:.3f} - AI rendered text")
            elif avg_stability > 0.75:
                score += 22.0
            elif avg_stability < 0.6:  # 真實特徵（文字抖動）
                score -= 25.0
                logging.info(f"TFP v2: Low text stability {avg_stability:.3f} - Real dynamic text")
            elif avg_stability < 0.7:
                score -= 15.0

        # 2. 營銷關鍵詞/模式
        if len(marketing_keyword_counts) > 0:
            avg_keywords = np.mean(marketing_keyword_counts)

            # AI特徵：多個營銷關鍵詞
            if avg_keywords >= 4:
                score += 40.0
                logging.info(f"TFP v2: Multiple marketing keywords {avg_keywords:.1f} - AI commercial video")
            elif avg_keywords >= 2:
                score += 25.0
            elif avg_keywords == 0:  # 真實特徵（無營銷話術）
                score -= 20.0

        # 3. 文本密度
        if len(text_density_scores) > 0:
            avg_density = np.mean(text_density_scores)

            # AI帶貨片特徵：文本密度高
            if avg_density > 0.08:  # >8%畫面是文字
                score += 22.0
            elif avg_density > 0.05:
                score += 12.0
            elif avg_density < 0.02:  # 低密度（真實或無文字）
                score -= 10.0

        # 4. 文本位置方差
        if len(text_position_variance) > 0:
            avg_variance = np.mean(text_position_variance)

            # AI特徵：位置固定（低方差）
            # 真實特徵：位置變化（高方差）
            if avg_variance < 100:  # 極低方差（位置固定）
                score += 18.0
            elif avg_variance > 1000:  # 高方差（位置變化）
                score -= 12.0

        # 5. 營銷模式分數
        if marketing_pattern_score >= 3:  # 多個營銷模式特徵
            score += 20.0
            logging.info(f"TFP v2: Marketing pattern detected (score={marketing_pattern_score}) - AI commercial")

        # 限制分數範圍
        score = max(5.0, min(95.0, score))

        logging.info(f"TFP v2.0: stability={np.mean(text_stability_scores) if len(text_stability_scores) > 0 else 0:.3f}, "
                    f"density={np.mean(text_density_scores) if len(text_density_scores) > 0 else 0:.4f}, "
                    f"keywords={np.mean(marketing_keyword_counts) if len(marketing_keyword_counts) > 0 else 0:.1f}, "
                    f"pattern={marketing_pattern_score}, score={score:.1f}")

        return score

    except Exception as e:
        logging.error(f"Error in text_fingerprinting v2.0: {e}")
        return 50.0

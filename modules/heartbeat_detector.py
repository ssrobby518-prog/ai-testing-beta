#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模組 Heartbeat Detector v2.0 - TSAR-RAPTOR Phase II
第一性原理：真實人類心跳有混沌變異性（HRV），AI無法模擬自律神經系統的隨機性
rPPG技術提取心跳信號，關鍵是檢測HRV（Heart Rate Variability），非單純心率

關鍵差異:
- 真實: HRV > 50ms (自律神經調控，beat-to-beat變化) + 頻域LF/HF比例 0.5-2.0
- AI: HRV < 20ms (合成信號過於規則) + 頻域單一峰值

優化記錄:
- v1.0: 差距+1.7 (只檢測心率存在，AI也可以有"plausible"心率)
- v2.0: 預期+35 (檢測HRV混沌性，AI無法模擬生理隨機性)
"""

import logging
import cv2
import numpy as np
from scipy import signal, stats

logging.basicConfig(level=logging.INFO)

def detect(file_path):
    """
    第一性原理v2.0：rPPG + HRV混沌性檢測

    真實人類心跳特徵（第一性原理）：
    1. HRV（Heart Rate Variability）:
       - 時域: SDNN > 50ms（相鄰R-R間期標準差）
       - 頻域: LF/HF比例 0.5-2.0（交感/副交感神經平衡）
       - pNN50 > 5%（相鄰R-R差異>50ms的比例）

    2. 心跳波形複雜度:
       - 真實PPG信號有收縮峰、舒張切跡（Dicrotic Notch）
       - 波形複雜度高（多個諧波）

    3. 呼吸性竇性心律不齊（RSA）:
       - 吸氣時心率加快，呼氣時心率減慢
       - 頻率約0.15-0.4 Hz（呼吸頻率）

    AI生成視頻缺陷：
    1. HRV過低: 合成信號太規則（SDNN < 20ms）
    2. 單一頻率: 只有一個峰值，缺少LF/HF成分
    3. 波形簡單: 缺少生理細節（舒張切跡）
    """
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return 50.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps > 120:
            fps = 30.0

        # 關鍵指標（v2.0重新設計）
        hrv_sdnn_values = []          # HRV時域指標（標準差）
        hrv_rmssd_values = []         # HRV時域指標（連續差的平方根）
        lf_hf_ratios = []             # 頻域LF/HF比例
        waveform_complexities = []    # 波形複雜度
        rsa_presence = []             # 呼吸性竇性心律不齊

        # 提取綠色通道信號（rPPG）
        sample_frames = min(180, total_frames)  # 需要更多幀以檢測HRV（至少6秒）

        green_signal = []
        frame_count = 0

        # 臉部檢測器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        for i in range(sample_frames):
            if total_frames > 0:
                frame_pos = int(i * total_frames / sample_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 檢測臉部ROI
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            if len(faces) > 0:
                # 選擇最大的臉
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

                # 提取額頭和臉頰區域（心跳信號最強）
                # 額頭：臉部上方1/4
                forehead_y = y + int(h * 0.1)
                forehead_h = int(h * 0.25)
                forehead_x = x + int(w * 0.25)
                forehead_w = int(w * 0.5)

                roi = frame[forehead_y:forehead_y+forehead_h, forehead_x:forehead_x+forehead_w]

                if roi.size > 0:
                    # 提取綠色通道（血液吸收綠光，心跳信號明顯）
                    green_channel = roi[:, :, 1]

                    # 計算平均綠色值
                    green_mean = np.mean(green_channel)
                    green_signal.append(green_mean)
                    frame_count += 1

        cap.release()

        # 需要至少6秒的數據（180幀@30fps）來檢測HRV
        if len(green_signal) < 60:
            return 50.0

        green_signal = np.array(green_signal)

        # === 1. 信號預處理 ===
        # 去趨勢（移除低頻漂移）
        # 使用高通濾波器（0.5 Hz）移除基線漂移
        b, a = signal.butter(3, 0.5 / (fps / 2), btype='high')
        green_detrended = signal.filtfilt(b, a, green_signal)

        # 帶通濾波（0.7-4.0 Hz，對應42-240 BPM）
        b, a = signal.butter(3, [0.7 / (fps / 2), 4.0 / (fps / 2)], btype='band')
        green_filtered = signal.filtfilt(b, a, green_detrended)

        # === 2. 心跳峰值檢測 ===
        # 使用scipy的find_peaks檢測R峰（收縮峰）
        # 設置最小峰值間隔（0.4秒，對應最大150 BPM）
        min_distance = int(0.4 * fps)

        peaks, properties = signal.find_peaks(
            green_filtered,
            distance=min_distance,
            prominence=np.std(green_filtered) * 0.5  # 峰值顯著性
        )

        if len(peaks) < 5:
            # 檢測到的峰值太少，無法計算HRV
            return 50.0

        # === 3. HRV時域分析（核心指標）===
        # 計算R-R間期（相鄰峰值的時間間隔）
        rr_intervals = np.diff(peaks) / fps * 1000  # 轉換為毫秒

        if len(rr_intervals) < 4:
            return 50.0

        # 3.1 SDNN（R-R間期標準差）
        # 第一性原理：真實人類 SDNN > 50ms，AI < 20ms
        hrv_sdnn = np.std(rr_intervals)
        hrv_sdnn_values.append(hrv_sdnn)

        # 3.2 RMSSD（相鄰R-R間期差的平方根均值）
        # 反映短期變異性（副交感神經活動）
        successive_diffs = np.diff(rr_intervals)
        hrv_rmssd = np.sqrt(np.mean(successive_diffs ** 2))
        hrv_rmssd_values.append(hrv_rmssd)

        # 3.3 pNN50（相鄰R-R差異>50ms的比例）
        pnn50 = np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs) * 100

        # === 4. HRV頻域分析 ===
        # 第一性原理：真實心跳有兩個頻段
        # - LF (Low Frequency): 0.04-0.15 Hz（交感+副交感）
        # - HF (High Frequency): 0.15-0.4 Hz（副交感，呼吸性竇性心律）

        # 將不規則的R-R間期插值為規則採樣
        rr_time = np.cumsum(rr_intervals) / 1000  # 轉換為秒
        rr_time = np.insert(rr_time, 0, 0)  # 添加起始點

        # 插值到4Hz採樣率
        interp_rate = 4.0  # Hz
        regular_time = np.arange(0, rr_time[-1], 1.0 / interp_rate)

        if len(regular_time) > 10:
            # 線性插值
            rr_interp = np.interp(regular_time, rr_time, np.insert(rr_intervals, 0, rr_intervals[0]))

            # FFT分析
            fft_result = np.fft.rfft(rr_interp - np.mean(rr_interp))
            fft_freqs = np.fft.rfftfreq(len(rr_interp), d=1.0 / interp_rate)
            fft_power = np.abs(fft_result) ** 2

            # LF和HF頻段能量
            lf_mask = (fft_freqs >= 0.04) & (fft_freqs < 0.15)
            hf_mask = (fft_freqs >= 0.15) & (fft_freqs < 0.4)

            lf_power = np.sum(fft_power[lf_mask])
            hf_power = np.sum(fft_power[hf_mask])

            if hf_power > 0:
                lf_hf_ratio = lf_power / hf_power
                lf_hf_ratios.append(lf_hf_ratio)

                # 檢測呼吸性竇性心律不齊（RSA）
                # 真實特徵：HF頻段有明顯能量（>10% 總能量）
                total_power = np.sum(fft_power)
                hf_ratio = hf_power / (total_power + 1e-10)

                if hf_ratio > 0.10:  # HF頻段占比>10%
                    rsa_presence.append(1.0)
                else:
                    rsa_presence.append(0.0)

        # === 5. 波形複雜度分析 ===
        # 第一性原理：真實PPG波形有多個諧波（收縮峰、舒張切跡）
        # AI波形：通常只有單一正弦波

        # 計算波形的頻譜複雜度
        # 檢測信號中有多少個顯著的頻率成分
        fft_signal = np.fft.rfft(green_filtered)
        fft_signal_freqs = np.fft.rfftfreq(len(green_filtered), d=1.0 / fps)
        fft_signal_power = np.abs(fft_signal) ** 2

        # 在心率頻段（0.7-4.0 Hz）檢測峰值數量
        hr_mask = (fft_signal_freqs >= 0.7) & (fft_signal_freqs <= 4.0)
        hr_spectrum = fft_signal_power[hr_mask]

        if len(hr_spectrum) > 5:
            # 檢測頻譜峰值（諧波）
            spectrum_peaks, _ = signal.find_peaks(hr_spectrum, prominence=np.max(hr_spectrum) * 0.1)

            # 真實波形：多個諧波（2-4個峰值）
            # AI波形：單一峰值（1個）
            waveform_complexity = len(spectrum_peaks)
            waveform_complexities.append(waveform_complexity)

        # === v2.0 評分邏輯（第一性原理驅動）===
        score = 50.0  # 中性基礎分

        # 1. HRV時域指標（核心指標 - 權重最高）
        if len(hrv_sdnn_values) > 0:
            avg_sdnn = np.mean(hrv_sdnn_values)

            # 第一性原理：真實SDNN > 50ms，AI < 20ms
            if avg_sdnn < 15:  # AI特徵（極低變異性）
                score += 40.0
                logging.info(f"HBD v2: Very low HRV SDNN {avg_sdnn:.1f}ms - AI synthetic heartbeat")
            elif avg_sdnn < 25:
                score += 30.0
            elif avg_sdnn < 40:
                score += 18.0
            elif avg_sdnn > 60:  # 真實特徵（自然變異性）
                score -= 30.0
                logging.info(f"HBD v2: High HRV SDNN {avg_sdnn:.1f}ms - Real human heartbeat")
            elif avg_sdnn > 50:
                score -= 20.0

        # 2. RMSSD（短期變異性）
        if len(hrv_rmssd_values) > 0:
            avg_rmssd = np.mean(hrv_rmssd_values)

            if avg_rmssd < 12:  # AI特徵
                score += 25.0
            elif avg_rmssd < 20:
                score += 15.0
            elif avg_rmssd > 40:  # 真實特徵
                score -= 18.0

        # 3. pNN50（beat-to-beat變異性）
        if pnn50 < 2:  # AI特徵（幾乎無變異）
            score += 20.0
        elif pnn50 > 10:  # 真實特徵
            score -= 15.0

        # 4. LF/HF比例（自律神經平衡）
        if len(lf_hf_ratios) > 0:
            avg_lf_hf = np.mean(lf_hf_ratios)

            # 第一性原理：真實比例 0.5-2.0（交感/副交感平衡）
            # AI：通常缺少HF成分（比例>5）或只有單一頻率（比例<0.2）
            if avg_lf_hf < 0.2 or avg_lf_hf > 5.0:  # AI特徵（失衡）
                score += 22.0
                logging.info(f"HBD v2: Abnormal LF/HF ratio {avg_lf_hf:.2f} - AI feature")
            elif 0.5 <= avg_lf_hf <= 2.0:  # 真實特徵（平衡）
                score -= 15.0

        # 5. 呼吸性竇性心律不齊（RSA）
        if len(rsa_presence) > 0:
            avg_rsa = np.mean(rsa_presence)
            if avg_rsa > 0.5:  # 真實特徵（有RSA）
                score -= 12.0
            elif avg_rsa < 0.2:  # AI特徵（無RSA）
                score += 15.0

        # 6. 波形複雜度
        if len(waveform_complexities) > 0:
            avg_complexity = np.mean(waveform_complexities)
            if avg_complexity <= 1:  # AI特徵（單一頻率）
                score += 18.0
                logging.info(f"HBD v2: Simple waveform (1 peak) - AI synthetic signal")
            elif avg_complexity >= 3:  # 真實特徵（多諧波）
                score -= 12.0

        # 限制分數範圍
        score = max(5.0, min(95.0, score))

        logging.info(f"HBD v2.0: SDNN={np.mean(hrv_sdnn_values) if len(hrv_sdnn_values) > 0 else 0:.1f}ms, "
                    f"RMSSD={np.mean(hrv_rmssd_values) if len(hrv_rmssd_values) > 0 else 0:.1f}ms, "
                    f"pNN50={pnn50:.1f}%, LF/HF={np.mean(lf_hf_ratios) if len(lf_hf_ratios) > 0 else 0:.2f}, "
                    f"complexity={np.mean(waveform_complexities) if len(waveform_complexities) > 0 else 0:.1f}, score={score:.1f}")

        return score

    except Exception as e:
        logging.error(f"Error in heartbeat_detector v2.0: {e}")
        return 50.0

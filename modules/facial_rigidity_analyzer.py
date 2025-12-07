#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blue Team Phase I - Module 1: Facial Rigidity Analyzer (面部剛性檢測)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

第一性原理 (First Principle Axiom):
    "真實世界是剛體運動的連續物理記錄；AI 影片是像素概率的離散數學預測。
    因此，真實物體的幾何結構在時間軸上是守恆的 (Conservation of Geometry)，
    而 AI 生成的物體會出現非物理的微觀漂移 (Micro-Drift)。"

攻擊目標 (Attack Target):
    AI 對人臉骨骼結構理解的缺失 - 頭骨不應該變形！

技術棧 (Technical Stack):
    - Google MediaPipe Face Mesh (468 landmarks, 亞像素精度)
    - Procrustes Analysis (普氏分析 - 剛體對齊)
    - Euclidean Distance Variance (歐氏距離變異數)

成功標準 (Success Criteria):
    - Real Video: Jitter Score < 0.05 pixels (僅感測器噪聲)
    - AI Video: Jitter Score > 0.5 pixels (結構不穩定)

沙皇炸彈原則 (Tsar Bomba):
    骨骼結構守恆是物理不可偽造的 - 這是對AI生成的核武級攻擊

猛禽3引擎原則 (Raptor 3):
    純計算模組，無I/O依賴，可並行執行
"""

import logging
import numpy as np
from typing import List, Tuple, Optional

logging.basicConfig(level=logging.INFO)


def detect(file_path: str) -> float:
    """
    主檢測函數 - 符合現有模組接口

    Args:
        file_path: 視頻文件路徑

    Returns:
        float: AI概率 [0-100]
            - 0-20: 剛體穩定（真實）
            - 20-60: 中等漂移（可疑）
            - 60-100: 嚴重漂移（AI）
    """
    try:
        # 懶加載依賴（避免影響其他模組）
        try:
            import cv2
            import mediapipe as mp
        except ImportError:
            logging.warning("MediaPipe not installed, returning neutral score")
            return 50.0

        # === TSAR原則：一次性加載視頻和模型 ===
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return 50.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # MediaPipe Face Mesh 初始化
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # 啟用468點高精度
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # === RAPTOR原則：智能採樣 ===
        # 只採樣關鍵幀，避免過度計算
        sample_frames = min(60, total_frames)

        landmarks_sequence = []  # 儲存每幀的468個特徵點

        for i in range(sample_frames):
            if total_frames > 0:
                frame_pos = int(i * total_frames / sample_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

            ret, frame = cap.read()
            if not ret:
                break

            # 轉換為RGB（MediaPipe要求）
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape

            # 檢測面部特徵點
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                # 提取468個3D特徵點（歸一化坐標）
                points_3d = []
                for landmark in face_landmarks.landmark:
                    # 轉換為像素坐標（亞像素精度）
                    x = landmark.x * w
                    y = landmark.y * h
                    z = landmark.z * w  # MediaPipe的z是相對深度
                    points_3d.append([x, y, z])

                landmarks_sequence.append(np.array(points_3d))

        cap.release()
        face_mesh.close()

        # === 檢查：至少需要10幀有效數據 ===
        if len(landmarks_sequence) < 10:
            logging.warning("Insufficient face landmarks, returning neutral")
            return 50.0

        # === SPARK-PLUG原則：核心計算邏輯 ===
        jitter_score = _calculate_geometric_jitter(landmarks_sequence)

        # === 第一性原理判定 ===
        return _jitter_to_ai_probability(jitter_score)

    except Exception as e:
        logging.error(f"Error in facial_rigidity_analyzer: {e}")
        return 50.0


def _calculate_geometric_jitter(
    landmarks_sequence: List[np.ndarray]
) -> float:
    """
    計算幾何抖動分數（Jitter Score）

    核心算法：Procrustes Analysis（普氏分析）

    步驟：
    1. 選定錨點（鼻樑+眼眶骨）作為剛體參考
    2. 將每幀的人臉對齊到第一幀（消除頭部運動）
    3. 計算關鍵幾何距離的變異數

    Args:
        landmarks_sequence: 每幀的468點列表

    Returns:
        float: Jitter分數（像素單位）
    """
    # === 定義錨點（剛體結構）===
    # MediaPipe Face Mesh 關鍵點索引
    ANCHOR_POINTS = [
        1,    # 鼻樑頂部
        4,    # 鼻尖
        33,   # 左眼眶外角
        133,  # 左眼眶內角
        362,  # 右眼眶內角
        263,  # 右眼眶外角
    ]

    # === 定義測量距離（應該守恆的幾何特徵）===
    # 這些是骨骼結構，不應該隨時間變化
    RIGID_DISTANCES = [
        (33, 263),   # 瞳距（Inter-pupillary distance）
        (4, 1),      # 鼻長
        (33, 133),   # 左眼寬
        (362, 263),  # 右眼寬
    ]

    # 參考幀（第一幀）
    reference_landmarks = landmarks_sequence[0]

    # 計算每個距離在時間序列上的變異
    distance_variances = []

    for point_a, point_b in RIGID_DISTANCES:
        distances_over_time = []

        for frame_landmarks in landmarks_sequence:
            # === Procrustes對齊（消除頭部運動）===
            # 使用錨點進行剛體對齊
            aligned_landmarks = _procrustes_align(
                source=frame_landmarks,
                target=reference_landmarks,
                anchor_indices=ANCHOR_POINTS
            )

            # 計算歐氏距離
            point_a_3d = aligned_landmarks[point_a]
            point_b_3d = aligned_landmarks[point_b]
            distance = np.linalg.norm(point_a_3d - point_b_3d)
            distances_over_time.append(distance)

        # 計算變異數
        variance = np.var(distances_over_time)
        distance_variances.append(variance)

    # 平均變異數作為Jitter Score
    jitter_score = float(np.mean(distance_variances))

    logging.info(f"Facial Rigidity: Jitter={jitter_score:.4f} pixels²")
    return jitter_score


def _procrustes_align(
    source: np.ndarray,
    target: np.ndarray,
    anchor_indices: List[int]
) -> np.ndarray:
    """
    Procrustes對齊（普氏分析）- 剛體對齊

    將source的錨點對齊到target的錨點（旋轉+平移，不縮放）

    Args:
        source: 待對齊的特徵點 (468, 3)
        target: 目標特徵點 (468, 3)
        anchor_indices: 錨點索引列表

    Returns:
        np.ndarray: 對齊後的source
    """
    # 提取錨點
    source_anchors = source[anchor_indices]
    target_anchors = target[anchor_indices]

    # 計算質心
    source_centroid = np.mean(source_anchors, axis=0)
    target_centroid = np.mean(target_anchors, axis=0)

    # 去質心化
    source_centered = source_anchors - source_centroid
    target_centered = target_anchors - target_centroid

    # 計算旋轉矩陣（SVD方法）
    H = source_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 確保是旋轉矩陣（行列式為1）
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 應用變換到全部特徵點
    source_centered_all = source - source_centroid
    aligned = (R @ source_centered_all.T).T + target_centroid

    return aligned


def _jitter_to_ai_probability(jitter_score: float) -> float:
    """
    將Jitter分數轉換為AI概率

    第一性原理映射：
    - jitter < 0.05: 真實（僅感測器噪聲）
    - 0.05 <= jitter < 0.5: 可疑（中等）
    - jitter >= 0.5: AI（幾何漂移）

    Args:
        jitter_score: Jitter分數（像素²）

    Returns:
        float: AI概率 [0-100]
    """
    # === 沙皇炸彈：絕對閾值判定 ===
    if jitter_score < 0.05:
        # 絕對真實
        ai_prob = 5.0
        logging.info("Facial Rigidity: ABSOLUTE REAL (rigid structure)")

    elif jitter_score >= 2.0:
        # 嚴重漂移 - 絕對AI
        ai_prob = 95.0
        logging.info("Facial Rigidity: ABSOLUTE AI (severe geometric drift)")

    elif jitter_score >= 0.5:
        # 明顯漂移 - 強AI信號
        # 線性映射：0.5 -> 70, 2.0 -> 95
        ai_prob = 70.0 + (jitter_score - 0.5) * (95.0 - 70.0) / (2.0 - 0.5)
        logging.info(f"Facial Rigidity: Strong AI signal (jitter={jitter_score:.3f})")

    else:
        # 中等範圍：0.05 -> 20, 0.5 -> 70
        ai_prob = 20.0 + (jitter_score - 0.05) * (70.0 - 20.0) / (0.5 - 0.05)
        logging.info(f"Facial Rigidity: Moderate signal (jitter={jitter_score:.3f})")

    return max(5.0, min(95.0, ai_prob))


# === 導出函數（供優化版總控使用）===
def analyze_from_frames(frames_with_faces) -> float:
    """
    從預處理幀中分析（適配core/架構）

    Args:
        frames_with_faces: PreprocessedFrame列表

    Returns:
        float: AI概率
    """
    # TODO: 實現適配層（當core/架構升級後）
    pass

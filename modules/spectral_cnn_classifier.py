#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blue Team Phase II - Module 4: Spectral CNN Classifier (頻域卷積神經網絡)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

第一性原理 (First Principle Axiom):
    "真實影像的噪聲源於光子的隨機性 (Poisson Noise)，在頻譜上呈均勻或自然衰減分佈；
    AI 影像源於反卷積與上採樣 (Upsampling)，在頻譜上必然留下週期性的數學偽影。"

    人類視覺皮層會忽略高頻噪聲，但傅立葉變換 (FFT) 能讓這些『生成指紋』無所遁形。

攻擊目標 (Attack Target):
    檢測生成模型留下的週期性網格痕跡（棋盤格效應 Checkerboard Artifacts）

技術棧 (Technical Stack):
    - NumPy FFT2 (2D快速傅立葉變換)
    - ResNet-18 / EfficientNet-B0 (輕量級CNN)
    - Input: Spectrum Images (頻譜圖，而非RGB人臉)
    - Loss: Binary Cross Entropy

目標 (Objective):
    攔截 30% 視覺上完美、但保留了『棋盤格效應』的高階偽造影片

成功標準 (Success Criteria):
    - 頻譜圖上檢測到異常峰值（Star-like patterns, Cross shapes, Grid lines）

沙皇炸彈原則 (Tsar Bomba):
    CNN能比人眼更敏銳地識別頻譜圖中的微弱網格紋理 - 這是對Diffusion Model的核武攻擊

猛禽3引擎原則 (Raptor 3):
    模型推理可GPU加速，與其他模組完全解耦
"""

import logging
import numpy as np
import os
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)

# === 模型路徑配置 ===
MODEL_PATH = "models/spectral_cnn.pth"  # 預訓練模型路徑


def detect(file_path: str) -> float:
    """
    主檢測函數 - 符合現有模組接口

    Args:
        file_path: 視頻文件路徑

    Returns:
        float: AI概率 [0-100]
    """
    try:
        # === 檢查模型是否存在 ===
        if not os.path.exists(MODEL_PATH):
            logging.warning(f"Spectral CNN model not found at {MODEL_PATH}, using fallback FFT analysis")
            return _fallback_fft_analysis(file_path)

        # 懶加載依賴
        try:
            import cv2
            import torch
            from torchvision import transforms
        except ImportError:
            logging.warning("PyTorch not installed, using fallback")
            return _fallback_fft_analysis(file_path)

        # === TSAR原則：提取頻譜圖 ===
        spectrum_image = _extract_spectrum_image(file_path)
        if spectrum_image is None:
            return 50.0

        # === 加載CNN模型 ===
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _load_spectral_cnn_model(MODEL_PATH, device)
        model.eval()

        # === 預處理頻譜圖 ===
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])  # 單通道歸一化
        ])

        spectrum_tensor = transform(spectrum_image).unsqueeze(0).to(device)

        # === SPARK-PLUG原則：模型推理 ===
        with torch.no_grad():
            output = model(spectrum_tensor)
            ai_probability = torch.sigmoid(output).item() * 100.0

        logging.info(f"Spectral CNN: AI_P={ai_probability:.2f}")
        return ai_probability

    except Exception as e:
        logging.error(f"Error in spectral_cnn_classifier: {e}")
        return _fallback_fft_analysis(file_path)


def _extract_spectrum_image(file_path: str) -> Optional[np.ndarray]:
    """
    提取視頻的頻譜圖（Spectrum Heatmap）

    流程：
    1. 讀取多幀視頻
    2. 高通濾波（移除主體內容，保留噪聲殘差）
    3. 2D FFT轉換
    4. 對數縮放 + 中心化
    5. 生成熱力圖

    Returns:
        np.ndarray: 灰度頻譜圖 (H, W) uint8
    """
    import cv2

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_frames = min(30, total_frames)

    magnitudes = []

    for i in range(sample_frames):
        if total_frames > 0:
            frame_pos = int(i * total_frames / sample_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

        ret, frame = cap.read()
        if not ret:
            break

        # 轉換為灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # === 高通濾波（DoG - Difference of Gaussians）===
        # 移除低頻內容（人臉、背景），保留高頻噪聲
        blur_low = cv2.GaussianBlur(gray, (5, 5), 1.0)
        blur_high = cv2.GaussianBlur(gray, (21, 21), 5.0)
        high_pass = cv2.subtract(blur_low, blur_high)

        # 調整大小為固定尺寸（加速FFT）
        resized = cv2.resize(high_pass, (512, 512))

        # === 2D FFT ===
        fft = np.fft.fft2(resized)
        fft_shifted = np.fft.fftshift(fft)  # 中心化
        magnitude = np.abs(fft_shifted)

        # 對數縮放（增強可視化）
        magnitude_log = 20 * np.log(magnitude + 1e-10)
        magnitudes.append(magnitude_log)

    cap.release()

    if len(magnitudes) == 0:
        return None

    # 平均頻譜
    avg_magnitude = np.mean(magnitudes, axis=0)

    # 歸一化到 [0, 255]
    spectrum_normalized = cv2.normalize(
        avg_magnitude,
        None,
        0, 255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    return spectrum_normalized


def _load_spectral_cnn_model(model_path: str, device):
    """
    加載預訓練的Spectral CNN模型

    架構：ResNet-18（修改為單通道輸入）

    Args:
        model_path: 模型權重路徑
        device: torch設備

    Returns:
        torch.nn.Module: 加載的模型
    """
    import torch
    import torch.nn as nn
    from torchvision import models

    # 創建ResNet-18（修改第一層為單通道）
    model = models.resnet18(pretrained=False)

    # 修改第一層卷積（3通道 -> 1通道）
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # 修改最後一層（輸出單個概率）
    model.fc = nn.Linear(model.fc.in_features, 1)

    # 加載權重
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f"Loaded Spectral CNN from {model_path}")
    else:
        logging.warning(f"Model weights not found at {model_path}, using random initialization")

    return model.to(device)


def _fallback_fft_analysis(file_path: str) -> float:
    """
    備用方案：使用傳統FFT分析（不依賴CNN）

    當模型未訓練時的降級方案

    Args:
        file_path: 視頻路徑

    Returns:
        float: AI概率
    """
    import cv2

    try:
        spectrum_image = _extract_spectrum_image(file_path)
        if spectrum_image is None:
            return 50.0

        # === 手動檢測頻譜異常 ===
        # 1. 檢測星狀模式（Star-like patterns）
        # 2. 檢測十字線（Cross shapes）
        # 3. 檢測網格線（Grid lines）

        h, w = spectrum_image.shape
        center_y, center_x = h // 2, w // 2

        # 檢測中心區域的能量分佈
        center_region = spectrum_image[
            center_y - 50:center_y + 50,
            center_x - 50:center_x + 50
        ]
        center_energy = np.mean(center_region)

        # 檢測高頻區域的能量
        high_freq_region = spectrum_image[
            :h//4, :w//4
        ]
        high_freq_energy = np.mean(high_freq_region)

        # 檢測異常峰值
        threshold = np.percentile(spectrum_image, 95)
        peak_count = np.sum(spectrum_image > threshold)

        # 簡單啟發式規則
        ai_score = 30.0

        # 高頻能量異常低（AI截斷）
        if high_freq_energy < center_energy * 0.3:
            ai_score += 30.0

        # 異常峰值過多（棋盤格效應）
        if peak_count > (h * w * 0.05):
            ai_score += 25.0

        # 限制範圍
        ai_score = max(10.0, min(90.0, ai_score))

        logging.info(f"Spectral FFT (fallback): AI_P={ai_score:.2f}")
        return ai_score

    except Exception as e:
        logging.error(f"Error in fallback FFT: {e}")
        return 50.0


# === 訓練腳本（獨立運行）===
def train_spectral_cnn(
    dataset_path: str,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001
):
    """
    訓練Spectral CNN分類器

    數據集格式：
        dataset_path/
            real/
                spectrum_001.png
                spectrum_002.png
                ...
            fake/
                spectrum_001.png
                spectrum_002.png
                ...

    Args:
        dataset_path: 數據集根目錄
        epochs: 訓練輪數
        batch_size: 批次大小
        learning_rate: 學習率
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms

        logging.info("=== Training Spectral CNN ===")

        # 數據增強
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

        # 加載數據集
        dataset = datasets.ImageFolder(dataset_path, transform=transform)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 創建模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _load_spectral_cnn_model(MODEL_PATH, device)

        # 損失函數和優化器
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 訓練循環
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                # 前向傳播
                outputs = model(images)
                loss = criterion(outputs, labels)

                # 反向傳播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 統計
                total_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            accuracy = correct / total * 100
            logging.info(f"Epoch {epoch+1}/{epochs}: Loss={total_loss:.4f}, Acc={accuracy:.2f}%")

        # 保存模型
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        logging.info(f"Model saved to {MODEL_PATH}")

    except ImportError:
        logging.error("PyTorch not installed, cannot train model")


if __name__ == "__main__":
    # 示例：訓練模型
    # train_spectral_cnn("datasets/spectrum_dataset", epochs=20)
    pass

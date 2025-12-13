#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSAR-RAPTOR Configuration
系統路徑配置文件
"""

from pathlib import Path

# ========== Layer 1 路徑配置 ==========

# Layer 1 基礎目錄
LAYER1_BASE_DIR = Path(r"C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok tinder videos")

# Layer 1 數據目錄
LAYER1_DATA_DIR = LAYER1_BASE_DIR / "data"

# Excel 文件路徑（修正：添加 .xlsx 擴展名）
EXCEL_A_PATH = LAYER1_DATA_DIR / "excel_a_labels_raw.xlsx"
EXCEL_B_PATH = LAYER1_DATA_DIR / "excel_b_features.xlsx"
EXCEL_C_PATH = LAYER1_DATA_DIR / "excel_c_analysis.xlsx"

# Layer 1 視頻分類文件夾
LAYER1_VIDEO_FOLDERS = {
    'real': LAYER1_BASE_DIR / 'real',
    'ai': LAYER1_BASE_DIR / 'ai',
    'uncertain': LAYER1_BASE_DIR / 'not sure',
    'exclude': LAYER1_BASE_DIR / 'movies'
}

# ========== Layer 2 路徑配置 ==========

# Layer 2 基礎目錄
LAYER2_BASE_DIR = Path(r"C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok videos download")

# Layer 2 數據目錄
LAYER2_DATA_DIR = LAYER2_BASE_DIR / "data"

# Excel D 文件路徑
EXCEL_D_PATH = LAYER2_DATA_DIR / "excel_d_detection_results.xlsx"

# Layer 2 視頻分類文件夾
LAYER2_VIDEO_FOLDERS = {
    'REAL': LAYER2_BASE_DIR / 'real',
    'AI': LAYER2_BASE_DIR / 'ai',
    'NOT_SURE': LAYER2_BASE_DIR / 'not sure',
    '電影動畫': LAYER2_BASE_DIR / '電影動畫'
}

# ========== 自動創建目錄 ==========

def ensure_directories():
    """確保所有必要的目錄存在"""
    # Layer 1 目錄
    LAYER1_BASE_DIR.mkdir(parents=True, exist_ok=True)
    LAYER1_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for folder in LAYER1_VIDEO_FOLDERS.values():
        folder.mkdir(parents=True, exist_ok=True)

    # Layer 2 目錄
    LAYER2_BASE_DIR.mkdir(parents=True, exist_ok=True)
    LAYER2_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for folder in LAYER2_VIDEO_FOLDERS.values():
        folder.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print("✅ 所有目錄已創建")
    print(f"\nLayer 1 基礎目錄: {LAYER1_BASE_DIR}")
    print(f"Layer 2 基礎目錄: {LAYER2_BASE_DIR}")

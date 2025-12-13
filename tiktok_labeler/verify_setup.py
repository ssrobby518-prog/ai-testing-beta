#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSAR-RAPTOR Setup Verification
驗證系統設置和路徑配置

運行此腳本以確保所有路徑正確配置
"""

from pathlib import Path
import sys
import os

# 添加項目路徑
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    LAYER1_BASE_DIR, LAYER1_DATA_DIR, LAYER1_VIDEO_FOLDERS,
    EXCEL_A_PATH, EXCEL_B_PATH, EXCEL_C_PATH,
    LAYER2_BASE_DIR, LAYER2_DATA_DIR, LAYER2_VIDEO_FOLDERS,
    EXCEL_D_PATH,
    ensure_directories
)


def verify_layer1():
    """驗證 Layer 1 配置"""
    print("\n" + "="*80)
    print("Layer 1 Path Verification")
    print("="*80)

    # 檢查基礎目錄
    print(f"\nBase Directory: {LAYER1_BASE_DIR}")
    if LAYER1_BASE_DIR.exists():
        print("  [OK] Exists")
    else:
        print("  [WARN] Not exists (will create)")
        LAYER1_BASE_DIR.mkdir(parents=True, exist_ok=True)
        print("  [OK] Created")

    # 檢查數據目錄
    print(f"\nData Directory: {LAYER1_DATA_DIR}")
    if LAYER1_DATA_DIR.exists():
        print("  [OK] Exists")
    else:
        print("  [WARN] Not exists (will create)")
        LAYER1_DATA_DIR.mkdir(parents=True, exist_ok=True)
        print("  [OK] Created")

    # 檢查視頻文件夾
    print("\nVideo Classification Folders:")
    for label, folder in LAYER1_VIDEO_FOLDERS.items():
        print(f"  * {label}: {folder}")
        if folder.exists():
            # 統計視頻數量
            video_count = len(list(folder.glob("*.mp4")))
            print(f"    [OK] Exists ({video_count} videos)")
        else:
            print(f"    [WARN] Not exists (will create)")
            folder.mkdir(parents=True, exist_ok=True)
            print(f"    [OK] Created")

    # 檢查 Excel 文件
    print("\nExcel Files:")
    for name, path in [("Excel A", EXCEL_A_PATH), ("Excel B", EXCEL_B_PATH), ("Excel C", EXCEL_C_PATH)]:
        print(f"  * {name}: {path}")
        if path.exists():
            import pandas as pd
            try:
                df = pd.read_excel(path)
                print(f"    [OK] Exists ({len(df)} rows)")
            except:
                print(f"    [WARN] Exists but cannot read")
        else:
            print(f"    [INFO] Not generated yet (auto-created after labeling)")


def verify_layer2():
    """驗證 Layer 2 配置"""
    print("\n" + "="*80)
    print("Layer 2 Path Verification")
    print("="*80)

    # 檢查基礎目錄
    print(f"\nBase Directory: {LAYER2_BASE_DIR}")
    if LAYER2_BASE_DIR.exists():
        print("  [OK] Exists")
    else:
        print("  [WARN] Not exists (will create)")
        LAYER2_BASE_DIR.mkdir(parents=True, exist_ok=True)
        print("  [OK] Created")

    # 檢查數據目錄
    print(f"\nData Directory: {LAYER2_DATA_DIR}")
    if LAYER2_DATA_DIR.exists():
        print("  [OK] Exists")
    else:
        print("  [WARN] Not exists (will create)")
        LAYER2_DATA_DIR.mkdir(parents=True, exist_ok=True)
        print("  [OK] Created")

    # 檢查視頻文件夾
    print("\nVideo Classification Folders:")
    for label, folder in LAYER2_VIDEO_FOLDERS.items():
        print(f"  * {label}: {folder}")
        if folder.exists():
            video_count = len(list(folder.glob("*.mp4")))
            print(f"    [OK] Exists ({video_count} videos)")
        else:
            print(f"    [WARN] Not exists (will create)")
            folder.mkdir(parents=True, exist_ok=True)
            print(f"    [OK] Created")

    # 檢查 Excel D
    print(f"\nExcel D: {EXCEL_D_PATH}")
    if EXCEL_D_PATH.exists():
        import pandas as pd
        try:
            df = pd.read_excel(EXCEL_D_PATH)
            print(f"  [OK] Exists ({len(df)} rows)")
        except:
            print(f"  [WARN] Exists but cannot read")
    else:
        print(f"  [INFO] Not generated yet (auto-created after AI detection)")


def verify_components():
    """驗證關鍵組件"""
    print("\n" + "="*80)
    print("Component Files Verification")
    print("="*80)

    components = [
        ("Config File", "config.py"),
        ("Backend Server", "backend/server.py"),
        ("Downloader (Classified)", "downloader/tiktok_downloader_classified.py"),
        ("Feature Extractor (Layer 1)", "analyzer/feature_extractor_layer1.py"),
        ("Big Data Analyzer", "analyzer/big_data_analyzer.py"),
        ("Module Optimizer", "auto_reconstructor/module_optimizer.py"),
        ("Layer 1 Pipeline", "pipeline/layer1_pipeline.py"),
        ("Layer 2 Pipeline", "pipeline/layer2_pipeline.py"),
        ("Chrome Extension", "chrome_extension/manifest.json"),
    ]

    for name, path in components:
        full_path = BASE_DIR / path
        print(f"  * {name}: ", end="")
        if full_path.exists():
            print("[OK]")
        else:
            print(f"[ERROR] Missing: {path}")


def verify_dependencies():
    """驗證依賴庫"""
    print("\n" + "="*80)
    print("Dependencies Verification")
    print("="*80)

    dependencies = [
        "flask",
        "flask_cors",
        "pandas",
        "openpyxl",
        "cv2",
        "scipy",
        "numpy"
    ]

    for dep in dependencies:
        print(f"  * {dep}: ", end="")
        try:
            if dep == "cv2":
                import cv2
            elif dep == "flask_cors":
                from flask_cors import CORS
            else:
                __import__(dep)
            print("[OK]")
        except ImportError:
            print("[ERROR] Not installed")
            print(f"    Install: pip install {dep if dep != 'cv2' else 'opencv-python'}")

    # 檢查 yt-dlp
    print(f"  * yt-dlp: ", end="")
    import subprocess
    try:
        result = subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"[OK] (version: {version})")
        else:
            print("[ERROR] Cannot run")
    except FileNotFoundError:
        print("[ERROR] Not installed")
        print("    Install: pip install yt-dlp")
    except Exception as e:
        print(f"[WARN] {e}")


def main():
    """主程式"""
    print("\n" + "="*80)
    print("TSAR-RAPTOR System Verification")
    print("="*80)

    # 確保所有目錄存在
    ensure_directories()

    # 驗證 Layer 1
    verify_layer1()

    # 驗證 Layer 2
    verify_layer2()

    # 驗證組件
    verify_components()

    # 驗證依賴
    verify_dependencies()

    # 總結
    print("\n" + "="*80)
    print("Verification Complete!")
    print("="*80)
    print("\nNext Steps:")
    print("  1. Start backend: cd backend && python server.py")
    print("  2. Install Chrome extension: chrome://extensions/")
    print("  3. Start labeling: https://www.tiktok.com/foryou")
    print("  4. Run pipeline: cd pipeline && python layer1_pipeline.py")
    print("\nFor detailed guide, see: README_LAYER1.md or QUICKSTART.md")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

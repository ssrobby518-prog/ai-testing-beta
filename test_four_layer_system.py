#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四層架構系統測試腳本
====================

驗證系統是否正確安裝和配置

運行方式：
    python test_four_layer_system.py
"""

import os
import sys
import importlib

def print_header(text):
    """打印標題"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)

def print_success(text):
    """打印成功信息"""
    print(f"✅ {text}")

def print_error(text):
    """打印錯誤信息"""
    print(f"❌ {text}")

def print_warning(text):
    """打印警告信息"""
    print(f"⚠️  {text}")

def test_python_version():
    """測試 Python 版本"""
    print_header("測試 1: Python 版本")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python 版本: {version_str} (符合要求 >= 3.8)")
        return True
    else:
        print_error(f"Python 版本: {version_str} (需要 >= 3.8)")
        return False

def test_dependencies():
    """測試依賴庫"""
    print_header("測試 2: 依賴庫")

    required_packages = {
        'numpy': 'NumPy',
        'cv2': 'OpenCV (cv2)',
        'pandas': 'Pandas',
        'openpyxl': 'openpyxl',
        'pymediainfo': 'pymediainfo'
    }

    all_installed = True

    for package, name in required_packages.items():
        try:
            importlib.import_module(package)
            print_success(f"{name} 已安裝")
        except ImportError:
            print_error(f"{name} 未安裝 - 請運行: pip install {package if package != 'cv2' else 'opencv-python'}")
            all_installed = False

    return all_installed

def test_core_files():
    """測試核心文件是否存在"""
    print_header("測試 3: 核心文件")

    required_files = {
        'core/four_layer_system.py': '四層架構核心',
        'core/generation_analyzer.py': '第二層分析器',
        'core/detection_engine.py': '檢測引擎',
        'autotesting_four_layer.py': '四層總控程式',
        'autotesting.py': '傳統檢測系統',
        'README.md': '主要文檔',
        'FOUR_LAYER_ARCHITECTURE_SUMMARY.md': '架構總結',
        'QUICKSTART_FOUR_LAYER.md': '快速開始'
    }

    all_exist = True

    for file, desc in required_files.items():
        if os.path.exists(file):
            print_success(f"{desc}: {file}")
        else:
            print_error(f"{desc} 不存在: {file}")
            all_exist = False

    return all_exist

def test_modules():
    """測試檢測模組"""
    print_header("測試 4: 檢測模組")

    required_modules = [
        'frequency_analyzer',
        'texture_noise_detector',
        'model_fingerprint_detector',
        'lighting_geometry_checker',
        'heartbeat_detector',
        'blink_dynamics_analyzer',
        'text_fingerprinting',
        'sensor_noise_authenticator',
        'physics_violation_detector'
    ]

    all_exist = True

    for module_name in required_modules:
        module_path = f'modules/{module_name}.py'
        if os.path.exists(module_path):
            print_success(f"模組: {module_name}")
        else:
            print_error(f"模組不存在: {module_name}")
            all_exist = False

    return all_exist

def test_directory_structure():
    """測試目錄結構"""
    print_header("測試 5: 目錄結構")

    required_dirs = {
        'input': '輸入目錄（視頻）',
        'output': '輸出目錄（報告）',
        'data_pools': '資料池目錄',
        'modules': '檢測模組目錄',
        'core': '核心系統目錄',
        'tiktok_labeler': 'TikTok 工具目錄'
    }

    all_exist = True

    for dir_name, desc in required_dirs.items():
        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            print_success(f"{desc}: {dir_name}/")
        else:
            print_warning(f"{desc} 不存在，將自動創建: {dir_name}/")
            try:
                os.makedirs(dir_name, exist_ok=True)
                print_success(f"已創建: {dir_name}/")
            except Exception as e:
                print_error(f"創建失敗: {dir_name}/ - {e}")
                all_exist = False

    return all_exist

def test_import_core_modules():
    """測試導入核心模組"""
    print_header("測試 6: 導入核心模組")

    try:
        from core import four_layer_system
        print_success("導入成功: core.four_layer_system")
    except Exception as e:
        print_error(f"導入失敗: core.four_layer_system - {e}")
        return False

    try:
        from core import generation_analyzer
        print_success("導入成功: core.generation_analyzer")
    except Exception as e:
        print_error(f"導入失敗: core.generation_analyzer - {e}")
        return False

    # 測試數據結構
    try:
        from core.four_layer_system import (
            GenerationMechanism,
            EconomicBehavior,
            PoolType,
            DataArbitrationEngine,
            EconomicBehaviorClassifier,
            FourLayerSystemCoordinator
        )
        print_success("數據結構定義正確")
    except Exception as e:
        print_error(f"數據結構導入失敗: {e}")
        return False

    return True

def test_create_sample_data():
    """測試創建示例數據結構"""
    print_header("測試 7: 創建示例數據")

    try:
        from core.four_layer_system import (
            GenerationMechanism,
            VideoMetadata,
            GenerationAnalysisResult
        )

        # 創建示例元數據
        metadata = VideoMetadata(
            file_path="test.mp4",
            file_name="test.mp4",
            bitrate=1000000,
            fps=30.0,
            duration=10.0,
            resolution="1920x1080",
            download_source="test",
            timestamp="2025-12-20"
        )
        print_success("創建 VideoMetadata 成功")

        # 創建示例分析結果
        result = GenerationAnalysisResult(
            generation_mechanism=GenerationMechanism.AI,
            ai_probability=85.0,
            module_scores={"test_module": 80.0},
            weighted_scores={"test_module": 80.0},
            face_presence=0.5,
            static_ratio=0.1,
            is_phone_video=True,
            metadata=metadata,
            confidence=0.9
        )
        print_success("創建 GenerationAnalysisResult 成功")

        # 測試轉換為字典
        result_dict = result.to_dict()
        print_success("數據序列化成功")

        return True
    except Exception as e:
        print_error(f"創建示例數據失敗: {e}")
        return False

def test_arbitration_engine():
    """測試仲裁引擎"""
    print_header("測試 8: 仲裁引擎")

    try:
        from core.four_layer_system import (
            DataArbitrationEngine,
            GenerationMechanism,
            VideoMetadata,
            GenerationAnalysisResult
        )

        # 創建測試數據
        metadata = VideoMetadata(
            file_path="test.mp4",
            file_name="test.mp4",
            bitrate=1000000,
            fps=30.0,
            duration=10.0,
            resolution="1920x1080",
            download_source="test",
            timestamp="2025-12-20"
        )

        # 測試高置信度 AI
        result_ai = GenerationAnalysisResult(
            generation_mechanism=GenerationMechanism.AI,
            ai_probability=85.0,
            module_scores={
                "module1": 85.0,
                "module2": 83.0,
                "module3": 87.0
            },
            weighted_scores={
                "module1": 85.0,
                "module2": 83.0,
                "module3": 87.0
            },
            face_presence=0.5,
            static_ratio=0.1,
            is_phone_video=True,
            metadata=metadata,
            confidence=0.9
        )

        decision = DataArbitrationEngine.arbitrate(result_ai)
        print_success(f"仲裁決策: {decision.pool_assignment.value}")
        print_success(f"仲裁理由: {decision.decision_rationale}")

        return True
    except Exception as e:
        print_error(f"仲裁引擎測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函數"""
    print("\n" + "="*80)
    print("  四層架構系統測試")
    print("="*80)
    print("\n正在執行系統測試...\n")

    tests = [
        ("Python 版本", test_python_version),
        ("依賴庫", test_dependencies),
        ("核心文件", test_core_files),
        ("檢測模組", test_modules),
        ("目錄結構", test_directory_structure),
        ("導入核心模組", test_import_core_modules),
        ("示例數據", test_create_sample_data),
        ("仲裁引擎", test_arbitration_engine)
    ]

    results = []

    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print_error(f"測試 '{name}' 發生異常: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # 打印總結
    print_header("測試總結")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {name}")

    print(f"\n總計: {passed}/{total} 測試通過")

    if passed == total:
        print("\n" + "="*80)
        print("  🎉 所有測試通過！系統已正確安裝。")
        print("="*80)
        print("\n下一步:")
        print("  1. 將視頻放入 input/ 目錄")
        print("  2. 運行: python autotesting_four_layer.py")
        print("  3. 查看 data_pools/ 目錄的分配結果")
        print("\n快速開始指南: QUICKSTART_FOUR_LAYER.md")
        print("="*80 + "\n")
        return 0
    else:
        print("\n" + "="*80)
        print(f"  ⚠️  {total - passed} 個測試失敗，請檢查上述錯誤。")
        print("="*80)
        print("\n常見問題:")
        print("  1. 依賴庫缺失: pip install numpy opencv-python pandas openpyxl pymediainfo")
        print("  2. 文件缺失: 請確保已下載完整項目")
        print("  3. Python 版本: 需要 Python 3.8+")
        print("="*80 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())

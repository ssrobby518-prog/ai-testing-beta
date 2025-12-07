#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blue Team Defense System - 快速測試腳本
用於驗證所有新模組是否正常工作
"""

import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def test_imports():
    """測試所有依賴是否可導入"""
    print("\n" + "="*80)
    print("測試 1/5: 檢查依賴導入")
    print("="*80)

    results = {}

    # 核心依賴
    core_deps = ['cv2', 'numpy', 'pandas', 'pymediainfo']
    for dep in core_deps:
        try:
            __import__(dep)
            results[dep] = "✓ OK"
        except ImportError:
            results[dep] = "✗ MISSING (核心依賴，必須安裝)"

    # 可選依賴
    optional_deps = {
        'mediapipe': 'Facial Rigidity Analyzer',
        'torch': 'Spectral CNN Classifier',
        'xgboost': 'XGBoost Ensemble',
        'shap': 'SHAP Explainability'
    }

    for dep, feature in optional_deps.items():
        try:
            __import__(dep)
            results[dep] = f"✓ OK ({feature})"
        except ImportError:
            results[dep] = f"⚠ MISSING ({feature} 將降級)"

    for dep, status in results.items():
        print(f"  {dep:15} {status}")

    return all("✗" not in v for k, v in results.items() if k in core_deps)


def test_modules_load():
    """測試所有新模組是否能加載"""
    print("\n" + "="*80)
    print("測試 2/5: 加載新模組")
    print("="*80)

    modules_to_test = [
        'facial_rigidity_analyzer',
        'frequency_analyzer_v2',
        'spectral_cnn_classifier',
    ]

    results = {}

    for module_name in modules_to_test:
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                module_name,
                f'modules/{module_name}.py'
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            # 檢查是否有detect函數
            if hasattr(mod, 'detect'):
                results[module_name] = "✓ OK"
            else:
                results[module_name] = "✗ 缺少detect()函數"

        except Exception as e:
            results[module_name] = f"✗ 加載失敗: {str(e)[:50]}"

    for module, status in results.items():
        print(f"  {module:30} {status}")

    return all("✓" in v for v in results.values())


def test_xgboost_ensemble():
    """測試XGBoost集成引擎"""
    print("\n" + "="*80)
    print("測試 3/5: XGBoost集成引擎")
    print("="*80)

    try:
        from core.xgboost_ensemble import XGBoostEnsemble

        ensemble = XGBoostEnsemble()

        # 模擬輸入
        module_scores = {
            'model_fingerprint_detector': 75.0,
            'frequency_analyzer': 65.0,
            'sensor_noise_authenticator': 70.0,
            'physics_violation_detector': 60.0,
            'texture_noise_detector': 55.0,
            'text_fingerprinting': 50.0,
            'metadata_extractor': 45.0,
            'heartbeat_detector': 40.0,
            'blink_dynamics_analyzer': 42.0,
            'lighting_geometry_checker': 48.0,
            'av_sync_verifier': 46.0,
            'semantic_stylometry': 50.0,
        }

        metadata = {
            'bitrate': 1200000,
            'fps': 30.0,
            'face_presence': 0.8,
            'static_ratio': 0.1,
            'width': 1920,
            'height': 1080
        }

        result = ensemble.predict(module_scores, metadata)

        print(f"  AI Probability: {result.ai_probability:.2f}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Top Reasons: {result.top_reasons[:2]}")

        if 0 <= result.ai_probability <= 100:
            print("  ✓ XGBoost引擎正常工作")
            return True
        else:
            print("  ✗ AI概率超出範圍")
            return False

    except Exception as e:
        print(f"  ✗ 測試失敗: {e}")
        return False


def test_blue_team_controller():
    """測試藍隊總控"""
    print("\n" + "="*80)
    print("測試 4/5: 藍隊總控系統")
    print("="*80)

    try:
        # 檢查文件是否存在
        import os
        if not os.path.exists('autotesting_blue_team.py'):
            print("  ✗ autotesting_blue_team.py 不存在")
            return False

        print("  ✓ 總控文件存在")

        # 檢查配置
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "autotesting_blue_team",
            "autotesting_blue_team.py"
        )
        controller = importlib.util.module_from_spec(spec)

        print("  ✓ 總控可以導入")

        # 檢查BLUE_TEAM_MODULES配置
        if hasattr(controller, 'BLUE_TEAM_MODULES'):
            modules_count = len(controller.BLUE_TEAM_MODULES)
            print(f"  ✓ 配置了 {modules_count} 個模組")
            return True
        else:
            print("  ✗ 缺少BLUE_TEAM_MODULES配置")
            return False

    except Exception as e:
        print(f"  ✗ 測試失敗: {e}")
        return False


def test_documentation():
    """測試文檔是否完整"""
    print("\n" + "="*80)
    print("測試 5/5: 文檔完整性")
    print("="*80)

    import os

    docs_to_check = [
        'BLUE_TEAM_UPGRADE_GUIDE.md',
        'QUICK_START.md',
    ]

    results = {}

    for doc in docs_to_check:
        if os.path.exists(doc):
            size = os.path.getsize(doc)
            results[doc] = f"✓ OK ({size} bytes)"
        else:
            results[doc] = "✗ MISSING"

    for doc, status in results.items():
        print(f"  {doc:35} {status}")

    return all("✓" in v for v in results.values())


def main():
    """運行所有測試"""
    print("\n" + "="*80)
    print(" Blue Team Defense System - 系統測試")
    print("="*80)

    tests = [
        ("依賴檢查", test_imports),
        ("模組加載", test_modules_load),
        ("XGBoost引擎", test_xgboost_ensemble),
        ("藍隊總控", test_blue_team_controller),
        ("文檔完整性", test_documentation),
    ]

    results = []

    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            logging.error(f"{name} 測試異常: {e}")
            results.append((name, False))

    # 總結
    print("\n" + "="*80)
    print(" 測試總結")
    print("="*80)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {name:20} {status}")

    print(f"\n  總計: {passed}/{total} 通過")

    if passed == total:
        print("\n  🎉 所有測試通過！藍隊系統已就緒。")
        print("\n  下一步:")
        print("    1. 將視頻放入 input/ 目錄")
        print("    2. 運行: python autotesting_blue_team.py")
        print("    3. 查看: output/blue_team_report_*.xlsx")
        return 0
    else:
        print("\n  ⚠ 部分測試失敗，請檢查上述錯誤。")
        print("\n  提示:")
        print("    - 核心依賴失敗：運行 pip install opencv-python numpy pandas pymediainfo")
        print("    - 可選依賴失敗：系統會自動降級，不影響核心功能")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSAR-RAPTOR Layer 1 Pipeline
人工主導標註完整流水線

設計原則:
- 第一性原理: 人類判定 → 數據分析 → 模組優化
- 沙皇炸彈: 級聯學習，數據驅動
- 猛禽3: 一鍵執行，全自動

完整流程:
1. Chrome擴展標註 → Excel A
2. 批量下載並自動分類到文件夾
3. 特徵提取 → Excel B
4. 大數據分析 → Excel C
5. 模組自動優化
"""

import sys
from pathlib import Path
import logging
import argparse
from typing import Dict

# 添加項目路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 導入配置
from config import (
    EXCEL_A_PATH, EXCEL_B_PATH, EXCEL_C_PATH,
    LAYER1_BASE_DIR, LAYER1_DATA_DIR,
    ensure_directories
)

# 導入各組件
from downloader.tiktok_downloader_classified import TikTokDownloaderClassified
from analyzer.feature_extractor_layer1 import FeatureExtractorLayer1
from analyzer.big_data_analyzer import BigDataAnalyzer
from auto_reconstructor.module_optimizer import ModuleOptimizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Layer1Pipeline:
    """Layer 1 自我學習流水線總控"""

    def __init__(self):
        """初始化"""
        # 確保所有目錄存在
        ensure_directories()

        logger.info("Layer 1 流水線初始化完成")
        logger.info(f"  • 基礎目錄: {LAYER1_BASE_DIR}")
        logger.info(f"  • Excel A: {EXCEL_A_PATH}")
        logger.info(f"  • Excel B: {EXCEL_B_PATH}")
        logger.info(f"  • Excel C: {EXCEL_C_PATH}")

    def run_full_pipeline(self) -> Dict:
        """
        運行完整 Layer 1 流水線

        流程:
        1. 檢查 Excel A 是否有標註
        2. 批量下載視頻並自動分類
        3. 特徵提取 → Excel B
        4. 大數據分析 → Excel C
        5. 模組優化

        Returns:
            執行統計
        """
        logger.info(f"\n{'='*80}")
        logger.info("🚀 TSAR-RAPTOR Layer 1 人工主導標註流水線 - 啟動")
        logger.info(f"{'='*80}\n")

        stats = {}

        # Step 1: 檢查 Excel A
        if not EXCEL_A_PATH.exists():
            logger.error(f"❌ Excel A 不存在: {EXCEL_A_PATH}")
            logger.error("   請先使用 Chrome 擴展進行標註")
            return {}

        # Step 2: 批量下載並自動分類
        logger.info("📥 [Step 1/4] 批量下載並自動分類視頻...")
        downloader = TikTokDownloaderClassified(
            excel_a_path=str(EXCEL_A_PATH),
            max_workers=4
        )
        download_stats = downloader.download_from_excel_a()
        stats['download'] = download_stats

        # Step 3: 特徵提取
        logger.info("\n🔬 [Step 2/4] 特徵提取...")
        extractor = FeatureExtractorLayer1(
            output_excel_b=str(EXCEL_B_PATH),
            max_workers=4,
            sample_frames=30
        )
        df_features = extractor.batch_extract()
        stats['features'] = {'total': len(df_features)}

        # Step 4: 大數據分析
        logger.info("\n📊 [Step 3/4] 大數據分析...")
        analyzer = BigDataAnalyzer(
            excel_b_path=str(EXCEL_B_PATH),
            output_excel_c=str(EXCEL_C_PATH)
        )
        analysis_results = analyzer.analyze()
        stats['analysis'] = {'features_analyzed': len(analysis_results.get('ranked_features', []))}

        # Step 5: 模組優化
        logger.info("\n⚙️  [Step 4/4] 模組自動優化...")
        optimized_config_path = LAYER1_DATA_DIR / "optimized_config.json"
        optimizer = ModuleOptimizer(
            excel_c_path=str(EXCEL_C_PATH),
            config_output=str(optimized_config_path)
        )
        optimized_config = optimizer.optimize()
        stats['optimization'] = {'modules_optimized': len(optimized_config.get('module_weights', {}))}

        # 最終統計
        logger.info(f"\n{'='*80}")
        logger.info("✅ Layer 1 流水線完成！")
        logger.info(f"{'='*80}")
        logger.info(f"  • 下載視頻: {download_stats.get('success', 0)} 成功, {download_stats.get('failed', 0)} 失敗")
        if 'by_category' in download_stats:
            logger.info(f"    分類統計:")
            logger.info(f"      - Real: {download_stats['by_category']['real']}")
            logger.info(f"      - AI: {download_stats['by_category']['ai']}")
            logger.info(f"      - Uncertain: {download_stats['by_category']['uncertain']}")
            logger.info(f"      - Movies: {download_stats['by_category']['exclude']}")
        logger.info(f"  • 提取特徵: {len(df_features)} 個視頻")
        logger.info(f"  • 分析特徵: {len(analysis_results.get('ranked_features', []))} 個特徵")
        logger.info(f"  • 優化模組: {len(optimized_config.get('module_weights', {}))} 個模組")
        logger.info(f"{'='*80}\n")

        return stats


def main():
    """主程式"""
    parser = argparse.ArgumentParser(description="Layer 1 人工主導標註流水線")

    parser.add_argument(
        '--check-paths',
        action='store_true',
        help='檢查路徑配置'
    )

    args = parser.parse_args()

    # 創建流水線
    pipeline = Layer1Pipeline()

    if args.check_paths:
        print(f"\n{'='*80}")
        print("路徑配置:")
        print(f"{'='*80}")
        print(f"基礎目錄: {LAYER1_BASE_DIR}")
        print(f"數據目錄: {LAYER1_DATA_DIR}")
        print(f"\nExcel 文件:")
        print(f"  • Excel A: {EXCEL_A_PATH}")
        print(f"  • Excel B: {EXCEL_B_PATH}")
        print(f"  • Excel C: {EXCEL_C_PATH}")
        print(f"\n視頻文件夾:")
        from config import LAYER1_VIDEO_FOLDERS
        for label, folder in LAYER1_VIDEO_FOLDERS.items():
            print(f"  • {label}: {folder}")
        print(f"{'='*80}\n")
        return

    # 執行完整流水線
    stats = pipeline.run_full_pipeline()

    if stats:
        print(f"\n✅ Layer 1 流水線執行完成！")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSAR-RAPTOR Self-Learning Pipeline
完整自我學習流水線 - 整合兩層系統

設計原則:
- 第一性原理: 數據 → 分析 → 優化 → 循環
- 沙皇炸彈: 級聯學習，指數增長
- 猛禽3: 全自動化，零人工干預（除標註外）

完整流程:
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: 人工主導標註                                       │
│ ──────────────────────────────────────────────────────────│
│ 1. Chrome擴展標註 → Excel A                                 │
│ 2. 批量下載視頻                                             │
│ 3. 特徵提取 → Excel B                                       │
│ 4. 大數據分析 → Excel C                                     │
│ 5. 模組自動優化                                             │
└─────────────────────────────────────────────────────────────┘
              ↓ 升級AI模組
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: AI主導自動化                                       │
│ ──────────────────────────────────────────────────────────│
│ 1. 自動下載2000個視頻                                       │
│ 2. AI檢測模組判定                                           │
│ 3. 提取不確定視頻                                           │
│ 4. 本地Tinder復審                                           │
│ 5. 持續訓練優化                                             │
│ ↻ 循環                                                     │
└─────────────────────────────────────────────────────────────┘
"""

import sys
from pathlib import Path
import logging
import argparse
from typing import Dict

# 添加項目路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 導入各組件
from downloader.tiktok_downloader import TikTokDownloader
from analyzer.feature_extractor import FeatureExtractor
from analyzer.big_data_analyzer import BigDataAnalyzer
from auto_reconstructor.module_optimizer import ModuleOptimizer
from local_reviewer.review_interface import LocalReviewer, load_uncertain_videos_from_detection_results

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SelfLearningPipeline:
    """自我學習流水線總控"""

    def __init__(
        self,
        excel_a_path: str,
        video_dir: str = "../data/tiktok_videos",
        data_dir: str = "../data/tiktok_labels"
    ):
        """
        Args:
            excel_a_path: Excel A 路徑（人工標註數據）
            video_dir: 視頻下載目錄
            data_dir: 數據輸出目錄
        """
        self.excel_a_path = Path(excel_a_path)
        self.video_dir = Path(video_dir)
        self.data_dir = Path(data_dir)

        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 文件路徑
        self.excel_b_path = self.data_dir / "excel_b_features.xlsx"
        self.excel_c_path = self.data_dir / "excel_c_analysis.xlsx"
        self.optimized_config_path = self.data_dir / "optimized_config.json"

        logger.info("自我學習流水線初始化完成")
        logger.info(f"  • Excel A: {self.excel_a_path}")
        logger.info(f"  • 視頻目錄: {self.video_dir}")
        logger.info(f"  • 數據目錄: {self.data_dir}")

    def run_layer1_pipeline(self) -> Dict:
        """
        運行第一層流水線（人工主導）

        流程:
        1. 檢查 Excel A 是否有新標註
        2. 批量下載視頻
        3. 特徵提取 → Excel B
        4. 大數據分析 → Excel C
        5. 模組優化

        Returns:
            執行統計
        """
        logger.info(f"\n{'='*80}")
        logger.info("Layer 1: 人工主導標註流水線")
        logger.info(f"{'='*80}\n")

        stats = {}

        # Step 1: 檢查 Excel A
        if not self.excel_a_path.exists():
            logger.error(f"❌ Excel A 不存在: {self.excel_a_path}")
            logger.error("   請先使用 Chrome 擴展進行標註")
            return {}

        # Step 2: 批量下載視頻
        logger.info("📥 [Step 1/4] 批量下載視頻...")
        downloader = TikTokDownloader(
            excel_a_path=str(self.excel_a_path),
            download_dir=str(self.video_dir),
            max_workers=4
        )
        download_stats = downloader.download_from_excel_a(exclude_labels=['exclude'])
        stats['download'] = download_stats

        # Step 3: 特徵提取
        logger.info("\n🔬 [Step 2/4] 特徵提取...")
        extractor = FeatureExtractor(
            video_dir=str(self.video_dir),
            output_excel_b=str(self.excel_b_path),
            max_workers=4,
            sample_frames=30
        )
        df_features = extractor.batch_extract()
        stats['features'] = {'total': len(df_features)}

        # Step 4: 大數據分析
        logger.info("\n📊 [Step 3/4] 大數據分析...")
        analyzer = BigDataAnalyzer(
            excel_b_path=str(self.excel_b_path),
            output_excel_c=str(self.excel_c_path)
        )
        analysis_results = analyzer.analyze()
        stats['analysis'] = {'features_analyzed': len(analysis_results.get('ranked_features', []))}

        # Step 5: 模組優化
        logger.info("\n⚙️  [Step 4/4] 模組自動優化...")
        optimizer = ModuleOptimizer(
            excel_c_path=str(self.excel_c_path),
            config_output=str(self.optimized_config_path)
        )
        optimized_config = optimizer.optimize()
        stats['optimization'] = {'modules_optimized': len(optimized_config.get('module_weights', {}))}

        logger.info(f"\n{'='*80}")
        logger.info("✅ Layer 1 流水線完成！")
        logger.info(f"{'='*80}")
        logger.info(f"  • 下載視頻: {download_stats['success']} 成功, {download_stats['failed']} 失敗")
        logger.info(f"  • 提取特徵: {len(df_features)} 個視頻")
        logger.info(f"  • 分析特徵: {len(analysis_results.get('ranked_features', []))} 個特徵")
        logger.info(f"  • 優化模組: {len(optimized_config.get('module_weights', {}))} 個模組")
        logger.info(f"{'='*80}\n")

        return stats

    def run_layer2_pipeline(
        self,
        detection_results_csv: str,
        enable_review: bool = True
    ) -> Dict:
        """
        運行第二層流水線（AI主導自動化）

        流程:
        1. 讀取AI檢測結果
        2. 提取不確定視頻
        3. 本地Tinder復審
        4. 持續訓練（整合到autotesting_integrated.py）

        Args:
            detection_results_csv: AI檢測結果CSV
            enable_review: 是否啟用人工復審

        Returns:
            執行統計
        """
        logger.info(f"\n{'='*80}")
        logger.info("Layer 2: AI主導自動化流水線")
        logger.info(f"{'='*80}\n")

        stats = {}

        # Step 1: 加載不確定視頻
        logger.info("📋 [Step 1/2] 加載不確定視頻...")
        uncertain_videos = load_uncertain_videos_from_detection_results(
            detection_results_csv,
            str(self.video_dir)
        )
        stats['uncertain_count'] = len(uncertain_videos)

        if not uncertain_videos:
            logger.info("✅ 沒有不確定視頻，無需復審")
            return stats

        # Step 2: 本地復審
        if enable_review:
            logger.info(f"\n👁️  [Step 2/2] 本地Tinder復審 ({len(uncertain_videos)} 個視頻)...")
            reviewer = LocalReviewer(
                uncertain_videos=uncertain_videos,
                output_csv=str(self.data_dir / "layer2_review_results.csv")
            )
            review_stats = reviewer.batch_review()
            stats['review'] = review_stats
        else:
            logger.info("⏭️  跳過人工復審（enable_review=False）")
            stats['review'] = {'skipped': True}

        logger.info(f"\n{'='*80}")
        logger.info("✅ Layer 2 流水線完成！")
        logger.info(f"{'='*80}")
        if enable_review and 'review' in stats:
            logger.info(f"  • 不確定視頻: {stats['uncertain_count']}")
            logger.info(f"  • 已復審: {stats['review'].get('reviewed', 0)}")
            logger.info(f"  • Real: {stats['review'].get('real', 0)}")
            logger.info(f"  • AI: {stats['review'].get('ai', 0)}")
        logger.info(f"{'='*80}\n")

        return stats

    def run_full_pipeline(
        self,
        run_layer1: bool = True,
        run_layer2: bool = False,
        detection_results_csv: str = None
    ) -> Dict:
        """
        運行完整流水線（兩層）

        Args:
            run_layer1: 是否運行 Layer 1
            run_layer2: 是否運行 Layer 2
            detection_results_csv: AI檢測結果CSV（Layer 2 需要）

        Returns:
            完整執行統計
        """
        logger.info(f"\n{'='*100}")
        logger.info("🚀 TSAR-RAPTOR Self-Learning Pipeline - 啟動")
        logger.info(f"{'='*100}\n")

        full_stats = {}

        # Layer 1
        if run_layer1:
            layer1_stats = self.run_layer1_pipeline()
            full_stats['layer1'] = layer1_stats

        # Layer 2
        if run_layer2:
            if not detection_results_csv:
                logger.error("❌ Layer 2 需要提供 detection_results_csv")
            else:
                layer2_stats = self.run_layer2_pipeline(detection_results_csv)
                full_stats['layer2'] = layer2_stats

        logger.info(f"\n{'='*100}")
        logger.info("🎉 完整流水線執行完畢！")
        logger.info(f"{'='*100}\n")

        return full_stats


def main():
    """主程式"""
    parser = argparse.ArgumentParser(description="TSAR-RAPTOR 自我學習流水線")

    # 基本參數
    parser.add_argument(
        '--excel-a',
        type=str,
        default='../data/tiktok_labels/excel_a_labels_raw.xlsx',
        help='Excel A 路徑（人工標註數據）'
    )
    parser.add_argument(
        '--video-dir',
        type=str,
        default='../data/tiktok_videos',
        help='視頻目錄'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../data/tiktok_labels',
        help='數據目錄'
    )

    # 流程控制
    parser.add_argument(
        '--layer1',
        action='store_true',
        help='運行 Layer 1（人工主導標註流水線）'
    )
    parser.add_argument(
        '--layer2',
        action='store_true',
        help='運行 Layer 2（AI主導自動化流水線）'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='運行完整流水線（Layer 1 + Layer 2）'
    )

    # Layer 2 參數
    parser.add_argument(
        '--detection-results',
        type=str,
        help='AI檢測結果CSV（Layer 2 需要）'
    )

    args = parser.parse_args()

    # 創建流水線
    pipeline = SelfLearningPipeline(
        excel_a_path=args.excel_a,
        video_dir=args.video_dir,
        data_dir=args.data_dir
    )

    # 執行流程
    if args.full:
        stats = pipeline.run_full_pipeline(
            run_layer1=True,
            run_layer2=True,
            detection_results_csv=args.detection_results
        )
    elif args.layer1:
        stats = pipeline.run_layer1_pipeline()
    elif args.layer2:
        if not args.detection_results:
            logger.error("❌ Layer 2 需要提供 --detection-results")
            return
        stats = pipeline.run_layer2_pipeline(args.detection_results)
    else:
        parser.print_help()
        print("\n提示: 使用 --layer1, --layer2 或 --full 選擇運行模式")
        return

    print(f"\n✅ 流水線執行完成！")


if __name__ == "__main__":
    main()

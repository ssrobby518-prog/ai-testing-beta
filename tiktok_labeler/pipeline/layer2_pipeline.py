#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSAR-RAPTOR Layer 2 自動化流水線
AI主導自動化 - 批量下載 → AI檢測 → 自動分類 → 人工復審

設計原則:
- 第一性原理: AI主導，人類輔助
- 沙皇炸彈: 海量數據，自動化處理
- 猛禽3: 一鍵執行，全自動流水線

完整流程:
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: AI主導自動化流水線                                 │
│ ──────────────────────────────────────────────────────────│
│ 1. 批量下載TikTok視頻（2000個）                             │
│ 2. AI檢測模組自動分類（real/ai/not sure/電影動畫）          │
│ 3. 生成 Excel D（分類結果 + 特徵記錄）                      │
│ 4. 自動移動文件到對應文件夾                                 │
│ 5. 從 "not sure" 文件夾提取不確定視頻                       │
│ 6. 本地Tinder復審系統                                       │
│ 7. 復審後自動移動到正確分類文件夾                           │
│ 8. 更新 Excel D 人工復審結果                                │
│ ↻ 循環優化 → 99% 準確率                                    │
└─────────────────────────────────────────────────────────────┘
"""

import sys
from pathlib import Path
import logging
from typing import Dict, List
import argparse

# 添加項目路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 導入各組件
from mass_downloader.url_scraper import TikTokURLScraper
from mass_downloader.mass_downloader import TikTokMassDownloader
from ai_classifier.ai_detector import AIDetectionClassifier
from ai_classifier.excel_d_generator import ExcelDGenerator
from file_organizer.auto_classifier import FileAutoClassifier
from local_reviewer.review_interface import LocalReviewer, load_uncertain_videos_from_folder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Layer2Pipeline:
    """Layer 2 自動化流水線總控"""

    def __init__(
        self,
        url_list_file: str = None,
        download_dir: str = "../../tiktok videos download",
        target_count: int = 2000,
        max_workers_download: int = 8,
        max_workers_detect: int = 4
    ):
        """
        Args:
            url_list_file: URL列表文件路徑
            download_dir: 下載目錄（也是分類根目錄）
            target_count: 目標下載數量
            max_workers_download: 下載並行數
            max_workers_detect: 檢測並行數
        """
        self.url_list_file = url_list_file
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.target_count = target_count
        self.max_workers_download = max_workers_download
        self.max_workers_detect = max_workers_detect

        # 文件路徑
        self.data_dir = self.download_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.excel_d_path = self.data_dir / "excel_d_detection_results.xlsx"

        logger.info("Layer 2 流水線初始化完成")
        logger.info(f"  • 下載目錄: {self.download_dir}")
        logger.info(f"  • 目標數量: {target_count}")
        logger.info(f"  • Excel D: {self.excel_d_path}")

    def run_full_pipeline(
        self,
        skip_download: bool = False,
        skip_detection: bool = False,
        skip_classification: bool = False,
        skip_review: bool = False
    ) -> Dict:
        """
        運行完整 Layer 2 流水線

        Args:
            skip_download: 跳過下載步驟
            skip_detection: 跳過AI檢測步驟
            skip_classification: 跳過文件分類步驟
            skip_review: 跳過人工復審步驟

        Returns:
            執行統計
        """
        logger.info(f"\n{'='*100}")
        logger.info("🚀 TSAR-RAPTOR Layer 2 AI主導自動化流水線 - 啟動")
        logger.info(f"{'='*100}\n")

        stats = {}

        # Step 1: 批量下載視頻
        if not skip_download:
            logger.info(f"📥 [Step 1/6] 批量下載TikTok視頻...")
            download_stats = self._batch_download()
            stats['download'] = download_stats
        else:
            logger.info("⏭️  跳過下載步驟")

        # Step 2: AI檢測分類
        if not skip_detection:
            logger.info(f"\n🤖 [Step 2/6] AI檢測模組自動分類...")
            detection_results = self._ai_detection()
            stats['detection'] = {
                'total': len(detection_results),
                'real': sum(1 for r in detection_results if r['classification'] == 'REAL'),
                'ai': sum(1 for r in detection_results if r['classification'] == 'AI'),
                'not_sure': sum(1 for r in detection_results if r['classification'] == 'NOT_SURE'),
                'movie': sum(1 for r in detection_results if r['classification'] == '電影動畫')
            }
        else:
            logger.info("⏭️  跳過AI檢測步驟")
            detection_results = []

        # Step 3: 生成 Excel D
        if detection_results:
            logger.info(f"\n📊 [Step 3/6] 生成 Excel D...")
            df_excel_d = self._generate_excel_d(detection_results)
            stats['excel_d'] = {'rows': len(df_excel_d)}
        else:
            logger.info("⏭️  無檢測結果，跳過 Excel D 生成")

        # Step 4: 自動文件分類
        if not skip_classification and detection_results:
            logger.info(f"\n📦 [Step 4/6] 自動移動文件到分類文件夾...")
            classification_stats = self._classify_files(detection_results)
            stats['classification'] = classification_stats
        else:
            logger.info("⏭️  跳過文件分類步驟")

        # Step 5: 加載 not sure 視頻
        logger.info(f"\n🔍 [Step 5/6] 加載不確定視頻...")
        not_sure_folder = self.download_dir / "not sure"
        uncertain_videos = load_uncertain_videos_from_folder(str(not_sure_folder))
        stats['uncertain_count'] = len(uncertain_videos)

        if not uncertain_videos:
            logger.info("✅ 沒有不確定視頻，無需復審")
            skip_review = True

        # Step 6: 本地Tinder復審
        if not skip_review and uncertain_videos:
            logger.info(f"\n👁️  [Step 6/6] 本地Tinder復審 ({len(uncertain_videos)} 個視頻)...")
            review_stats = self._review_uncertain_videos(uncertain_videos)
            stats['review'] = review_stats
        else:
            logger.info("⏭️  跳過人工復審步驟")

        # 最終統計
        logger.info(f"\n{'='*100}")
        logger.info("🎉 Layer 2 流水線執行完畢！")
        logger.info(f"{'='*100}")
        if 'download' in stats:
            logger.info(f"  • 下載視頻: {stats['download'].get('success', 0)} 成功")
        if 'detection' in stats:
            logger.info(f"  • AI檢測: {stats['detection']['total']} 個視頻")
            logger.info(f"    - REAL: {stats['detection']['real']}")
            logger.info(f"    - AI: {stats['detection']['ai']}")
            logger.info(f"    - NOT_SURE: {stats['detection']['not_sure']}")
            logger.info(f"    - 電影動畫: {stats['detection']['movie']}")
        if 'classification' in stats:
            logger.info(f"  • 文件分類: {stats['classification'].get('moved', 0)} 個已移動")
        if 'review' in stats:
            logger.info(f"  • 人工復審: {stats['review'].get('reviewed', 0)} 個已復審")
        logger.info(f"{'='*100}\n")

        return stats

    def _batch_download(self) -> Dict:
        """Step 1: 批量下載視頻"""
        if not self.url_list_file or not Path(self.url_list_file).exists():
            logger.warning(f"⚠️  URL列表文件不存在: {self.url_list_file}")
            logger.warning("   使用 url_scraper.py 生成URL列表")
            return {'success': 0, 'failed': 0}

        # 創建下載器
        downloader = TikTokMassDownloader(
            url_list_file=self.url_list_file,
            download_dir=str(self.download_dir),
            max_workers=self.max_workers_download,
            target_count=self.target_count
        )

        # 執行下載
        stats = downloader.batch_download()
        return stats

    def _ai_detection(self) -> List[Dict]:
        """Step 2: AI檢測分類"""
        # 創建檢測器
        detector = AIDetectionClassifier(
            video_dir=str(self.download_dir),
            max_workers=self.max_workers_detect
        )

        # 批量檢測
        results = detector.batch_detect()
        return results

    def _generate_excel_d(self, detection_results: List[Dict]) -> "pd.DataFrame":
        """Step 3: 生成 Excel D"""
        # 創建生成器
        generator = ExcelDGenerator(
            video_dir=str(self.download_dir),
            output_excel_d=str(self.excel_d_path)
        )

        # 生成 Excel D
        df = generator.generate_from_detection_results(detection_results)
        return df

    def _classify_files(self, detection_results: List[Dict]) -> Dict:
        """Step 4: 自動文件分類"""
        # 創建分類器
        classifier = FileAutoClassifier(
            source_dir=str(self.download_dir),
            base_output_dir=str(self.download_dir)
        )

        # 執行分類
        stats = classifier.classify_from_detection_results(
            detection_results,
            move_files=True
        )
        return stats

    def _review_uncertain_videos(self, uncertain_videos: List[str]) -> Dict:
        """Step 6: 本地Tinder復審"""
        # 創建復審器
        reviewer = LocalReviewer(
            uncertain_videos=uncertain_videos,
            output_csv=str(self.data_dir / "layer2_review_results.csv"),
            base_output_dir=str(self.download_dir),
            excel_d_path=str(self.excel_d_path),
            auto_move_files=True
        )

        # 執行復審
        stats = reviewer.batch_review()
        return stats


def main():
    """主程式"""
    parser = argparse.ArgumentParser(description="Layer 2 AI主導自動化流水線")

    # 基本參數
    parser.add_argument(
        '--url-list',
        type=str,
        help='URL列表文件路徑'
    )
    parser.add_argument(
        '--download-dir',
        type=str,
        default='../../tiktok videos download',
        help='下載目錄'
    )
    parser.add_argument(
        '--target',
        type=int,
        default=2000,
        help='目標下載數量'
    )
    parser.add_argument(
        '--download-workers',
        type=int,
        default=8,
        help='下載並行數'
    )
    parser.add_argument(
        '--detect-workers',
        type=int,
        default=4,
        help='檢測並行數'
    )

    # 流程控制
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='跳過下載步驟'
    )
    parser.add_argument(
        '--skip-detection',
        action='store_true',
        help='跳過AI檢測步驟'
    )
    parser.add_argument(
        '--skip-classification',
        action='store_true',
        help='跳過文件分類步驟'
    )
    parser.add_argument(
        '--skip-review',
        action='store_true',
        help='跳過人工復審步驟'
    )

    args = parser.parse_args()

    # 創建流水線
    pipeline = Layer2Pipeline(
        url_list_file=args.url_list,
        download_dir=args.download_dir,
        target_count=args.target,
        max_workers_download=args.download_workers,
        max_workers_detect=args.detect_workers
    )

    # 執行完整流水線
    stats = pipeline.run_full_pipeline(
        skip_download=args.skip_download,
        skip_detection=args.skip_detection,
        skip_classification=args.skip_classification,
        skip_review=args.skip_review
    )

    print(f"\n✅ Layer 2 流水線執行完成！")


if __name__ == "__main__":
    main()

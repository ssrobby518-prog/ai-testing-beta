#!/usr/bin/env python3
"""
TSAR-RAPTOR Integrated System - 完整人眼輔助學習系統
集成 AI檢測 + 人工標註 + 持續訓練

整合流程:
1. 運行 TSAR-RAPTOR AI 檢測
2. 識別 GRAY_ZONE 視頻（20-60% AI概率）
3. 加入人工標註佇列
4. 人工標註（可選）
5. 檢查是否達到重訓練閾值
6. 自動重訓練並部署改進模型

設計原則:
- 第一性原理: 物理特性 + 人類智慧雙重驗證
- 沙皇炸彈: 三階段級聯 + 97%物理純度
- 猛禽3: 簡約高效，持續迭代
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import time

# 添加項目根目錄到路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from autotesting_v3 import TSARRaptorDetector, DetectionResult
from core.human_annotator import HumanAnnotator
from core.continuous_trainer import ContinuousTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TSARRaptorIntegratedSystem:
    """TSAR-RAPTOR 集成系統"""

    def __init__(
        self,
        enable_human_annotation: bool = True,
        annotator_id: str = "default",
        auto_retrain: bool = True
    ):
        """
        Args:
            enable_human_annotation: 啟用人工標註
            annotator_id: 標註者ID
            auto_retrain: 自動重訓練
        """
        self.detector = TSARRaptorDetector()
        self.annotator = HumanAnnotator(annotator_id) if enable_human_annotation else None
        self.trainer = ContinuousTrainer() if auto_retrain else None
        self.enable_human_annotation = enable_human_annotation
        self.auto_retrain = auto_retrain

        logger.info("TSAR-RAPTOR 集成系統初始化完成")
        logger.info(f"  • 人工標註: {'啟用' if enable_human_annotation else '禁用'}")
        logger.info(f"  • 自動重訓練: {'啟用' if auto_retrain else '禁用'}")

    def process_videos(
        self,
        video_paths: List[str],
        annotate_gray_zone: bool = True
    ) -> Tuple[List[DetectionResult], int]:
        """
        處理視頻列表，完整集成流程

        Args:
            video_paths: 視頻路徑列表
            annotate_gray_zone: 是否標註灰色地帶視頻

        Returns:
            (檢測結果列表, 標註完成數量)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"TSAR-RAPTOR 集成系統開始處理 {len(video_paths)} 個視頻")
        logger.info(f"{'='*80}\n")

        # Phase 1: AI 檢測
        logger.info("Phase 1: TSAR-RAPTOR AI 檢測...")
        results = []
        gray_zone_videos = []

        for i, video_path in enumerate(video_paths, 1):
            logger.info(f"[{i}/{len(video_paths)}] 檢測: {os.path.basename(video_path)}")

            try:
                result = self.detector.detect(video_path)
                results.append(result)

                # 識別灰色地帶
                if result.needs_human_review():
                    gray_zone_videos.append((video_path, result))
                    logger.info(f"  ⚠️  GRAY_ZONE: AI_P={result.ai_probability:.1f}%, 需要人工復審")
                elif result.ai_probability >= 60:
                    logger.info(f"  🚫 BLOCK: AI_P={result.ai_probability:.1f}%")
                else:
                    logger.info(f"  ✅ PASS: AI_P={result.ai_probability:.1f}%")

            except Exception as e:
                logger.error(f"  ❌ 檢測失敗: {e}")
                continue

        # 統計 Phase 1 結果
        self._show_detection_summary(results, gray_zone_videos)

        # Phase 2: 人工標註（可選）
        annotations_completed = 0
        if self.enable_human_annotation and annotate_gray_zone and gray_zone_videos:
            logger.info(f"\nPhase 2: 人工標註灰色地帶視頻...")
            annotations_completed = self._annotate_gray_zone(gray_zone_videos)

        # Phase 3: 自動重訓練（可選）
        if self.auto_retrain and self.trainer:
            logger.info(f"\nPhase 3: 檢查重訓練條件...")
            self.trainer.check_and_retrain(force=False)

        logger.info(f"\n{'='*80}")
        logger.info(f"TSAR-RAPTOR 集成系統處理完成")
        logger.info(f"{'='*80}\n")

        return results, annotations_completed

    def _show_detection_summary(
        self,
        results: List[DetectionResult],
        gray_zone_videos: List[Tuple[str, DetectionResult]]
    ):
        """顯示檢測摘要"""
        total = len(results)
        gray_zone = len(gray_zone_videos)
        blocked = sum(1 for r in results if r.ai_probability >= 60)
        passed = sum(1 for r in results if r.ai_probability < 20)
        flagged = total - blocked - passed

        print(f"\n{'─'*80}")
        print(f"Phase 1 檢測摘要:")
        print(f"  • 總計: {total} 個視頻")
        print(f"  • 🚫 BLOCK (AI_P >= 60%): {blocked}")
        print(f"  • 🚩 FLAG (20% < AI_P < 60%): {flagged}")
        print(f"  • ✅ PASS (AI_P < 20%): {passed}")
        print(f"  • ⚠️  需人工復審: {gray_zone}")
        print(f"{'─'*80}\n")

    def _annotate_gray_zone(
        self,
        gray_zone_videos: List[Tuple[str, DetectionResult]]
    ) -> int:
        """標註灰色地帶視頻"""
        if not self.annotator:
            logger.warning("人工標註器未啟用")
            return 0

        print(f"\n{'='*80}")
        print(f"發現 {len(gray_zone_videos)} 個灰色地帶視頻需要人工復審")
        print(f"{'='*80}\n")

        # 詢問是否進行標註
        response = input(f"是否開始人工標註？(y/n，默認n): ").lower().strip()
        if response != 'y':
            logger.info("跳過人工標註")
            return 0

        # 準備標註數據
        annotation_data = []
        for video_path, result in gray_zone_videos:
            ai_result = {
                'ai_probability': result.ai_probability,
                'confidence': result.confidence,
                'top_reasons': result.top_reasons
            }
            annotation_data.append((video_path, ai_result))

        # 批量標註
        completed = self.annotator.batch_annotate(annotation_data)

        logger.info(f"✅ 完成 {completed} 個視頻的人工標註")
        return completed

    def show_system_status(self):
        """顯示系統狀態"""
        print(f"\n{'='*80}")
        print(f"{'TSAR-RAPTOR 集成系統狀態'.center(80)}")
        print(f"{'='*80}\n")

        # 人工標註統計
        if self.annotator:
            self.annotator.show_statistics()

        # 持續訓練狀態
        if self.trainer:
            self.trainer.show_training_status()

    def force_retrain(self):
        """強制重訓練"""
        if not self.trainer:
            logger.error("持續訓練器未啟用")
            return

        logger.info("強制重訓練...")
        new_model_path = self.trainer.check_and_retrain(force=True)

        if new_model_path:
            logger.info(f"✅ 訓練完成: {new_model_path}")
        else:
            logger.warning("訓練失敗或無可用數據")


def main():
    """主程式"""
    parser = argparse.ArgumentParser(
        description="TSAR-RAPTOR 集成系統 - AI檢測 + 人眼輔助學習"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='input',
        help='輸入目錄或視頻文件路徑'
    )
    parser.add_argument(
        '--no-annotation',
        action='store_true',
        help='禁用人工標註'
    )
    parser.add_argument(
        '--no-retrain',
        action='store_true',
        help='禁用自動重訓練'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='只顯示系統狀態'
    )
    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='強制重訓練模型'
    )
    parser.add_argument(
        '--annotator-id',
        type=str,
        default='default',
        help='標註者ID'
    )

    args = parser.parse_args()

    # 創建集成系統
    system = TSARRaptorIntegratedSystem(
        enable_human_annotation=not args.no_annotation,
        annotator_id=args.annotator_id,
        auto_retrain=not args.no_retrain
    )

    # 只顯示狀態
    if args.status:
        system.show_system_status()
        return

    # 強制重訓練
    if args.force_retrain:
        system.force_retrain()
        return

    # 獲取視頻列表
    input_path = Path(args.input)
    if input_path.is_file():
        video_paths = [str(input_path)]
    elif input_path.is_dir():
        video_paths = [
            str(p) for p in input_path.glob('*.mp4')
        ]
    else:
        logger.error(f"無效的輸入路徑: {input_path}")
        return

    if not video_paths:
        logger.error(f"未找到視頻文件: {input_path}")
        return

    # 處理視頻
    start_time = time.time()
    results, annotations = system.process_videos(
        video_paths,
        annotate_gray_zone=not args.no_annotation
    )
    elapsed = time.time() - start_time

    # 最終統計
    print(f"\n{'='*80}")
    print(f"TSAR-RAPTOR 集成系統執行統計:")
    print(f"  • 處理視頻: {len(video_paths)}")
    print(f"  • 完成標註: {annotations}")
    print(f"  • 執行時間: {elapsed:.2f} 秒")
    print(f"  • 平均速度: {elapsed/len(video_paths):.2f} 秒/視頻")
    print(f"{'='*80}\n")

    # 顯示系統狀態
    system.show_system_status()


if __name__ == "__main__":
    main()

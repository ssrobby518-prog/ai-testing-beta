#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
TSAR-RAPTOR File Auto-Classifier
自動將視頻分類移動到對應文件夾

設計原則:
- 第一性原理: 原子性操作，數據不丟失
- 沙皇炸彈: 批量移動，極速完成
- 猛禽3: 簡約接口，自動創建目錄

文件夾結構:
C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok videos download\
├── real\           ← 真實視頻
├── ai\             ← AI生成視頻
├── not sure\       ← 不確定視頻
└── 電影動畫\        ← 電影/動畫視頻
"""

import shutil
from pathlib import Path
import logging
from typing import List, Dict
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FileAutoClassifier:
    """視頻文件自動分類器"""

    def __init__(
        self,
        source_dir: str,
        base_output_dir: str = None
    ):
        """
        Args:
            source_dir: 源視頻目錄
            base_output_dir: 分類輸出根目錄（None則使用source_dir的父目錄）
        """
        self.source_dir = Path(source_dir).resolve()

        if base_output_dir:
            resolved_base = Path(base_output_dir).resolve()
            allowed = {self.source_dir, self.source_dir.parent}
            if resolved_base not in allowed:
                raise ValueError(f"base_output_dir must be source_dir or its parent: source_dir={self.source_dir}, base_output_dir={resolved_base}")
            self.base_output_dir = resolved_base
        else:
            self.base_output_dir = self.source_dir.parent

        # 創建分類文件夾
        self.folders = {
            'REAL': self.base_output_dir / 'real',
            'AI': self.base_output_dir / 'ai',
            'NOT_SURE': self.base_output_dir / 'not sure',
            '電影動畫': self.base_output_dir / '電影動畫'
        }

        for folder in self.folders.values():
            folder.mkdir(parents=True, exist_ok=True)

        logger.info("文件自動分類器初始化完成")
        logger.info(f"  • 源目錄: {self.source_dir}")
        logger.info(f"  • 輸出根目錄: {self.base_output_dir}")
        logger.info(f"  • 分類文件夾:")
        for label, folder in self.folders.items():
            logger.info(f"    - {label}: {folder}")

    def classify_from_detection_results(
        self,
        detection_results: List[Dict],
        move_files: bool = True,
        rename_pattern: str = "{label}_{video_id}.mp4"
    ) -> Dict:
        """
        根據檢測結果分類文件

        Args:
            detection_results: AI檢測結果列表
            move_files: 是否真正移動文件（False則僅統計）
            rename_pattern: 重命名模式

        Returns:
            統計結果
        """
        logger.info(f"🚀 開始分類文件: {len(detection_results)} 個視頻")

        stats = {
            'total': len(detection_results),
            'moved': 0,
            'failed': 0,
            'by_category': {
                'REAL': 0,
                'AI': 0,
                'NOT_SURE': 0,
                '電影動畫': 0
            }
        }

        for result in detection_results:
            try:
                source_path = Path(result['video_path'])
                classification = result['classification']
                video_id = self._extract_video_id(source_path)

                # 目標文件夾
                target_folder = self.folders.get(classification)
                if not target_folder:
                    logger.warning(f"⚠️  未知分類: {classification}")
                    stats['failed'] += 1
                    continue

                # 生成新文件名
                label_lower = classification.lower().replace(' ', '_')
                new_filename = rename_pattern.format(
                    label=label_lower,
                    video_id=video_id
                )
                target_path = target_folder / new_filename

                # 移動文件
                if move_files:
                    if source_path.exists():
                        shutil.move(str(source_path), str(target_path))
                        logger.info(f"✅ [{classification}] {source_path.name} → {target_path.name}")
                        stats['moved'] += 1
                        stats['by_category'][classification] += 1
                    else:
                        logger.warning(f"⚠️  源文件不存在: {source_path}")
                        stats['failed'] += 1
                else:
                    # 僅統計
                    logger.info(f"📊 [模擬] {classification}: {source_path.name} → {target_path.name}")
                    stats['by_category'][classification] += 1

            except Exception as e:
                logger.error(f"❌ 移動失敗 [{source_path.name}]: {e}")
                stats['failed'] += 1

        # 顯示統計
        logger.info(f"\n{'='*80}")
        logger.info(f"分類完成:")
        logger.info(f"  • 總計: {stats['total']}")
        logger.info(f"  • 已移動: {stats['moved']}")
        logger.info(f"  • 失敗: {stats['failed']}")
        logger.info(f"  • 分類統計:")
        for label, count in stats['by_category'].items():
            logger.info(f"    - {label}: {count}")
        logger.info(f"{'='*80}\n")

        return stats

    def classify_from_excel_d(
        self,
        excel_d_path: str,
        move_files: bool = True
    ) -> Dict:
        """
        從 Excel D 讀取分類結果並移動文件

        Args:
            excel_d_path: Excel D 路徑
            move_files: 是否真正移動文件

        Returns:
            統計結果
        """
        excel_d_path = Path(excel_d_path)
        if not excel_d_path.exists():
            logger.error(f"❌ Excel D 不存在: {excel_d_path}")
            return {}

        # 讀取 Excel D
        df = pd.read_excel(excel_d_path)
        logger.info(f"✅ 已加載 Excel D: {len(df)} 行")

        # 轉換為檢測結果格式
        detection_results = []
        for _, row in df.iterrows():
            # 優先使用人工復審結果
            classification = row.get('人工復審結果', '')
            if not classification or pd.isna(classification):
                classification = row.get('AI檢測分類', 'NOT_SURE')

            # 構建完整路徑
            file_path = row.get('檔案路徑', '')
            if file_path and not Path(file_path).is_absolute():
                # 相對路徑 → 絕對路徑
                file_path = self.source_dir.parent / file_path

            detection_results.append({
                'video_path': str(file_path),
                'classification': classification
            })

        # 執行分類
        return self.classify_from_detection_results(detection_results, move_files)

    def move_single_video(
        self,
        video_path: str,
        classification: str,
        rename: bool = True
    ) -> bool:
        """
        移動單個視頻

        Args:
            video_path: 視頻路徑
            classification: 分類 (REAL/AI/NOT_SURE/電影動畫)
            rename: 是否重命名

        Returns:
            是否成功
        """
        try:
            source_path = Path(video_path)
            if not source_path.exists():
                logger.error(f"❌ 文件不存在: {source_path}")
                return False

            # 目標文件夾
            target_folder = self.folders.get(classification)
            if not target_folder:
                logger.error(f"❌ 未知分類: {classification}")
                return False

            # 生成目標路徑
            if rename:
                video_id = self._extract_video_id(source_path)
                label_lower = classification.lower().replace(' ', '_')
                new_filename = f"{label_lower}_{video_id}.mp4"
                target_path = target_folder / new_filename
            else:
                target_path = target_folder / source_path.name

            # 移動文件
            shutil.move(str(source_path), str(target_path))
            logger.info(f"✅ [{classification}] {source_path.name} → {target_path.name}")

            return True

        except Exception as e:
            logger.error(f"❌ 移動失敗: {e}")
            return False

    def get_folder_statistics(self) -> Dict:
        """
        獲取各文件夾統計信息

        Returns:
            統計字典
        """
        stats = {}

        for label, folder in self.folders.items():
            video_count = len(list(folder.glob("*.mp4")))
            stats[label] = {
                'path': str(folder),
                'count': video_count
            }

        return stats

    def _extract_video_id(self, video_path: Path) -> str:
        """
        從文件名提取視頻ID

        Args:
            video_path: 視頻路徑

        Returns:
            視頻ID
        """
        import re
        match = re.search(r'(\d+)', video_path.stem)
        if match:
            return match.group(1)
        return video_path.stem


def main():
    """測試文件分類器"""
    import argparse

    parser = argparse.ArgumentParser(description="視頻文件自動分類器")
    parser.add_argument(
        '--source-dir',
        type=str,
        required=True,
        help='源視頻目錄'
    )
    parser.add_argument(
        '--excel-d',
        type=str,
        help='Excel D 路徑'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='僅統計，不實際移動文件'
    )

    args = parser.parse_args()

    # 創建分類器
    classifier = FileAutoClassifier(source_dir=args.source_dir)

    # 顯示文件夾統計
    folder_stats = classifier.get_folder_statistics()
    print(f"\n{'='*80}")
    print("當前文件夾統計:")
    for label, stats in folder_stats.items():
        print(f"  • {label}: {stats['count']} 個視頻")
    print(f"{'='*80}\n")

    # 執行分類
    if args.excel_d:
        move_files = not args.dry_run
        stats = classifier.classify_from_excel_d(args.excel_d, move_files)

        if args.dry_run:
            print("\n⚠️  模擬模式（--dry-run）：未實際移動文件")
        else:
            print(f"\n✅ 分類完成！已移動 {stats['moved']} 個視頻")


if __name__ == "__main__":
    main()

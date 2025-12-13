#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSAR-RAPTOR Feature Extractor (Layer 1)
從分類視頻文件夾提取特徵 → 生成 Excel B

設計原則:
- 第一性原理: 物理特徵不可偽造
- 沙皇炸彈: 批量並行提取
- 猛禽3: 簡約接口

功能:
1. 從 real/ai/not sure/movies 文件夾加載視頻
2. 提取15+特徵
3. 生成 Excel B
"""

import sys
from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from typing import Dict, List
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# 導入配置
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR.parent))
from config import LAYER1_VIDEO_FOLDERS, EXCEL_B_PATH

# 導入原有特徵提取器
from analyzer.feature_extractor import FeatureExtractor as BaseFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureExtractorLayer1:
    """Layer 1 特徵提取器（從分類文件夾）"""

    def __init__(
        self,
        output_excel_b: str = None,
        max_workers: int = 4,
        sample_frames: int = 30
    ):
        """
        Args:
            output_excel_b: Excel B 輸出路徑
            max_workers: 並行處理數
            sample_frames: 採樣幀數
        """
        self.video_folders = LAYER1_VIDEO_FOLDERS
        self.output_excel_b = Path(output_excel_b) if output_excel_b else EXCEL_B_PATH
        self.output_excel_b.parent.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.sample_frames = sample_frames

        # 使用基礎特徵提取器
        self.base_extractor = BaseFeatureExtractor(
            video_dir="",
            output_excel_b=str(self.output_excel_b),
            max_workers=max_workers,
            sample_frames=sample_frames
        )

        logger.info("Layer 1 特徵提取器初始化完成")
        logger.info(f"  • Excel B: {self.output_excel_b}")
        logger.info(f"  • 並行數: {self.max_workers}")
        logger.info(f"  • 採樣幀數: {self.sample_frames}")

    def get_all_videos(self) -> List[tuple]:
        """
        從所有分類文件夾獲取視頻

        Returns:
            (視頻路徑, 標籤) 列表
        """
        videos = []

        for label, folder in self.video_folders.items():
            if folder.exists():
                video_files = list(folder.glob("*.mp4"))
                for vf in video_files:
                    videos.append((vf, label))

        logger.info(f"✅ 找到 {len(videos)} 個視頻")
        logger.info(f"   分類統計:")
        for label in ['real', 'ai', 'uncertain', 'exclude']:
            count = sum(1 for v in videos if v[1] == label)
            logger.info(f"     - {label}: {count}")

        return videos

    def extract_single_video_with_label(self, video_info: tuple) -> Dict:
        """
        提取單個視頻特徵（帶標籤）

        Args:
            video_info: (視頻路徑, 標籤)

        Returns:
            特徵字典
        """
        video_path, label = video_info

        try:
            # 使用基礎提取器提取特徵
            features = self.base_extractor.extract_single_video(video_path)

            # 添加標籤信息
            features['label'] = label
            features['label_cn'] = self._translate_label(label)

            return features

        except Exception as e:
            logger.error(f"❌ 提取失敗 [{video_path.name}]: {e}")
            return {
                'filename': video_path.name,
                'label': label,
                'label_cn': self._translate_label(label),
                'error': str(e)
            }

    def _translate_label(self, label: str) -> str:
        """翻譯標籤為中文"""
        translations = {
            'real': '真實',
            'ai': 'AI',
            'uncertain': '不確定',
            'exclude': '電影/動畫'
        }
        return translations.get(label, label)

    def batch_extract(self) -> pd.DataFrame:
        """
        批量提取所有視頻特徵

        Returns:
            Excel B DataFrame
        """
        # 獲取所有視頻
        videos = self.get_all_videos()

        if not videos:
            logger.warning("⚠️  沒有找到視頻")
            return pd.DataFrame()

        logger.info(f"🚀 開始批量提取: {len(videos)} 個視頻（並行數: {self.max_workers}）")

        features_list = []

        # 並行提取
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.extract_single_video_with_label, v): v for v in videos}

            for i, future in enumerate(as_completed(futures), 1):
                features = future.result()
                if features:
                    features_list.append(features)

                # 進度顯示
                if i % 10 == 0 or i == len(videos):
                    logger.info(f"📊 進度: {i}/{len(videos)} ({i/len(videos)*100:.1f}%)")

        # 創建 DataFrame
        df = pd.DataFrame(features_list)

        # 保存到 Excel B
        df.to_excel(self.output_excel_b, index=False)
        logger.info(f"✅ Excel B 已保存: {self.output_excel_b}")

        # 統計
        logger.info(f"\n{'='*80}")
        logger.info(f"特徵提取完成:")
        logger.info(f"  • 總計: {len(df)} 個視頻")
        logger.info(f"  • 特徵數: {len(df.columns)} 列")
        logger.info(f"{'='*80}\n")

        return df


def main():
    """主程式"""
    import argparse

    parser = argparse.ArgumentParser(description="Layer 1 特徵提取器")
    parser.add_argument(
        '--output',
        type=str,
        help='Excel B 輸出路徑（默認使用配置文件）'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='並行處理數'
    )
    parser.add_argument(
        '--sample-frames',
        type=int,
        default=30,
        help='採樣幀數'
    )

    args = parser.parse_args()

    # 創建提取器
    extractor = FeatureExtractorLayer1(
        output_excel_b=args.output,
        max_workers=args.workers,
        sample_frames=args.sample_frames
    )

    # 批量提取
    df = extractor.batch_extract()

    print(f"\n✅ 特徵提取完成！")
    print(f"   共 {len(df)} 個視頻")
    print(f"   Excel B: {extractor.output_excel_b}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSAR-RAPTOR Excel D Generator
生成Excel D（AI檢測結果 + 特徵記錄）

設計原則:
- 第一性原理: 完整特徵記錄供自我訓練
- 沙皇炸彈: 海量數據累積
- 猛禽3: 簡約格式，易於分析

Excel D 包含:
1. 基本信息: 序號, 影片網址, AI檢測分類, 信心度, 視頻ID, 檔案路徑, 分析時間
2. 關鍵特徵: 15+個視覺/運動/頻域特徵
3. 復審信息: 人工復審結果, 復審時間, 備註
"""

import sys
from pathlib import Path
import pandas as pd
import logging
from typing import List, Dict
from datetime import datetime
import cv2
import numpy as np

# 添加項目路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analyzer.feature_extractor import FeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExcelDGenerator:
    """Excel D 生成器"""

    def __init__(
        self,
        video_dir: str,
        output_excel_d: str,
        url_mapping: Dict[str, str] = None
    ):
        """
        Args:
            video_dir: 視頻目錄
            output_excel_d: Excel D 輸出路徑
            url_mapping: 視頻ID → URL 映射字典
        """
        self.video_dir = Path(video_dir)
        self.output_excel_d = Path(output_excel_d)
        self.output_excel_d.parent.mkdir(parents=True, exist_ok=True)
        self.url_mapping = url_mapping or {}

        # 特徵提取器
        self.feature_extractor = FeatureExtractor(
            video_dir=str(self.video_dir),
            output_excel_b="temp_features.xlsx",
            max_workers=4,
            sample_frames=30
        )

        logger.info("Excel D 生成器初始化完成")
        logger.info(f"  • 視頻目錄: {self.video_dir}")
        logger.info(f"  • 輸出路徑: {self.output_excel_d}")

    def generate_from_detection_results(
        self,
        detection_results: List[Dict]
    ) -> pd.DataFrame:
        """
        從檢測結果生成 Excel D

        Args:
            detection_results: AI檢測結果列表

        Returns:
            Excel D DataFrame
        """
        logger.info(f"🔬 開始生成 Excel D: {len(detection_results)} 個視頻")

        rows = []

        for i, result in enumerate(detection_results, 1):
            try:
                # 基本信息
                video_path = Path(result['video_path'])
                video_id = self._extract_video_id(video_path)
                url = self.url_mapping.get(video_id, f"https://www.tiktok.com/video/{video_id}")

                # AI檢測結果
                classification = result['classification']
                confidence = result.get('confidence', 0.0)
                ai_score = result.get('ai_score', 50.0)

                # 提取特徵
                features = self._extract_features_from_video(video_path)

                # 組合數據行
                row = {
                    # 基本信息
                    '序號': i,
                    '影片網址': url,
                    'AI檢測分類': classification,
                    '信心度': round(confidence, 2),
                    '視頻ID': video_id,
                    '檔案路徑': str(video_path.relative_to(video_path.parent.parent)),
                    '分析時間': datetime.now().isoformat(),

                    # 關鍵特徵（從 features 提取）
                    **features,

                    # 復審信息（初始為空）
                    '人工復審結果': '',
                    '復審時間': '',
                    '備註': ''
                }

                rows.append(row)

                # 進度顯示
                if i % 10 == 0 or i == len(detection_results):
                    logger.info(f"📊 進度: {i}/{len(detection_results)} ({i/len(detection_results)*100:.1f}%)")

            except Exception as e:
                logger.error(f"❌ 處理失敗 [{video_path.name}]: {e}")
                continue

        # 創建 DataFrame
        df = pd.DataFrame(rows)

        # 保存到 Excel
        df.to_excel(self.output_excel_d, index=False)
        logger.info(f"✅ Excel D 已保存: {self.output_excel_d}")

        return df

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

    def _extract_features_from_video(self, video_path: Path) -> Dict:
        """
        從視頻提取特徵

        Args:
            video_path: 視頻路徑

        Returns:
            特徵字典
        """
        try:
            # 使用現有特徵提取器
            features = self.feature_extractor.extract_single_video(video_path)

            # 提取關鍵特徵
            return {
                'fps': features.get('fps', 0),
                'width': features.get('width', 0),
                'height': features.get('height', 0),
                'duration': features.get('duration', 0),
                'avg_brightness': round(features.get('avg_brightness', 0), 2),
                'avg_contrast': round(features.get('avg_contrast', 0), 2),
                'avg_saturation': round(features.get('avg_saturation', 0), 2),
                'avg_blur': round(features.get('avg_blur', 0), 2),
                'avg_optical_flow': round(features.get('avg_optical_flow', 0), 2),
                'scene_changes': features.get('scene_changes', 0),
                'dct_energy': round(features.get('dct_energy', 0), 2),
                'spectral_entropy': round(features.get('spectral_entropy', 0), 2),
                'audio_sample_rate': features.get('audio_sample_rate', 0),
                'audio_channels': features.get('audio_channels', 0),
                'bitrate': features.get('bitrate', 0)
            }

        except Exception as e:
            logger.warning(f"⚠️  特徵提取失敗 [{video_path.name}]: {e}")
            # 返回默認值
            return {
                'fps': 0, 'width': 0, 'height': 0, 'duration': 0,
                'avg_brightness': 0, 'avg_contrast': 0, 'avg_saturation': 0,
                'avg_blur': 0, 'avg_optical_flow': 0, 'scene_changes': 0,
                'dct_energy': 0, 'spectral_entropy': 0,
                'audio_sample_rate': 0, 'audio_channels': 0, 'bitrate': 0
            }

    def update_review_results(
        self,
        video_id: str,
        human_label: str,
        notes: str = ""
    ) -> bool:
        """
        更新人工復審結果

        Args:
            video_id: 視頻ID
            human_label: 人工標籤 (REAL/AI/電影動畫)
            notes: 備註

        Returns:
            是否成功
        """
        try:
            # 讀取現有 Excel D
            df = pd.read_excel(self.output_excel_d)

            # 查找對應行
            mask = df['視頻ID'] == video_id

            if mask.sum() == 0:
                logger.warning(f"⚠️  找不到視頻: {video_id}")
                return False

            # 更新復審信息
            df.loc[mask, '人工復審結果'] = human_label.upper()
            df.loc[mask, '復審時間'] = datetime.now().isoformat()
            df.loc[mask, '備註'] = notes

            # 保存
            df.to_excel(self.output_excel_d, index=False)
            logger.info(f"✅ 已更新復審結果: {video_id} → {human_label}")

            return True

        except Exception as e:
            logger.error(f"❌ 更新失敗: {e}")
            return False

    def get_statistics(self) -> Dict:
        """
        獲取 Excel D 統計信息

        Returns:
            統計字典
        """
        try:
            df = pd.read_excel(self.output_excel_d)

            stats = {
                'total': len(df),
                'real': len(df[df['AI檢測分類'] == 'REAL']),
                'ai': len(df[df['AI檢測分類'] == 'AI']),
                'not_sure': len(df[df['AI檢測分類'] == 'NOT_SURE']),
                'movie': len(df[df['AI檢測分類'] == '電影動畫']),
                'reviewed': len(df[df['人工復審結果'] != ''])
            }

            # 計算百分比
            if stats['total'] > 0:
                stats['real_pct'] = stats['real'] / stats['total'] * 100
                stats['ai_pct'] = stats['ai'] / stats['total'] * 100
                stats['not_sure_pct'] = stats['not_sure'] / stats['total'] * 100
                stats['movie_pct'] = stats['movie'] / stats['total'] * 100
                stats['reviewed_pct'] = stats['reviewed'] / stats['total'] * 100

            return stats

        except Exception as e:
            logger.error(f"❌ 獲取統計失敗: {e}")
            return {}


def main():
    """測試 Excel D 生成器"""
    import argparse

    parser = argparse.ArgumentParser(description="Excel D 生成器")
    parser.add_argument(
        '--video-dir',
        type=str,
        required=True,
        help='視頻目錄'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../../tiktok videos download/data/excel_d_detection_results.xlsx',
        help='Excel D 輸出路徑'
    )

    args = parser.parse_args()

    # 創建生成器
    generator = ExcelDGenerator(
        video_dir=args.video_dir,
        output_excel_d=args.output
    )

    # 模擬檢測結果（實際使用時從 ai_detector.py 獲取）
    detection_results = [
        {
            'video_path': str(Path(args.video_dir) / 'test_video.mp4'),
            'classification': 'REAL',
            'confidence': 85.5,
            'ai_score': 25.3
        }
    ]

    # 生成 Excel D
    df = generator.generate_from_detection_results(detection_results)

    print(f"\n✅ Excel D 已生成: {args.output}")
    print(f"   共 {len(df)} 行數據")


if __name__ == "__main__":
    main()

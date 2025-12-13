#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSAR-RAPTOR Feature Extractor - Excel B生成器
對下載的視頻進行特徵提取分析

設計原則:
- 第一性原理: 提取物理可測量特徵
- 沙皇炸彈: 多維度特徵，級聯分析
- 猛禽3: 高效並行，輕量級提取

Excel B 特徵列表:
1. 基本信息: video_id, label, filepath, file_size
2. 視頻特徵: 幀率, 分辨率, 時長, 碼率, 總幀數
3. 音頻特徵: 採樣率, 聲道數, 音頻碼率
4. 視覺特徵: 平均亮度, 對比度, 色彩飽和度, 模糊度
5. 運動特徵: 光流平均值, 場景變化次數
6. 頻域特徵: DCT能量, 頻譜熵
7. 參考模組: 調用12個檢測模組的快速版本
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """視頻特徵提取器"""

    def __init__(
        self,
        video_dir: str,
        output_excel_b: str = "excel_b_features.xlsx",
        max_workers: int = 4,
        sample_frames: int = 30  # 採樣幀數（輕量級）
    ):
        """
        Args:
            video_dir: 視頻目錄
            output_excel_b: Excel B 輸出路徑
            max_workers: 並行處理數
            sample_frames: 採樣幀數
        """
        self.video_dir = Path(video_dir)
        self.output_excel_b = Path(output_excel_b)
        self.max_workers = max_workers
        self.sample_frames = sample_frames

        logger.info("特徵提取器初始化完成")
        logger.info(f"  • 視頻目錄: {self.video_dir}")
        logger.info(f"  • 輸出 Excel B: {self.output_excel_b}")
        logger.info(f"  • 採樣幀數: {self.sample_frames}")

    def extract_metadata(self, video_path: Path) -> Dict:
        """
        使用 ffprobe 提取元數據

        Returns:
            {
                'duration': float,
                'fps': float,
                'width': int,
                'height': int,
                'bitrate': int,
                'audio_sample_rate': int,
                'audio_channels': int,
                ...
            }
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(video_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            data = json.loads(result.stdout)

            # 提取視頻流信息
            video_stream = next((s for s in data['streams'] if s['codec_type'] == 'video'), {})
            audio_stream = next((s for s in data['streams'] if s['codec_type'] == 'audio'), {})

            # 解析幀率
            fps_str = video_stream.get('r_frame_rate', '0/1')
            fps_parts = fps_str.split('/')
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 and fps_parts[1] != '0' else 0

            return {
                'duration': float(data.get('format', {}).get('duration', 0)),
                'fps': fps,
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'bitrate': int(data.get('format', {}).get('bit_rate', 0)),
                'total_frames': int(video_stream.get('nb_frames', 0)),
                'audio_sample_rate': int(audio_stream.get('sample_rate', 0)),
                'audio_channels': int(audio_stream.get('channels', 0)),
                'audio_bitrate': int(audio_stream.get('bit_rate', 0)),
                'codec': video_stream.get('codec_name', 'unknown')
            }
        except Exception as e:
            logger.error(f"❌ 提取元數據失敗: {video_path.name} | {e}")
            return {}

    def extract_visual_features(self, video_path: Path) -> Dict:
        """
        提取視覺特徵（採樣方式）

        Returns:
            {
                'avg_brightness': float,
                'avg_contrast': float,
                'avg_saturation': float,
                'avg_blur': float,
                'avg_optical_flow': float,
                'scene_changes': int
            }
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 計算採樣間隔
            if total_frames == 0:
                return {}

            step = max(total_frames // self.sample_frames, 1)

            brightness_list = []
            contrast_list = []
            saturation_list = []
            blur_list = []
            optical_flow_list = []
            prev_gray = None
            scene_changes = 0

            for i in range(0, total_frames, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break

                # 轉換為灰度和HSV
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # 亮度（V通道平均值）
                brightness = np.mean(hsv[:, :, 2])
                brightness_list.append(brightness)

                # 對比度（標準差）
                contrast = np.std(gray)
                contrast_list.append(contrast)

                # 飽和度（S通道平均值）
                saturation = np.mean(hsv[:, :, 1])
                saturation_list.append(saturation)

                # 模糊度（Laplacian方差，越小越模糊）
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                blur = laplacian.var()
                blur_list.append(blur)

                # 光流（運動）
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    avg_flow = np.mean(mag)
                    optical_flow_list.append(avg_flow)

                    # 場景變化（幀差異大於閾值）
                    frame_diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float)))
                    if frame_diff > 30:  # 閾值
                        scene_changes += 1

                prev_gray = gray

            cap.release()

            return {
                'avg_brightness': np.mean(brightness_list) if brightness_list else 0,
                'avg_contrast': np.mean(contrast_list) if contrast_list else 0,
                'avg_saturation': np.mean(saturation_list) if saturation_list else 0,
                'avg_blur': np.mean(blur_list) if blur_list else 0,
                'avg_optical_flow': np.mean(optical_flow_list) if optical_flow_list else 0,
                'scene_changes': scene_changes
            }
        except Exception as e:
            logger.error(f"❌ 提取視覺特徵失敗: {video_path.name} | {e}")
            return {}

    def extract_frequency_features(self, video_path: Path) -> Dict:
        """
        提取頻域特徵（DCT）

        Returns:
            {
                'dct_energy': float,
                'spectral_entropy': float
            }
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(total_frames // 10, 1)  # 採樣10幀

            dct_energies = []
            spectral_entropies = []

            for i in range(0, total_frames, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # DCT變換
                dct = cv2.dct(np.float32(gray) / 255.0)

                # DCT能量（高頻分量）
                dct_high_freq = dct[dct.shape[0]//2:, dct.shape[1]//2:]
                dct_energy = np.sum(dct_high_freq ** 2)
                dct_energies.append(dct_energy)

                # 頻譜熵
                dct_abs = np.abs(dct.flatten())
                dct_abs = dct_abs / (np.sum(dct_abs) + 1e-10)  # 歸一化
                spectral_entropy = -np.sum(dct_abs * np.log2(dct_abs + 1e-10))
                spectral_entropies.append(spectral_entropy)

            cap.release()

            return {
                'dct_energy': np.mean(dct_energies) if dct_energies else 0,
                'spectral_entropy': np.mean(spectral_entropies) if spectral_entropies else 0
            }
        except Exception as e:
            logger.error(f"❌ 提取頻域特徵失敗: {video_path.name} | {e}")
            return {}

    def extract_single_video(self, video_path: Path) -> Dict:
        """
        提取單個視頻的完整特徵

        Returns:
            特徵字典
        """
        logger.info(f"🔬 分析中: {video_path.name}")

        # 從文件名提取 label 和 video_id
        stem = video_path.stem  # e.g., "real_123" or "ai_456"
        parts = stem.split('_')
        label = parts[0] if len(parts) >= 2 else 'unknown'
        video_id = parts[1] if len(parts) >= 2 else 'unknown'

        # 基本信息
        features = {
            'video_id': video_id,
            'label': label,
            'filepath': str(video_path),
            'filename': video_path.name,
            'file_size_mb': video_path.stat().st_size / (1024 * 1024)
        }

        # 元數據
        metadata = self.extract_metadata(video_path)
        features.update(metadata)

        # 視覺特徵
        visual = self.extract_visual_features(video_path)
        features.update(visual)

        # 頻域特徵
        frequency = self.extract_frequency_features(video_path)
        features.update(frequency)

        logger.info(f"✅ 分析完成: {video_path.name}")
        return features

    def batch_extract(self) -> pd.DataFrame:
        """
        批量提取所有視頻特徵

        Returns:
            DataFrame
        """
        video_files = list(self.video_dir.glob("*.mp4"))
        logger.info(f"🚀 開始批量提取: {len(video_files)} 個視頻")

        features_list = []

        # 並行提取
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.extract_single_video, vf): vf for vf in video_files}

            for future in as_completed(futures):
                try:
                    features = future.result()
                    features_list.append(features)
                except Exception as e:
                    video_file = futures[future]
                    logger.error(f"❌ 提取失敗: {video_file.name} | {e}")

        # 轉換為DataFrame
        df = pd.DataFrame(features_list)

        # 保存到Excel B
        df.to_excel(self.output_excel_b, index=False)
        logger.info(f"\n✅ Excel B 已生成: {self.output_excel_b}")
        logger.info(f"   總計: {len(df)} 個視頻")

        return df


def main():
    """測試特徵提取器"""
    import argparse

    parser = argparse.ArgumentParser(description="TikTok視頻特徵提取器")
    parser.add_argument(
        '--input',
        type=str,
        default='../../data/tiktok_videos',
        help='視頻目錄'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../../data/tiktok_labels/excel_b_features.xlsx',
        help='輸出 Excel B 路徑'
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
    extractor = FeatureExtractor(
        video_dir=args.input,
        output_excel_b=args.output,
        max_workers=args.workers,
        sample_frames=args.sample_frames
    )

    # 執行提取
    df = extractor.batch_extract()

    # 顯示統計
    print(f"\n{'='*80}")
    print(f"特徵提取完成！")
    print(f"  • 視頻總數: {len(df)}")
    print(f"  • Real: {len(df[df['label'] == 'real'])}")
    print(f"  • AI: {len(df[df['label'] == 'ai'])}")
    print(f"  • Uncertain: {len(df[df['label'] == 'uncertain'])}")
    print(f"  • Excel B: {args.output}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSAR-RAPTOR AI Detection Classifier
整合現有AI檢測系統，自動分類TikTok視頻

設計原則:
- 第一性原理: 物理不可偽造特徵（FR-TSAR三階段）
- 沙皇炸彈: 批量並行檢測
- 猛禽3: 簡約接口，無縫整合

整合檢測模組:
1. Stage 1 - 物理剛性 (40%): PVD + 骨骼守恆
2. Stage 2 - 頻率結構 (30%): 頻域分析 + CNN分類
3. Stage 3 - 邏輯決策 (30%): XGBoost集成

分類邏輯:
- REAL: AI_P < 30
- AI: AI_P >= 70
- NOT_SURE: 30 <= AI_P < 70
- 電影動畫: 特殊規則檢測
"""

import sys
from pathlib import Path
import logging
from typing import Dict, List
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加項目路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AIDetectionClassifier:
    """AI檢測分類器（整合現有系統）"""

    def __init__(
        self,
        video_dir: str,
        max_workers: int = 4
    ):
        """
        Args:
            video_dir: 視頻目錄
            max_workers: 並行檢測數
        """
        self.video_dir = Path(video_dir)
        self.max_workers = max_workers

        # 嘗試導入現有檢測模組
        self._load_detection_modules()

        logger.info("AI檢測分類器初始化完成")
        logger.info(f"  • 視頻目錄: {self.video_dir}")
        logger.info(f"  • 並行數: {self.max_workers}")

    def _load_detection_modules(self):
        """加載現有檢測模組"""
        try:
            # 嘗試導入現有檢測模組
            from modules.physics_violation_detector_v2 import PhysicsViolationDetector
            from modules.frequency_analyzer_v2 import FrequencyAnalyzerV2
            from modules.facial_rigidity_analyzer import FacialRigidityAnalyzer

            self.pvd = PhysicsViolationDetector()
            self.freq_analyzer = FrequencyAnalyzerV2()
            self.facial_analyzer = FacialRigidityAnalyzer()

            self.modules_loaded = True
            logger.info("✅ 檢測模組加載成功")

        except ImportError as e:
            logger.warning(f"⚠️  檢測模組未加載: {e}")
            logger.warning("   將使用簡化檢測邏輯")
            self.modules_loaded = False

    def detect_single_video(self, video_path: Path) -> Dict:
        """
        檢測單個視頻

        Args:
            video_path: 視頻路徑

        Returns:
            檢測結果字典
        """
        try:
            if self.modules_loaded:
                # 使用完整檢測系統
                return self._detect_with_full_system(video_path)
            else:
                # 使用簡化檢測
                return self._detect_with_simplified_logic(video_path)

        except Exception as e:
            logger.error(f"❌ 檢測失敗 [{video_path.name}]: {e}")
            return {
                'video_path': str(video_path),
                'classification': 'NOT_SURE',
                'confidence': 0.0,
                'ai_score': 50.0,
                'error': str(e)
            }

    def _detect_with_full_system(self, video_path: Path) -> Dict:
        """
        使用完整檢測系統（整合現有模組）

        Args:
            video_path: 視頻路徑

        Returns:
            檢測結果
        """
        # 打開視頻
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"無法打開視頻: {video_path}")

        # Stage 1: 物理違規檢測 (40%)
        pvd_score = self.pvd.analyze(cap)
        pvd_contribution = pvd_score * 0.4

        # Stage 2: 頻域分析 (30%)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置
        freq_score = self.freq_analyzer.analyze(cap)
        freq_contribution = freq_score * 0.3

        # Stage 3: 面部剛性 (30%)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置
        facial_score = self.facial_analyzer.analyze(cap)
        facial_contribution = facial_score * 0.3

        cap.release()

        # 總分計算
        ai_score = pvd_contribution + freq_contribution + facial_contribution

        # 分類邏輯
        classification, confidence = self._classify_by_score(ai_score)

        return {
            'video_path': str(video_path),
            'classification': classification,
            'confidence': confidence,
            'ai_score': ai_score,
            'pvd_score': pvd_score,
            'freq_score': freq_score,
            'facial_score': facial_score
        }

    def _detect_with_simplified_logic(self, video_path: Path) -> Dict:
        """
        簡化檢測邏輯（模組未加載時使用）

        基於基本視覺特徵的快速檢測

        Args:
            video_path: 視頻路徑

        Returns:
            檢測結果
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"無法打開視頻: {video_path}")

        # 基本特徵提取
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 採樣幀進行分析
        ai_indicators = 0
        real_indicators = 0

        sample_interval = max(total_frames // 30, 1)

        for i in range(0, total_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 特徵1: 亮度異常
            brightness = np.mean(gray)
            if brightness > 160 or brightness < 70:
                ai_indicators += 1
            else:
                real_indicators += 1

            # 特徵2: 飽和度異常
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            saturation = np.mean(hsv[:, :, 1])
            if saturation > 150:
                ai_indicators += 1
            else:
                real_indicators += 1

            # 特徵3: 對比度
            contrast = np.std(gray)
            if contrast > 70:
                ai_indicators += 1
            else:
                real_indicators += 1

        cap.release()

        # 計算AI分數
        total_indicators = ai_indicators + real_indicators
        ai_score = (ai_indicators / total_indicators * 100) if total_indicators > 0 else 50.0

        # 分辨率特徵（AI常用正方形）
        if width == height and width in [512, 1024, 768]:
            ai_score += 15  # AI生成常用分辨率

        # fps特徵（AI常用24fps）
        if abs(fps - 24) < 1:
            ai_score += 5

        ai_score = min(100, max(0, ai_score))

        # 分類
        classification, confidence = self._classify_by_score(ai_score)

        return {
            'video_path': str(video_path),
            'classification': classification,
            'confidence': confidence,
            'ai_score': ai_score,
            'fps': fps,
            'resolution': f"{width}x{height}"
        }

    def _classify_by_score(self, ai_score: float) -> tuple:
        """
        根據AI分數進行分類

        Args:
            ai_score: AI分數 (0-100)

        Returns:
            (分類, 信心度)
        """
        # 電影動畫檢測（特殊規則）
        # TODO: 可以加入更複雜的電影動畫檢測邏輯

        if ai_score < 30:
            # REAL
            classification = 'REAL'
            confidence = 100 - ai_score  # 越低越確定是真實
        elif ai_score >= 70:
            # AI
            classification = 'AI'
            confidence = ai_score  # 越高越確定是AI
        else:
            # NOT_SURE (30-70之間)
            classification = 'NOT_SURE'
            # 信心度：離邊界越遠越不確定
            distance_to_boundary = min(abs(ai_score - 30), abs(ai_score - 70))
            confidence = 50 - distance_to_boundary

        return classification, max(0, min(100, confidence))

    def batch_detect(self, video_files: List[Path] = None) -> List[Dict]:
        """
        批量檢測視頻

        Args:
            video_files: 視頻文件列表（None則自動掃描目錄）

        Returns:
            檢測結果列表
        """
        if video_files is None:
            video_files = list(self.video_dir.glob("*.mp4"))

        logger.info(f"🚀 開始批量檢測: {len(video_files)} 個視頻（並行數: {self.max_workers}）")

        results = []

        # 並行檢測
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.detect_single_video, vf): vf for vf in video_files}

            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)

                # 進度顯示
                if i % 10 == 0 or i == len(video_files):
                    logger.info(f"📊 檢測進度: {i}/{len(video_files)} ({i/len(video_files)*100:.1f}%)")

        # 統計
        stats = self._calculate_stats(results)
        logger.info(f"\n{'='*80}")
        logger.info(f"檢測完成:")
        logger.info(f"  • 總計: {stats['total']}")
        logger.info(f"  • REAL: {stats['real']} ({stats['real_pct']:.1f}%)")
        logger.info(f"  • AI: {stats['ai']} ({stats['ai_pct']:.1f}%)")
        logger.info(f"  • NOT_SURE: {stats['not_sure']} ({stats['not_sure_pct']:.1f}%)")
        logger.info(f"  • 電影動畫: {stats['movie']} ({stats['movie_pct']:.1f}%)")
        logger.info(f"{'='*80}\n")

        return results

    def _calculate_stats(self, results: List[Dict]) -> Dict:
        """
        計算統計信息

        Args:
            results: 檢測結果列表

        Returns:
            統計字典
        """
        total = len(results)
        real_count = sum(1 for r in results if r['classification'] == 'REAL')
        ai_count = sum(1 for r in results if r['classification'] == 'AI')
        not_sure_count = sum(1 for r in results if r['classification'] == 'NOT_SURE')
        movie_count = sum(1 for r in results if r['classification'] == '電影動畫')

        return {
            'total': total,
            'real': real_count,
            'ai': ai_count,
            'not_sure': not_sure_count,
            'movie': movie_count,
            'real_pct': real_count / total * 100 if total > 0 else 0,
            'ai_pct': ai_count / total * 100 if total > 0 else 0,
            'not_sure_pct': not_sure_count / total * 100 if total > 0 else 0,
            'movie_pct': movie_count / total * 100 if total > 0 else 0
        }


def main():
    """測試檢測器"""
    import argparse

    parser = argparse.ArgumentParser(description="AI檢測分類器")
    parser.add_argument(
        '--video-dir',
        type=str,
        required=True,
        help='視頻目錄'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='並行檢測數'
    )

    args = parser.parse_args()

    # 創建檢測器
    detector = AIDetectionClassifier(
        video_dir=args.video_dir,
        max_workers=args.workers
    )

    # 批量檢測
    results = detector.batch_detect()

    print(f"\n✅ 檢測完成！共 {len(results)} 個視頻")


if __name__ == "__main__":
    main()

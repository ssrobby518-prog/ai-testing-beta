#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSAR-RAPTOR Local Review Interface (Extended for Layer 2)
本地Tinder式復審界面 - 處理AI檢測的"不確定"視頻

設計原則:
- 第一性原理: 人類是終極判定者
- 沙皇炸彈: 快速復審，數據爆炸
- 猛禽3: 簡約界面，極速操作

功能（Layer 2 擴展）:
1. 從 "not sure" 文件夾加載視頻
2. Tinder式快速復審（← Real | → AI | ↓ Movie/Anime）
3. 存儲復審結果
4. **自動移動已復審視頻到對應文件夾（real/ai/電影動畫）**
5. **更新 Excel D 人工復審結果**
"""

import cv2
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Optional
import time
import os
import sys
import platform
import shutil

# 添加項目路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LocalReviewer:
    """本地Tinder式復審界面（Layer 2 擴展版）"""

    def __init__(
        self,
        uncertain_videos: List[str],
        output_csv: str = "review_results.csv",
        base_output_dir: str = None,
        excel_d_path: str = None,
        auto_move_files: bool = True
    ):
        """
        Args:
            uncertain_videos: 不確定視頻路徑列表
            output_csv: 復審結果輸出CSV
            base_output_dir: 分類文件夾根目錄（用於移動文件）
            excel_d_path: Excel D 路徑（用於更新復審結果）
            auto_move_files: 是否自動移動已復審文件
        """
        self.uncertain_videos = uncertain_videos
        self.output_csv = Path(output_csv)
        self.current_index = 0
        self.results = []
        self.auto_move_files = auto_move_files
        self.excel_d_path = Path(excel_d_path) if excel_d_path else None

        # 設置分類文件夾
        if base_output_dir:
            self.base_output_dir = Path(base_output_dir)
        elif uncertain_videos:
            # 自動推斷：從 "not sure" 文件夾的父目錄
            first_video_path = Path(uncertain_videos[0])
            if first_video_path.parent.name == "not sure":
                self.base_output_dir = first_video_path.parent.parent
            else:
                self.base_output_dir = first_video_path.parent
        else:
            self.base_output_dir = None

        # 分類文件夾映射
        if self.base_output_dir:
            self.classification_folders = {
                'real': self.base_output_dir / 'real',
                'ai': self.base_output_dir / 'ai',
                '電影動畫': self.base_output_dir / '電影動畫'
            }
            # 確保文件夾存在
            for folder in self.classification_folders.values():
                folder.mkdir(parents=True, exist_ok=True)
        else:
            self.classification_folders = {}

        logger.info(f"本地復審器初始化完成（Layer 2擴展）")
        logger.info(f"  • 待復審視頻: {len(uncertain_videos)}")
        logger.info(f"  • 自動移動文件: {auto_move_files}")
        if self.base_output_dir:
            logger.info(f"  • 分類根目錄: {self.base_output_dir}")
        if self.excel_d_path:
            logger.info(f"  • Excel D: {self.excel_d_path}")

    def play_video(self, video_path: str):
        """
        播放視頻（使用系統默認播放器）

        Args:
            video_path: 視頻路徑
        """
        try:
            system = platform.system()
            if system == 'Windows':
                os.startfile(video_path)
            elif system == 'Darwin':  # macOS
                os.system(f'open "{video_path}"')
            elif system == 'Linux':
                os.system(f'xdg-open "{video_path}"')

            logger.info(f"▶️  播放視頻: {Path(video_path).name}")
        except Exception as e:
            logger.error(f"❌ 無法播放視頻: {e}")
            print(f"請手動打開: {video_path}")

    def show_video_thumbnail(self, video_path: str):
        """
        顯示視頻縮略圖（使用OpenCV）

        Args:
            video_path: 視頻路徑
        """
        try:
            cap = cv2.VideoCapture(video_path)

            # 讀取第一幀
            ret, frame = cap.read()
            if ret:
                # 調整大小
                height, width = frame.shape[:2]
                max_size = 800
                if width > max_size or height > max_size:
                    scale = max_size / max(width, height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))

                # 顯示
                cv2.imshow('Video Thumbnail - Press any key to close', frame)
                cv2.waitKey(2000)  # 顯示2秒
                cv2.destroyAllWindows()

            cap.release()
        except Exception as e:
            logger.error(f"❌ 無法顯示縮略圖: {e}")

    def review_single_video(self, video_path: str, ai_prediction: float = None) -> Optional[Dict]:
        """
        復審單個視頻

        Args:
            video_path: 視頻路徑
            ai_prediction: AI預測分數（可選）

        Returns:
            復審結果字典
        """
        print(f"\n{'='*80}")
        print(f"Tinder式復審 - 視頻 {self.current_index + 1}/{len(self.uncertain_videos)}")
        print(f"{'='*80}")
        print(f"📹 視頻: {Path(video_path).name}")
        print(f"📍 路徑: {video_path}")

        if ai_prediction is not None:
            print(f"🤖 AI預測: {ai_prediction:.1f}% (不確定)")

        print(f"\n{'─'*80}")
        print("操作指南:")
        print("  ← (l) - Real（真實）")
        print("  → (r) - AI（生成）")
        print("  ↓ (m) - Movie/Anime（電影/動畫）")
        print("  s - Skip（跳過）")
        print("  p - Play（播放視頻）")
        print("  t - Thumbnail（顯示縮略圖）")
        print("  q - Quit（退出）")
        print(f"{'─'*80}\n")

        # 自動播放視頻
        self.play_video(video_path)

        # 等待用戶輸入
        while True:
            choice = input("你的判斷 (l/r/m/s/p/t/q): ").lower().strip()

            if choice in ['l', 'left', '←']:
                label = 'real'
                print("✅ 標註為 Real")
                break
            elif choice in ['r', 'right', '→']:
                label = 'ai'
                print("🤖 標註為 AI")
                break
            elif choice in ['m', 'movie', 'down', '↓']:
                label = '電影動畫'
                print("🎬 標註為 電影/動畫")
                break
            elif choice == 's':
                print("⏭️  跳過此視頻")
                return None
            elif choice == 'p':
                self.play_video(video_path)
            elif choice == 't':
                self.show_video_thumbnail(video_path)
            elif choice == 'q':
                print("👋 退出復審")
                return {'quit': True}
            else:
                print("❌ 無效輸入，請輸入 l/r/m/s/p/t/q")

        # 信心等級
        while True:
            try:
                confidence = int(input("信心等級 (1-5): ").strip())
                if 1 <= confidence <= 5:
                    break
                else:
                    print("❌ 請輸入 1-5")
            except ValueError:
                print("❌ 請輸入數字")

        # 備註
        notes = input("備註（可選，直接Enter跳過）: ").strip()

        # 創建結果
        result = {
            'video_path': video_path,
            'filename': Path(video_path).name,
            'ai_prediction': ai_prediction,
            'human_label': label,
            'human_confidence': confidence,
            'notes': notes,
            'timestamp': pd.Timestamp.now()
        }

        # Layer 2 擴展: 自動移動文件
        if self.auto_move_files and self.classification_folders:
            moved_path = self._move_reviewed_video(video_path, label)
            if moved_path:
                result['moved_to'] = str(moved_path)

        # Layer 2 擴展: 更新 Excel D
        if self.excel_d_path and self.excel_d_path.exists():
            self._update_excel_d(video_path, label, notes)

        return result

    def _move_reviewed_video(self, video_path: str, classification: str) -> Optional[Path]:
        """
        移動已復審視頻到對應分類文件夾

        Args:
            video_path: 視頻路徑
            classification: 分類標籤 (real/ai/電影動畫)

        Returns:
            新文件路徑（成功）或 None（失敗）
        """
        try:
            source_path = Path(video_path)
            target_folder = self.classification_folders.get(classification)

            if not target_folder:
                logger.warning(f"⚠️  未知分類: {classification}")
                return None

            # 生成目標路徑（保持原文件名或重命名）
            video_id = self._extract_video_id(source_path)
            label_lower = classification.lower().replace(' ', '_')
            new_filename = f"{label_lower}_{video_id}.mp4"
            target_path = target_folder / new_filename

            # 移動文件
            shutil.move(str(source_path), str(target_path))
            logger.info(f"📦 文件已移動: {source_path.name} → {target_path}")

            return target_path

        except Exception as e:
            logger.error(f"❌ 移動文件失敗: {e}")
            return None

    def _update_excel_d(self, video_path: str, human_label: str, notes: str = ""):
        """
        更新 Excel D 的人工復審結果

        Args:
            video_path: 視頻路徑
            human_label: 人工標籤
            notes: 備註
        """
        try:
            # 讀取 Excel D
            df = pd.read_excel(self.excel_d_path)

            # 提取視頻ID
            video_id = self._extract_video_id(Path(video_path))

            # 查找對應行
            mask = df['視頻ID'].astype(str) == video_id

            if mask.sum() == 0:
                logger.warning(f"⚠️  Excel D 中找不到視頻: {video_id}")
                return

            # 更新復審信息
            df.loc[mask, '人工復審結果'] = human_label.upper()
            df.loc[mask, '復審時間'] = pd.Timestamp.now().isoformat()
            df.loc[mask, '備註'] = notes

            # 保存
            df.to_excel(self.excel_d_path, index=False)
            logger.info(f"📝 Excel D 已更新: {video_id} → {human_label}")

        except Exception as e:
            logger.error(f"❌ 更新 Excel D 失敗: {e}")

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

    def batch_review(self):
        """
        批量復審所有不確定視頻

        Returns:
            復審統計
        """
        logger.info(f"🚀 開始批量復審: {len(self.uncertain_videos)} 個視頻")

        stats = {'reviewed': 0, 'skipped': 0, 'real': 0, 'ai': 0}

        for i, video_path in enumerate(self.uncertain_videos):
            self.current_index = i

            # 復審
            result = self.review_single_video(video_path)

            if result is None:
                stats['skipped'] += 1
                continue

            if result.get('quit'):
                logger.info("用戶退出復審")
                break

            # 保存結果
            self.results.append(result)
            stats['reviewed'] += 1

            if result['human_label'] == 'real':
                stats['real'] += 1
            elif result['human_label'] == 'ai':
                stats['ai'] += 1

            # 每10個視頻顯示一次統計
            if (i + 1) % 10 == 0:
                self._show_progress(stats, i + 1)

        # 保存結果到CSV
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(self.output_csv, index=False)
            logger.info(f"✅ 復審結果已保存: {self.output_csv}")

        # 最終統計
        print(f"\n{'='*80}")
        print("批量復審完成:")
        print(f"  • 總計: {len(self.uncertain_videos)}")
        print(f"  • 已復審: {stats['reviewed']}")
        print(f"  • 跳過: {stats['skipped']}")
        print(f"  • Real: {stats['real']}")
        print(f"  • AI: {stats['ai']}")
        print(f"{'='*80}\n")

        return stats

    def _show_progress(self, stats: Dict, current: int):
        """顯示進度統計"""
        print(f"\n📊 進度統計 ({current}/{len(self.uncertain_videos)}):")
        print(f"  • 已復審: {stats['reviewed']}")
        print(f"  • Real: {stats['real']}")
        print(f"  • AI: {stats['ai']}")
        print(f"  • 跳過: {stats['skipped']}\n")


def load_uncertain_videos_from_detection_results(
    detection_results_csv: str,
    video_dir: str
) -> List[str]:
    """
    從AI檢測結果中加載不確定視頻

    Args:
        detection_results_csv: AI檢測結果CSV
        video_dir: 視頻目錄

    Returns:
        不確定視頻路徑列表
    """
    if not Path(detection_results_csv).exists():
        logger.error(f"❌ 檢測結果文件不存在: {detection_results_csv}")
        return []

    df = pd.read_csv(detection_results_csv)

    # 過濾不確定視頻（20 < AI_P < 60）
    df_uncertain = df[(df['ai_probability'] > 20) & (df['ai_probability'] < 60)]

    logger.info(f"✅ 找到 {len(df_uncertain)} 個不確定視頻")

    # 構建完整路徑
    video_paths = []
    for _, row in df_uncertain.iterrows():
        filename = row['filename']
        filepath = Path(video_dir) / filename
        if filepath.exists():
            video_paths.append(str(filepath))
        else:
            logger.warning(f"⚠️  視頻文件不存在: {filepath}")

    return video_paths


def load_uncertain_videos_from_folder(
    not_sure_folder: str
) -> List[str]:
    """
    從 "not sure" 文件夾加載視頻（Layer 2專用）

    Args:
        not_sure_folder: "not sure" 文件夾路徑

    Returns:
        視頻路徑列表
    """
    not_sure_folder = Path(not_sure_folder)

    if not not_sure_folder.exists():
        logger.error(f"❌ not sure 文件夾不存在: {not_sure_folder}")
        return []

    # 獲取所有 mp4 文件
    video_files = list(not_sure_folder.glob("*.mp4"))

    logger.info(f"✅ 從 not sure 文件夾找到 {len(video_files)} 個視頻")

    return [str(vf) for vf in video_files]


def main():
    """測試本地復審器"""
    import argparse

    parser = argparse.ArgumentParser(description="本地Tinder式復審界面")
    parser.add_argument(
        '--detection-results',
        type=str,
        help='AI檢測結果CSV路徑'
    )
    parser.add_argument(
        '--video-dir',
        type=str,
        default='../../data/tiktok_videos',
        help='視頻目錄'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./review_results.csv',
        help='復審結果輸出CSV'
    )
    parser.add_argument(
        '--videos',
        type=str,
        nargs='+',
        help='手動指定視頻路徑列表'
    )

    args = parser.parse_args()

    # 加載不確定視頻
    if args.videos:
        uncertain_videos = args.videos
    elif args.detection_results:
        uncertain_videos = load_uncertain_videos_from_detection_results(
            args.detection_results,
            args.video_dir
        )
    else:
        # 測試模式：掃描視頻目錄
        video_dir = Path(args.video_dir)
        if video_dir.exists():
            uncertain_videos = [str(p) for p in video_dir.glob("*.mp4")]
            logger.info(f"測試模式：找到 {len(uncertain_videos)} 個視頻")
        else:
            logger.error(f"❌ 視頻目錄不存在: {video_dir}")
            return

    if not uncertain_videos:
        logger.error("❌ 沒有待復審的視頻")
        return

    # 創建復審器
    reviewer = LocalReviewer(
        uncertain_videos=uncertain_videos,
        output_csv=args.output
    )

    # 開始復審
    stats = reviewer.batch_review()

    print(f"\n✅ 復審完成！復審結果已保存到: {args.output}")


if __name__ == "__main__":
    main()

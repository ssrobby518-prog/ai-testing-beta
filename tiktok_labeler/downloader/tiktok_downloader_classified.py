#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSAR-RAPTOR TikTok Downloader with Auto-Classification
根據Excel A批量下載TikTok視頻並自動分類到對應文件夾

設計原則:
- 第一性原理: 增量下載，避免重複
- 沙皇炸彈: 批量並行，極速下載
- 猛禽3: 簡約接口，自動分類

功能:
1. 讀取Excel A
2. 根據標籤分類下載到對應文件夾:
   - real → tiktok tinder videos/real/
   - ai → tiktok tinder videos/ai/
   - uncertain → tiktok tinder videos/not sure/
   - exclude → tiktok tinder videos/movies/
3. 自動重試失敗任務
"""

import subprocess
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re
import sys

# 導入配置
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR.parent))
from config import EXCEL_A_PATH, LAYER1_VIDEO_FOLDERS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TikTokDownloaderClassified:
    """TikTok 視頻批量下載器（自動分類版）"""

    def __init__(
        self,
        excel_a_path: str = None,
        max_workers: int = 4,
        retry_times: int = 3
    ):
        """
        Args:
            excel_a_path: Excel A 路徑（None則使用配置文件）
            max_workers: 並行下載數
            retry_times: 失敗重試次數
        """
        self.excel_a_path = Path(excel_a_path) if excel_a_path else EXCEL_A_PATH
        self.video_folders = LAYER1_VIDEO_FOLDERS
        self.max_workers = max_workers
        self.retry_times = retry_times

        # 確保所有文件夾存在
        for folder in self.video_folders.values():
            folder.mkdir(parents=True, exist_ok=True)

        logger.info("TikTok下載器（自動分類）初始化完成")
        logger.info(f"  • Excel A: {self.excel_a_path}")
        logger.info(f"  • 分類文件夾:")
        for label, folder in self.video_folders.items():
            logger.info(f"    - {label}: {folder}")
        logger.info(f"  • 並行數: {self.max_workers}")

    def load_labels(self) -> pd.DataFrame:
        """
        加載標註數據

        Returns:
            DataFrame
        """
        if not self.excel_a_path.exists():
            logger.error(f"❌ Excel A 不存在: {self.excel_a_path}")
            return pd.DataFrame()

        df = pd.read_excel(self.excel_a_path)
        logger.info(f"✅ 已加載 {len(df)} 條標註")

        # 兼容處理：支持舊格式和新格式
        if '判定結果' in df.columns:
            label_col = '判定結果'
            df['label_lower'] = df[label_col].str.lower()
        else:
            label_col = 'label'
            df['label_lower'] = df[label_col].str.lower()

        return df

    def get_download_tasks(self, df: pd.DataFrame) -> List[Dict]:
        """
        生成下載任務列表（增量下載 + 自動分類）

        Args:
            df: 標註數據

        Returns:
            下載任務列表
        """
        tasks = []

        # 獲取所有已下載的視頻ID
        existing_ids = self._get_all_existing_video_ids()

        # 兼容處理：支持舊格式和新格式
        url_col = '影片網址' if '影片網址' in df.columns else 'url'
        video_id_col = '視頻ID' if '視頻ID' in df.columns else 'video_id'
        author_col = '作者' if '作者' in df.columns else 'author'

        for _, row in df.iterrows():
            video_id = str(row[video_id_col])
            label_lower = row['label_lower']
            url = row[url_col]

            # 檢查是否已下載
            if video_id in existing_ids:
                continue

            # 映射標籤到文件夾
            folder_key = self._map_label_to_folder_key(label_lower)
            if folder_key not in self.video_folders:
                logger.warning(f"⚠️  未知標籤: {label_lower}，跳過")
                continue

            target_folder = self.video_folders[folder_key]

            # 生成文件名：{label}_{video_id}.mp4
            filename = f"{label_lower}_{video_id}.mp4"
            filepath = target_folder / filename

            tasks.append({
                'url': url,
                'video_id': video_id,
                'label': label_lower,
                'folder_key': folder_key,
                'filepath': filepath,
                'author': row.get(author_col, 'unknown')
            })

        logger.info(f"📥 待下載: {len(tasks)} 個視頻（已下載: {len(existing_ids)}）")
        return tasks

    def _get_all_existing_video_ids(self) -> set:
        """
        獲取所有已下載的視頻ID（從所有分類文件夾）

        Returns:
            視頻ID集合
        """
        existing_ids = set()

        for folder in self.video_folders.values():
            if folder.exists():
                for file in folder.glob("*.mp4"):
                    # 提取視頻ID
                    match = re.search(r'_(\d+)\.mp4$', file.name)
                    if match:
                        existing_ids.add(match.group(1))

        return existing_ids

    def _map_label_to_folder_key(self, label: str) -> str:
        """
        映射標籤到文件夾鍵

        Args:
            label: 標籤 (real/ai/uncertain/exclude)

        Returns:
            文件夾鍵
        """
        label = label.lower()

        if label == 'real':
            return 'real'
        elif label == 'ai':
            return 'ai'
        elif label in ['uncertain', 'not_sure', 'not sure']:
            return 'uncertain'
        elif label in ['exclude', 'movie', 'movies', '電影', '動畫', '電影動畫']:
            return 'exclude'
        else:
            # 默認為 uncertain
            logger.warning(f"⚠️  未識別的標籤: {label}，默認為 uncertain")
            return 'uncertain'

    def download_single_video(self, task: Dict) -> Dict:
        """
        下載單個視頻到指定分類文件夾

        Args:
            task: 下載任務

        Returns:
            結果字典
        """
        url = task['url']
        filepath = task['filepath']
        video_id = task['video_id']
        folder_key = task['folder_key']

        for attempt in range(1, self.retry_times + 1):
            try:
                logger.info(f"⬇️  [{folder_key}] 下載: {video_id} (嘗試 {attempt}/{self.retry_times})")

                # 使用 yt-dlp 下載
                cmd = [
                    'yt-dlp',
                    '-o', str(filepath),
                    '--quiet',
                    '--no-warnings',
                    url
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120  # 2分鐘超時
                )

                if result.returncode == 0 and filepath.exists():
                    file_size = filepath.stat().st_size / (1024 * 1024)  # MB
                    logger.info(f"✅ [{folder_key}] {video_id} 下載成功 ({file_size:.2f} MB)")
                    return {
                        'status': 'success',
                        'video_id': video_id,
                        'label': task['label'],
                        'folder_key': folder_key,
                        'filepath': str(filepath),
                        'file_size_mb': file_size
                    }
                else:
                    error_msg = result.stderr if result.stderr else "未知錯誤"
                    logger.warning(f"⚠️  [{folder_key}] {video_id} 下載失敗: {error_msg[:100]}")

            except subprocess.TimeoutExpired:
                logger.warning(f"⏱️  [{folder_key}] {video_id} 超時")
            except Exception as e:
                logger.error(f"❌ [{folder_key}] {video_id} 異常: {e}")

            # 重試前等待
            if attempt < self.retry_times:
                time.sleep(2)

        # 所有嘗試失敗
        return {
            'status': 'failed',
            'video_id': video_id,
            'label': task['label'],
            'folder_key': folder_key,
            'error': f'重試 {self.retry_times} 次後仍失敗'
        }

    def batch_download(self, tasks: List[Dict]) -> Dict:
        """
        批量下載視頻

        Args:
            tasks: 下載任務列表

        Returns:
            統計結果
        """
        if not tasks:
            logger.info("✅ 無需下載")
            return {'success': 0, 'failed': 0}

        logger.info(f"🚀 開始批量下載: {len(tasks)} 個視頻（並行數: {self.max_workers}）")

        success_count = 0
        failed_count = 0
        failed_videos = []
        success_by_category = {
            'real': 0,
            'ai': 0,
            'uncertain': 0,
            'exclude': 0
        }

        # 並行下載
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.download_single_video, task): task for task in tasks}

            for future in as_completed(futures):
                result = future.result()
                if result['status'] == 'success':
                    success_count += 1
                    folder_key = result['folder_key']
                    if folder_key in success_by_category:
                        success_by_category[folder_key] += 1
                else:
                    failed_count += 1
                    failed_videos.append(result['video_id'])

        logger.info(f"\n{'='*80}")
        logger.info(f"下載完成:")
        logger.info(f"  ✅ 成功: {success_count}")
        logger.info(f"  ❌ 失敗: {failed_count}")
        logger.info(f"  分類統計:")
        logger.info(f"    - Real: {success_by_category['real']}")
        logger.info(f"    - AI: {success_by_category['ai']}")
        logger.info(f"    - Uncertain: {success_by_category['uncertain']}")
        logger.info(f"    - Movies: {success_by_category['exclude']}")
        if failed_videos:
            logger.info(f"  失敗列表: {', '.join(failed_videos[:10])}...")
        logger.info(f"{'='*80}\n")

        return {
            'success': success_count,
            'failed': failed_count,
            'failed_videos': failed_videos,
            'by_category': success_by_category
        }

    def download_from_excel_a(self) -> Dict:
        """
        完整流程：從Excel A下載視頻並自動分類

        Returns:
            統計結果
        """
        # 1. 加載標註
        df = self.load_labels()
        if df.empty:
            return {'success': 0, 'failed': 0}

        # 2. 生成下載任務
        tasks = self.get_download_tasks(df)

        # 3. 批量下載
        stats = self.batch_download(tasks)

        return stats


def main():
    """主程式"""
    import argparse

    parser = argparse.ArgumentParser(description="TikTok視頻批量下載器（自動分類）")
    parser.add_argument(
        '--excel-a',
        type=str,
        help='Excel A 路徑（默認使用配置文件）'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='並行下載數'
    )

    args = parser.parse_args()

    # 創建下載器
    downloader = TikTokDownloaderClassified(
        excel_a_path=args.excel_a,
        max_workers=args.workers
    )

    # 執行下載
    stats = downloader.download_from_excel_a()

    print(f"\n✅ 下載任務完成！")
    print(f"   成功: {stats['success']} 個")
    print(f"   失敗: {stats['failed']} 個")
    if 'by_category' in stats:
        print(f"   分類統計:")
        print(f"     • Real: {stats['by_category']['real']}")
        print(f"     • AI: {stats['by_category']['ai']}")
        print(f"     • Uncertain: {stats['by_category']['uncertain']}")
        print(f"     • Movies: {stats['by_category']['exclude']}")


if __name__ == "__main__":
    main()

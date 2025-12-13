#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSAR-RAPTOR TikTok Downloader
根據Excel A批量下載TikTok視頻

設計原則:
- 第一性原理: 增量下載，避免重複
- 沙皇炸彈: 批量並行，極速下載
- 猛禽3: 簡約接口，自動重試

功能:
1. 讀取Excel A (labels_raw.xlsx)
2. 過濾需要下載的視頻（排除 exclude 類別）
3. 使用 yt-dlp 批量下載
4. 自動重命名：{label}_{video_id}.mp4
5. 下載失敗自動重試
"""

import subprocess
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TikTokDownloader:
    """TikTok 視頻批量下載器"""

    def __init__(
        self,
        excel_a_path: str,
        download_dir: str = "downloaded_videos",
        max_workers: int = 4,
        retry_times: int = 3
    ):
        """
        Args:
            excel_a_path: Excel A 路徑
            download_dir: 下載目錄
            max_workers: 並行下載數
            retry_times: 失敗重試次數
        """
        self.excel_a_path = Path(excel_a_path)
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.retry_times = retry_times

        logger.info("TikTok下載器初始化完成")
        logger.info(f"  • Excel A: {self.excel_a_path}")
        logger.info(f"  • 下載目錄: {self.download_dir}")
        logger.info(f"  • 並行數: {self.max_workers}")

    def load_labels(self, exclude_labels: List[str] = ['exclude']) -> pd.DataFrame:
        """
        加載標註數據

        Args:
            exclude_labels: 排除的標籤類別

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
            # 新格式（中文列名）
            label_col = '判定結果'
            # 轉換為小寫以統一處理
            df['label_lower'] = df[label_col].str.lower()
        else:
            # 舊格式（英文列名）
            label_col = 'label'
            df['label_lower'] = df[label_col].str.lower()

        # 過濾排除類別
        df_filtered = df[~df['label_lower'].isin([x.lower() for x in exclude_labels])]
        logger.info(f"   過濾後剩餘 {len(df_filtered)} 條（排除: {exclude_labels}）")

        return df_filtered

    def get_download_tasks(self, df: pd.DataFrame) -> List[Dict]:
        """
        生成下載任務列表（增量下載）

        Args:
            df: 標註數據

        Returns:
            下載任務列表
        """
        tasks = []
        existing_files = set(self.download_dir.glob("*.mp4"))
        existing_ids = {f.stem.split('_')[-1] for f in existing_files}

        # 兼容處理：支持舊格式和新格式
        url_col = '影片網址' if '影片網址' in df.columns else 'url'
        video_id_col = '視頻ID' if '視頻ID' in df.columns else 'video_id'
        author_col = '作者' if '作者' in df.columns else 'author'

        for _, row in df.iterrows():
            video_id = str(row[video_id_col])
            # 使用統一的 label_lower 列
            label = row['label_lower']
            url = row[url_col]

            # 檢查是否已下載
            if video_id in existing_ids:
                continue

            # 生成文件名：{label}_{video_id}.mp4
            filename = f"{label}_{video_id}.mp4"
            filepath = self.download_dir / filename

            tasks.append({
                'url': url,
                'video_id': video_id,
                'label': label,
                'filepath': filepath,
                'author': row.get(author_col, 'unknown')
            })

        logger.info(f"📥 待下載: {len(tasks)} 個視頻（已下載: {len(existing_ids)}）")
        return tasks

    def download_single_video(self, task: Dict) -> Dict:
        """
        下載單個視頻

        Args:
            task: 下載任務

        Returns:
            結果字典
        """
        url = task['url']
        filepath = task['filepath']
        video_id = task['video_id']

        for attempt in range(1, self.retry_times + 1):
            try:
                logger.info(f"⬇️  下載中: {video_id} (嘗試 {attempt}/{self.retry_times})")

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
                    logger.info(f"✅ 下載成功: {video_id} ({file_size:.2f} MB)")
                    return {
                        'status': 'success',
                        'video_id': video_id,
                        'filepath': str(filepath),
                        'file_size_mb': file_size
                    }
                else:
                    error_msg = result.stderr if result.stderr else "未知錯誤"
                    logger.warning(f"⚠️  下載失敗: {video_id} | {error_msg}")

            except subprocess.TimeoutExpired:
                logger.warning(f"⏱️  超時: {video_id}")
            except Exception as e:
                logger.error(f"❌ 異常: {video_id} | {e}")

            # 重試前等待
            if attempt < self.retry_times:
                time.sleep(2)

        # 所有嘗試失敗
        return {
            'status': 'failed',
            'video_id': video_id,
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

        # 並行下載
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.download_single_video, task): task for task in tasks}

            for future in as_completed(futures):
                result = future.result()
                if result['status'] == 'success':
                    success_count += 1
                else:
                    failed_count += 1
                    failed_videos.append(result['video_id'])

        logger.info(f"\n{'='*80}")
        logger.info(f"下載完成:")
        logger.info(f"  ✅ 成功: {success_count}")
        logger.info(f"  ❌ 失敗: {failed_count}")
        if failed_videos:
            logger.info(f"  失敗列表: {', '.join(failed_videos)}")
        logger.info(f"{'='*80}\n")

        return {
            'success': success_count,
            'failed': failed_count,
            'failed_videos': failed_videos
        }

    def download_from_excel_a(self, exclude_labels: List[str] = ['exclude']) -> Dict:
        """
        完整流程：從Excel A下載視頻

        Args:
            exclude_labels: 排除的標籤類別

        Returns:
            統計結果
        """
        # 1. 加載標註
        df = self.load_labels(exclude_labels)
        if df.empty:
            return {'success': 0, 'failed': 0}

        # 2. 生成下載任務
        tasks = self.get_download_tasks(df)

        # 3. 批量下載
        stats = self.batch_download(tasks)

        return stats


def main():
    """測試下載器"""
    import argparse

    parser = argparse.ArgumentParser(description="TikTok視頻批量下載器")
    parser.add_argument(
        '--excel-a',
        type=str,
        default='../../data/tiktok_labels/excel_a_labels_raw.xlsx',
        help='Excel A 路徑'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../../data/tiktok_videos',
        help='下載目錄'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='並行下載數'
    )
    parser.add_argument(
        '--include-exclude',
        action='store_true',
        help='包含 exclude 類別（電影/動畫）'
    )

    args = parser.parse_args()

    # 創建下載器
    downloader = TikTokDownloader(
        excel_a_path=args.excel_a,
        download_dir=args.output,
        max_workers=args.workers
    )

    # 執行下載
    exclude_labels = [] if args.include_exclude else ['exclude']
    stats = downloader.download_from_excel_a(exclude_labels=exclude_labels)

    print(f"\n✅ 下載任務完成！")
    print(f"   成功: {stats['success']} 個")
    print(f"   失敗: {stats['failed']} 個")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSAR-RAPTOR TikTok Mass Downloader
批量下載2000個TikTok視頻

設計原則:
- 第一性原理: 增量下載，斷點續傳
- 沙皇炸彈: 並行下載，極速完成
- 猛禽3: 簡約接口，自動重試

功能:
1. 從URL列表批量下載
2. 並行下載（可配置線程數）
3. 自動重試失敗任務
4. 增量下載（跳過已下載）
5. 進度追蹤和統計
"""

import subprocess
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TikTokMassDownloader:
    """TikTok 批量下載器（海量下載專用）"""

    def __init__(
        self,
        url_list_file: str = None,
        urls: List[str] = None,
        download_dir: str = "../../tiktok videos download",
        max_workers: int = 8,
        retry_times: int = 3,
        target_count: int = 2000
    ):
        """
        Args:
            url_list_file: URL列表文件路徑
            urls: URL列表（直接提供）
            download_dir: 下載目錄
            max_workers: 並行下載數
            retry_times: 失敗重試次數
            target_count: 目標下載數量
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.retry_times = retry_times
        self.target_count = target_count

        # 加載URL列表
        self.urls = []
        if urls:
            self.urls = urls
        elif url_list_file:
            self.urls = self._load_urls_from_file(url_list_file)

        logger.info("TikTok批量下載器初始化完成")
        logger.info(f"  • 下載目錄: {self.download_dir}")
        logger.info(f"  • URL數量: {len(self.urls)}")
        logger.info(f"  • 並行數: {self.max_workers}")
        logger.info(f"  • 目標數量: {self.target_count}")

    def _load_urls_from_file(self, file_path: str) -> List[str]:
        """
        從文件加載URL列表

        Args:
            file_path: 文件路徑

        Returns:
            URL列表
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"❌ URL列表文件不存在: {file_path}")
            return []

        urls = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                url = line.strip()
                if url and url.startswith('http'):
                    urls.append(url)

        logger.info(f"✅ 從文件加載 {len(urls)} 個URL")
        return urls

    def _extract_video_id(self, url: str) -> str:
        """
        從URL提取視頻ID

        Args:
            url: TikTok URL

        Returns:
            視頻ID
        """
        match = re.search(r'/video/(\d+)', url)
        if match:
            return match.group(1)
        return str(hash(url))[:10]  # 備用方案

    def get_existing_video_ids(self) -> Set[str]:
        """
        獲取已下載的視頻ID列表

        Returns:
            已下載的視頻ID集合
        """
        existing_files = list(self.download_dir.glob("*.mp4"))
        existing_ids = set()

        for file in existing_files:
            # 文件名格式: download_7123456789.mp4
            match = re.search(r'(\d+)', file.stem)
            if match:
                existing_ids.add(match.group(1))

        return existing_ids

    def create_download_tasks(self) -> List[Dict]:
        """
        創建下載任務列表（增量下載）

        Returns:
            下載任務列表
        """
        existing_ids = self.get_existing_video_ids()
        tasks = []

        for url in self.urls[:self.target_count]:
            video_id = self._extract_video_id(url)

            # 跳過已下載
            if video_id in existing_ids:
                continue

            # 生成文件名: download_{video_id}.mp4
            filename = f"download_{video_id}.mp4"
            filepath = self.download_dir / filename

            tasks.append({
                'url': url,
                'video_id': video_id,
                'filepath': filepath
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
                # 使用 yt-dlp 下載
                cmd = [
                    'yt-dlp',
                    '-o', str(filepath),
                    '--quiet',
                    '--no-warnings',
                    '--no-check-certificate',  # 忽略SSL證書錯誤
                    url
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=180  # 3分鐘超時
                )

                if result.returncode == 0 and filepath.exists():
                    file_size = filepath.stat().st_size / (1024 * 1024)  # MB
                    logger.info(f"✅ [{video_id}] 下載成功 ({file_size:.2f} MB)")
                    return {
                        'status': 'success',
                        'video_id': video_id,
                        'url': url,
                        'filepath': str(filepath),
                        'file_size_mb': file_size
                    }
                else:
                    error_msg = result.stderr if result.stderr else "未知錯誤"
                    logger.warning(f"⚠️  [{video_id}] 嘗試 {attempt}/{self.retry_times} 失敗: {error_msg[:100]}")

            except subprocess.TimeoutExpired:
                logger.warning(f"⏱️  [{video_id}] 超時（嘗試 {attempt}/{self.retry_times}）")
            except Exception as e:
                logger.error(f"❌ [{video_id}] 異常: {e}")

            # 重試前等待
            if attempt < self.retry_times:
                time.sleep(3)

        # 所有嘗試失敗
        return {
            'status': 'failed',
            'video_id': video_id,
            'url': url,
            'error': f'重試 {self.retry_times} 次後仍失敗'
        }

    def batch_download(self) -> Dict:
        """
        批量下載視頻

        Returns:
            統計結果
        """
        # 創建任務
        tasks = self.create_download_tasks()

        if not tasks:
            logger.info("✅ 無需下載（所有視頻已存在）")
            return {'success': 0, 'failed': 0, 'skipped': len(self.urls)}

        logger.info(f"🚀 開始批量下載: {len(tasks)} 個視頻（並行數: {self.max_workers}）")
        logger.info(f"   預計時間: {len(tasks) * 15 / self.max_workers / 60:.1f} 分鐘")

        success_count = 0
        failed_count = 0
        failed_videos = []
        success_videos = []

        start_time = time.time()

        # 並行下載
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.download_single_video, task): task for task in tasks}

            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()

                if result['status'] == 'success':
                    success_count += 1
                    success_videos.append(result)
                else:
                    failed_count += 1
                    failed_videos.append(result)

                # 進度顯示
                if i % 10 == 0 or i == len(tasks):
                    elapsed = time.time() - start_time
                    progress = i / len(tasks) * 100
                    logger.info(f"📊 進度: {i}/{len(tasks)} ({progress:.1f}%) | "
                               f"成功: {success_count} | 失敗: {failed_count} | "
                               f"耗時: {elapsed/60:.1f}分鐘")

        elapsed_total = time.time() - start_time

        logger.info(f"\n{'='*80}")
        logger.info(f"下載完成:")
        logger.info(f"  ✅ 成功: {success_count}")
        logger.info(f"  ❌ 失敗: {failed_count}")
        logger.info(f"  ⏱️  總耗時: {elapsed_total/60:.1f} 分鐘")
        logger.info(f"  📊 平均速度: {success_count/(elapsed_total/60):.1f} 個/分鐘")
        if failed_videos:
            logger.info(f"  失敗列表: {', '.join([v['video_id'] for v in failed_videos[:10]])}...")
        logger.info(f"{'='*80}\n")

        return {
            'success': success_count,
            'failed': failed_count,
            'success_videos': success_videos,
            'failed_videos': failed_videos,
            'elapsed_minutes': elapsed_total / 60
        }


def main():
    """主程式"""
    import argparse

    parser = argparse.ArgumentParser(description="TikTok批量下載器")
    parser.add_argument(
        '--url-list',
        type=str,
        required=True,
        help='URL列表文件路徑'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../../tiktok videos download',
        help='下載目錄'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='並行下載數'
    )
    parser.add_argument(
        '--target',
        type=int,
        default=2000,
        help='目標下載數量'
    )

    args = parser.parse_args()

    # 創建下載器
    downloader = TikTokMassDownloader(
        url_list_file=args.url_list,
        download_dir=args.output,
        max_workers=args.workers,
        target_count=args.target
    )

    # 執行下載
    stats = downloader.batch_download()

    print(f"\n✅ 批量下載完成！")
    print(f"   成功: {stats['success']} 個")
    print(f"   失敗: {stats['failed']} 個")
    print(f"   耗時: {stats['elapsed_minutes']:.1f} 分鐘")


if __name__ == "__main__":
    main()

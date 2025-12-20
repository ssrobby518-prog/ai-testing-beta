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
import os
import hashlib
from urllib.parse import urlparse

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
        max_workers: int = 8,
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
        self.yt_dlp_cmd = self._resolve_yt_dlp_cmd()
        self.cookies_from_browser = os.environ.get('YTDLP_COOKIES_FROM_BROWSER', '').strip()
        self.proxy = os.environ.get('YTDLP_PROXY', '').strip()

        # 確保所有文件夾存在
        for folder in self.video_folders.values():
            folder.mkdir(parents=True, exist_ok=True)

        logger.info("TikTok下載器（自動分類）初始化完成")
        logger.info(f"  • Excel A: {self.excel_a_path}")
        logger.info(f"  • 分類文件夾:")
        for label, folder in self.video_folders.items():
            logger.info(f"    - {label}: {folder}")
        logger.info(f"  • 並行數: {self.max_workers}")
        logger.info(f"  • yt-dlp: {' '.join(self.yt_dlp_cmd)}")

    def _resolve_yt_dlp_cmd(self) -> List[str]:
        try:
            subprocess.run(["yt-dlp", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return ["yt-dlp"]
        except Exception:
            try:
                subprocess.run([sys.executable, "-m", "yt_dlp", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                return [sys.executable, "-m", "yt_dlp"]
            except Exception:
                return ["yt-dlp"]

    def _is_allowed_tiktok_url(self, url: str) -> bool:
        u = str(url or "").strip()
        if not u:
            return False
        try:
            parsed = urlparse(u if "://" in u else f"https://{u}")
        except Exception:
            return False

        host = (parsed.hostname or "").lower()
        if not host:
            return False
        if host == "tiktok.com" or host.endswith(".tiktok.com"):
            return True
        return False

    def _extract_video_id_from_url(self, url: str) -> str:
        """
        提取視頻ID（支持多種URL格式）

        優先級：
        1. /video/{id} 格式（標準TikTok URL）
        2. video_id= 或 item_id= 參數
        3. URL中的任意數字ID（放寬限制：支持短ID用於測試）
        """
        s = str(url or "")

        # 標準 TikTok URL: /video/123456789
        m = re.search(r"/video/([a-zA-Z0-9_-]+)", s)
        if m:
            return m.group(1)

        # URL 參數: ?video_id=123 或 ?item_id=123
        m = re.search(r"(?:video_id=|item_id=)([a-zA-Z0-9_-]+)", s)
        if m:
            return m.group(1)

        # 兼容短ID（測試數據）：3-22位數字
        m = re.search(r"(?<!\d)(\d{3,22})(?!\d)", s)
        if m:
            return m.group(1)

        return ""

    def _normalize_video_id(self, video_id_value, url: str) -> str:
        extracted = self._extract_video_id_from_url(url)
        if extracted:
            return extracted

        raw = str(video_id_value).strip()
        if raw.endswith(".0") and raw[:-2].isdigit():
            raw = raw[:-2]

        if raw and raw.lower() not in {"nan", "none"}:
            return raw

        url_str = str(url or "").strip()
        if url_str:
            return hashlib.md5(url_str.encode("utf-8")).hexdigest()[:10]
        return ""

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
            df['label_lower'] = df[label_col].astype(str).str.lower()
        else:
            label_col = 'label'
            df['label_lower'] = df[label_col].astype(str).str.lower()

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

        if 'label_lower' not in df.columns:
            if '判定結果' in df.columns:
                df = df.copy()
                df['label_lower'] = df['判定結果'].astype(str).str.lower()
            elif 'label' in df.columns:
                df = df.copy()
                df['label_lower'] = df['label'].astype(str).str.lower()
            else:
                df = df.copy()
                df['label_lower'] = ""

        # 獲取所有已下載的視頻ID
        existing_ids = self._get_all_existing_video_ids()

        # 兼容處理：支持舊格式和新格式
        url_col = '影片網址' if '影片網址' in df.columns else 'url'
        video_id_col = '視頻ID' if '視頻ID' in df.columns else 'video_id'
        author_col = '作者' if '作者' in df.columns else 'author'

        for _, row in df.iterrows():
            url = str(row.get(url_col, '')).strip()
            if not self._is_allowed_tiktok_url(url):
                logger.warning(f"⚠️  非 TikTok URL，跳過: {url}")
                continue

            video_id = self._normalize_video_id(row.get(video_id_col, ''), url)
            label_lower = row['label_lower']

            # 如果無法提取 video_id，跳過
            if not video_id:
                logger.warning(f"⚠️  無法提取 video ID，跳過: {url}")
                continue

            if not url or not video_id:
                continue

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
                    match = re.search(r'_(.+)\.mp4$', file.name)
                    if match:
                        vid = match.group(1).strip()
                        if vid:
                            existing_ids.add(vid)
                    else:
                        match2 = re.search(r'^(.+)\.mp4$', file.name)
                        if match2:
                            vid = match2.group(1).strip()
                            if vid:
                                existing_ids.add(vid)

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

        if filepath.exists() and filepath.stat().st_size > 0:
            file_size = filepath.stat().st_size / (1024 * 1024)
            return {
                'status': 'success',
                'video_id': video_id,
                'label': task['label'],
                'folder_key': folder_key,
                'filepath': str(filepath),
                'file_size_mb': file_size
            }

        for attempt in range(1, self.retry_times + 1):
            try:
                logger.info(f"⬇️  [{folder_key}] 下載: {video_id} (嘗試 {attempt}/{self.retry_times})")

                # 使用 yt-dlp 下載（完整瀏覽器模擬 + 多重防護）
                base_cmd = [
                    *self.yt_dlp_cmd,
                    '-o', str(filepath),
                    '--quiet',
                    '--no-warnings',
                    '--no-check-certificate',
                    # 完整瀏覽器模擬
                    '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    '--add-header', 'Accept:text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                    '--add-header', 'Accept-Language:en-US,en;q=0.9',
                    '--add-header', 'Accept-Encoding:gzip, deflate, br',
                    '--add-header', 'DNT:1',
                    '--add-header', 'Connection:keep-alive',
                    '--add-header', 'Upgrade-Insecure-Requests:1',
                    '--add-header', 'Sec-Fetch-Dest:document',
                    '--add-header', 'Sec-Fetch-Mode:navigate',
                    '--add-header', 'Sec-Fetch-Site:none',
                    '--add-header', 'Sec-Fetch-User:?1',
                    '--add-header', 'sec-ch-ua:"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                    '--add-header', 'sec-ch-ua-mobile:?0',
                    '--add-header', 'sec-ch-ua-platform:"Windows"',
                    # Referer 模擬從 TikTok 主頁訪問
                    '--add-header', 'Referer:https://www.tiktok.com/foryou',
                    # 延遲設置（更保守）
                    '--sleep-requests', '4',  # 請求間隔4秒
                    '--sleep-interval', '2',  # 下載片段間隔2秒
                    '--max-sleep-interval', '6',  # 最大隨機延遲6秒
                    # 重試設置
                    '--retries', '10',  # yt-dlp內建重試（增加）
                    '--fragment-retries', '10',
                    # 限速模擬真實用戶（可選）
                    # '--limit-rate', '1M',  # 限速1MB/s
                ]
                if self.proxy:
                    base_cmd += ['--proxy', self.proxy]

                # 不使用 cookies（避免 Chrome 數據庫鎖定問題）
                cmd = [*base_cmd, url]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 增加到5分鐘（更多重試時間）
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

                    # 檢測特定錯誤類型
                    error_lower = error_msg.lower()
                    if 'ip address is blocked' in error_lower or 'ip' in error_lower and 'block' in error_lower:
                        logger.error(f"🚫 [{folder_key}] {video_id} IP被封鎖，請使用代理或瀏覽器cookies")
                        return {
                            'status': 'failed',
                            'video_id': video_id,
                            'label': task['label'],
                            'folder_key': folder_key,
                            'error': 'IP被封鎖 - 請配置代理或cookies'
                        }
                    elif 'private' in error_lower or 'unavailable' in error_lower:
                        logger.warning(f"🔒 [{folder_key}] {video_id} 視頻私密或不可用")
                        return {
                            'status': 'failed',
                            'video_id': video_id,
                            'label': task['label'],
                            'folder_key': folder_key,
                            'error': '視頻私密或不可用'
                        }

                    logger.warning(f"⚠️  [{folder_key}] {video_id} 下載失敗: {error_msg[:150]}")

            except subprocess.TimeoutExpired:
                logger.warning(f"⏱️  [{folder_key}] {video_id} 超時")
            except Exception as e:
                logger.error(f"❌ [{folder_key}] {video_id} 異常: {e}")

            # 重試前等待（IP封鎖時增加延遲）
            if attempt < self.retry_times:
                wait_time = 10 + (attempt * 5)  # 漸進式延遲：10s, 15s, 20s（更保守）
                logger.info(f"⏳ 等待 {wait_time} 秒後重試...")
                time.sleep(wait_time)

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
        all_results = []  # 收集所有結果用於更新 Excel A
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
                all_results.append(result)  # 收集結果

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
            'by_category': success_by_category,
            'results': all_results  # 用於更新 Excel A 狀態
        }

    def update_download_status(self, results: List[Dict]):
        """
        更新 Excel A 的下載狀態

        Args:
            results: 下載結果列表
        """
        if not results:
            return

        try:
            # 讀取 Excel A
            df = pd.read_excel(self.excel_a_path)

            # 確保有下載狀態列
            if '下載狀態' not in df.columns:
                df['下載狀態'] = '未下載'

            # 兼容處理
            url_col = '影片網址' if '影片網址' in df.columns else 'url'
            video_id_col = '視頻ID' if '視頻ID' in df.columns else 'video_id'

            # 更新狀態
            for result in results:
                video_id = result['video_id']
                status = result['status']

                # 找到對應行
                mask = df[video_id_col].astype(str) == str(video_id)

                if mask.any():
                    if status == 'success':
                        df.loc[mask, '下載狀態'] = '已下載'
                    else:
                        error = result.get('error', '下載失敗')
                        df.loc[mask, '下載狀態'] = f'下載失敗: {error}'

            # 保存
            df.to_excel(self.excel_a_path, index=False)
            logger.info(f"✅ 已更新 Excel A 下載狀態")

        except Exception as e:
            logger.error(f"❌ 更新下載狀態失敗: {e}")

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

        # 4. 更新下載狀態到 Excel A
        if 'results' in stats:
            self.update_download_status(stats['results'])

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
        default=8,
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

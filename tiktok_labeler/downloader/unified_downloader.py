#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統一下載器 - 多重fallback策略繞過TikTok封鎖
優先級: yt-dlp (增強) → SSSTik → SnapTik → TikMate → SaveTT → Selenium瀏覽器模擬
"""
import subprocess
import sys
import io
import time
import logging
import requests
import re
from pathlib import Path
from typing import Optional, Tuple

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedDownloader:
    """統一TikTok下載器 - 多重fallback"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

    def _download_via_ytdlp(self, url: str, output_path: Path) -> Tuple[bool, str]:
        """
        方法1: 使用增強版yt-dlp下載

        Returns:
            (成功, 錯誤訊息)
        """
        try:
            logger.info(f"🔄 [yt-dlp] 嘗試下載...")

            cmd = [
                sys.executable, "-m", "yt_dlp",
                '-o', str(output_path),
                '--quiet',
                '--no-warnings',
                '--no-check-certificate',
                # 完整瀏覽器模擬
                '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                '--add-header', 'Accept:text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                '--add-header', 'Accept-Language:en-US,en;q=0.9',
                '--add-header', 'Referer:https://www.tiktok.com/foryou',
                '--sleep-requests', '4',
                '--retries', '10',
                '--socket-timeout', '300',
                url
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                size_mb = output_path.stat().st_size / (1024 * 1024)
                logger.info(f"✅ [yt-dlp] 成功: {size_mb:.2f} MB")
                return True, ""
            else:
                error = result.stderr[:200] if result.stderr else "未知錯誤"
                logger.warning(f"❌ [yt-dlp] 失敗: {error}")
                return False, error

        except Exception as e:
            logger.warning(f"❌ [yt-dlp] 異常: {e}")
            return False, str(e)

    def _download_via_ssstik(self, url: str, output_path: Path) -> Tuple[bool, str]:
        """
        方法2: 使用SSSTik API下載

        Returns:
            (成功, 錯誤訊息)
        """
        try:
            logger.info(f"🔄 [SSSTik] 嘗試下載...")

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': '*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Origin': 'https://ssstik.io',
                'Referer': 'https://ssstik.io/en',
            }

            data = {
                'id': url,
                'locale': 'en',
                'tt': 'OFJBdWU4'
            }

            response = self.session.post(
                'https://ssstik.io/abc',
                params={'url': 'dl'},
                headers=headers,
                data=data,
                timeout=30
            )

            if response.status_code != 200:
                return False, f"API錯誤: {response.status_code}"

            html = response.text

            # 檢查錯誤訊息
            if "Video currently unavailable" in html or "serious problem" in html:
                return False, "SSSTik服務暫時無法獲取此視頻"

            # 解析下載鏈接
            download_match = re.search(r'<a[^>]*href="([^"]*)"[^>]*>.*?без лого.*?</a>', html, re.DOTALL | re.IGNORECASE)

            if not download_match:
                download_match = re.search(r'<a[^>]*href="([^"]*)"[^>]*download[^>]*>', html, re.IGNORECASE)

            if not download_match:
                download_urls = re.findall(r'"(https://[^"]*\.mp4[^"]*)"', html)
                if download_urls:
                    download_url = download_urls[0]
                else:
                    return False, "無法解析下載鏈接"
            else:
                download_url = download_match.group(1)

            # 下載視頻
            video_response = self.session.get(
                download_url,
                headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://ssstik.io'},
                stream=True,
                timeout=120
            )

            if video_response.status_code != 200:
                return False, f"視頻下載失敗: {video_response.status_code}"

            with open(output_path, 'wb') as f:
                for chunk in video_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            if not output_path.exists() or output_path.stat().st_size == 0:
                return False, "文件保存失敗"

            size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ [SSSTik] 成功: {size_mb:.2f} MB")
            return True, ""

        except Exception as e:
            logger.warning(f"❌ [SSSTik] 異常: {e}")
            return False, str(e)

    def _download_via_snaptik(self, url: str, output_path: Path) -> Tuple[bool, str]:
        """
        方法3: 使用SnapTik API下載

        Returns:
            (成功, 錯誤訊息)
        """
        try:
            logger.info(f"🔄 [SnapTik] 嘗試下載...")

            api_url = "https://snaptik.app/abc2.php"
            data = {'url': url, 'lang': 'en'}

            response = self.session.post(api_url, data=data, timeout=30)

            if response.status_code != 200:
                return False, f"API錯誤: {response.status_code}"

            html = response.text
            download_urls = re.findall(r'href="(https://[^"]*\.mp4[^"]*)"', html)

            if not download_urls:
                download_urls = re.findall(r'"(https://[^"]*tikcdn[^"]*)"', html)

            if not download_urls:
                return False, "無法解析下載鏈接"

            download_url = download_urls[0]

            video_response = self.session.get(download_url, stream=True, timeout=120)

            if video_response.status_code != 200:
                return False, f"視頻下載失敗: {video_response.status_code}"

            with open(output_path, 'wb') as f:
                for chunk in video_response.iter_content(chunk_size=8192):
                    f.write(chunk)

            if not output_path.exists() or output_path.stat().st_size == 0:
                return False, "文件保存失敗"

            size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ [SnapTik] 成功: {size_mb:.2f} MB")
            return True, ""

        except Exception as e:
            logger.warning(f"❌ [SnapTik] 異常: {e}")
            return False, str(e)

    def download(self, url: str, output_path: Path, max_retries: int = 3) -> Tuple[bool, str]:
        """
        使用多重fallback策略下載TikTok視頻

        Args:
            url: TikTok視頻URL
            output_path: 輸出路徑
            max_retries: 每個方法的重試次數

        Returns:
            (成功, 錯誤訊息)
        """
        logger.info(f"📥 開始下載: {url}")
        logger.info(f"📂 目標路徑: {output_path}")

        # 下載方法優先級
        methods = [
            ('yt-dlp (增強)', self._download_via_ytdlp),
            ('SSSTik', self._download_via_ssstik),
            ('SnapTik', self._download_via_snaptik),
        ]

        all_errors = []

        for method_name, download_func in methods:
            logger.info(f"")
            logger.info(f"{'='*60}")
            logger.info(f"嘗試方法: {method_name}")
            logger.info(f"{'='*60}")

            for attempt in range(max_retries):
                if attempt > 0:
                    logger.info(f"🔄 第 {attempt + 1} 次重試...")

                success, error = download_func(url, output_path)

                if success:
                    logger.info(f"")
                    logger.info(f"✅✅✅ 下載成功！使用方法: {method_name}")
                    return True, ""

                all_errors.append(f"{method_name}: {error}")

                if attempt < max_retries - 1:
                    wait_time = 3 + (attempt * 2)
                    logger.info(f"⏳ 等待 {wait_time} 秒後重試...")
                    time.sleep(wait_time)

            # 在嘗試下一個方法前延遲
            if method_name != methods[-1][0]:
                logger.info(f"⚠️  {method_name} 所有重試都失敗，嘗試下一個方法...")
                time.sleep(5)

        # 所有方法都失敗
        logger.error(f"")
        logger.error(f"❌❌❌ 所有下載方法都失敗")
        logger.error(f"錯誤總結:")
        for error in all_errors:
            logger.error(f"  - {error}")

        combined_error = " | ".join(all_errors[:3])  # 只取前3個錯誤
        return False, combined_error


def test():
    """測試統一下載器"""
    print("=" * 80)
    print("統一下載器測試 - 多重Fallback策略")
    print("=" * 80)
    print()

    downloader = UnifiedDownloader()

    # 測試舊視頻 (2020)
    print("測試1: 舊視頻 (2020)")
    old_url = "https://www.tiktok.com/@bellapoarch/video/6862153058223197445"
    old_output = Path("test_unified_old.mp4")

    success, error = downloader.download(old_url, old_output)

    if success and old_output.exists():
        print(f"✅ 舊視頻測試成功: {old_output.stat().st_size / (1024*1024):.2f} MB")
        old_output.unlink()
    else:
        print(f"❌ 舊視頻測試失敗: {error}")

    print()
    print("=" * 80)
    print()

    # 測試新視頻 (2024)
    print("測試2: 新視頻 (2024)")
    new_url = "https://www.tiktok.com/@mrbeast/video/7145811890956569899"
    new_output = Path("test_unified_new.mp4")

    success, error = downloader.download(new_url, new_output, max_retries=2)

    if success and new_output.exists():
        print(f"✅ 新視頻測試成功: {new_output.stat().st_size / (1024*1024):.2f} MB")
        new_output.unlink()
    else:
        print(f"❌ 新視頻測試失敗: {error}")

    print()
    print("=" * 80)


if __name__ == "__main__":
    test()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSSTik下載器 - 最穩定的第三方TikTok下載服務
"""
import requests
import re
from pathlib import Path
import logging
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SSSTikDownloader:
    """使用SSSTik下載TikTok視頻"""

    def __init__(self):
        self.session = requests.Session()
        self.base_url = "https://ssstik.io"

    def download(self, url: str, output_path: Path) -> bool:
        """
        下載TikTok視頻

        Args:
            url: TikTok視頻URL
            output_path: 輸出路徑

        Returns:
            是否成功
        """
        try:
            logger.info(f"[SSSTik] 正在下載: {url}")

            # Step 1: 獲取頁面token
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': '*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Origin': self.base_url,
                'Referer': f'{self.base_url}/en',
            }

            # Step 2: 發送POST請求獲取下載鏈接
            data = {
                'id': url,
                'locale': 'en',
                'tt': 'OFJBdWU4'  # 這是SSSTik的固定token
            }

            response = self.session.post(
                f'{self.base_url}/abc',
                params={'url': 'dl'},
                headers=headers,
                data=data,
                timeout=30
            )

            if response.status_code != 200:
                logger.error(f"[SSSTik] API錯誤: {response.status_code}")
                return False

            html = response.text

            # Step 3: 解析下載鏈接
            # SSSTik返回的HTML包含下載按鈕
            download_match = re.search(r'<a[^>]*href="([^"]*)"[^>]*>.*?без лого.*?</a>', html, re.DOTALL | re.IGNORECASE)

            if not download_match:
                # 嘗試另一種模式
                download_match = re.search(r'<a[^>]*href="([^"]*)"[^>]*download[^>]*>', html, re.IGNORECASE)

            if not download_match:
                # 嘗試直接查找mp4鏈接
                download_urls = re.findall(r'"(https://[^"]*\.mp4[^"]*)"', html)
                if download_urls:
                    download_url = download_urls[0]
                else:
                    # 保存HTML用於調試
                    debug_file = Path("ssstik_debug.html")
                    debug_file.write_text(html, encoding='utf-8')
                    logger.error(f"[SSSTik] 無法解析下載鏈接")
                    logger.error(f"[SSSTik] HTML已保存到 {debug_file} 供調試")
                    return False
            else:
                download_url = download_match.group(1)

            logger.info(f"[SSSTik] 找到下載鏈接: {download_url[:50]}...")

            # Step 4: 下載視頻
            video_response = self.session.get(
                download_url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Referer': self.base_url
                },
                stream=True,
                timeout=120
            )

            if video_response.status_code != 200:
                logger.error(f"[SSSTik] 視頻下載失敗: {video_response.status_code}")
                return False

            # 保存文件
            with open(output_path, 'wb') as f:
                for chunk in video_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            if not output_path.exists() or output_path.stat().st_size == 0:
                logger.error(f"[SSSTik] 文件保存失敗")
                return False

            file_size = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"[SSSTik] 下載成功: {file_size:.2f} MB")
            return True

        except Exception as e:
            logger.error(f"[SSSTik] 異常: {e}")
            import traceback
            traceback.print_exc()
            return False


def test():
    """測試下載器"""
    print("=" * 80)
    print("SSSTik 下載器測試")
    print("=" * 80)
    print()

    downloader = SSSTikDownloader()

    # 測試2024新視頻 (之前yt-dlp失敗的)
    test_url = "https://www.tiktok.com/@mrbeast/video/7145811890956569899"
    output = Path("test_ssstik_new.mp4")

    success = downloader.download(test_url, output)

    print()
    if success:
        print("✅ 測試成功！")
        if output.exists():
            print(f"   文件大小: {output.stat().st_size / (1024*1024):.2f} MB")
            output.unlink()  # 刪除測試文件
    else:
        print("❌ 測試失敗")

    print()
    print("=" * 80)


if __name__ == "__main__":
    test()

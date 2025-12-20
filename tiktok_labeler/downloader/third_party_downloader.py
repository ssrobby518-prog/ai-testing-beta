#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第三方API下載器 - 使用SnapTik/TikMate等服務API繞過IP封鎖
"""
import requests
import re
from pathlib import Path
import logging
from typing import Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThirdPartyDownloader:
    """使用第三方服務下載TikTok視頻"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

    def download_via_snaptik(self, url: str, output_path: Path) -> bool:
        """
        使用 SnapTik API 下載

        Args:
            url: TikTok視頻URL
            output_path: 輸出路徑

        Returns:
            是否成功
        """
        try:
            logger.info(f"📥 [SnapTik] 正在下載: {url}")

            # Step 1: 獲取下載鏈接
            api_url = "https://snaptik.app/abc2.php"
            data = {
                'url': url,
                'lang': 'en'
            }

            response = self.session.post(api_url, data=data, timeout=30)

            if response.status_code != 200:
                logger.error(f"❌ [SnapTik] API返回錯誤: {response.status_code}")
                return False

            # Step 2: 解析下載鏈接
            # SnapTik返回的HTML中包含下載鏈接
            html = response.text

            # 查找下載鏈接（通常是 .mp4 URL）
            download_urls = re.findall(r'href="(https://[^"]*\.mp4[^"]*)"', html)

            if not download_urls:
                # 嘗試另一種模式
                download_urls = re.findall(r'"(https://[^"]*tikcdn[^"]*)"', html)

            if not download_urls:
                logger.error(f"❌ [SnapTik] 無法解析下載鏈接")
                return False

            download_url = download_urls[0]
            logger.info(f"✅ [SnapTik] 找到下載鏈接")

            # Step 3: 下載視頻
            video_response = self.session.get(download_url, stream=True, timeout=120)

            if video_response.status_code != 200:
                logger.error(f"❌ [SnapTik] 視頻下載失敗: {video_response.status_code}")
                return False

            # 保存文件
            with open(output_path, 'wb') as f:
                for chunk in video_response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ [SnapTik] 下載成功: {file_size:.2f} MB")
            return True

        except Exception as e:
            logger.error(f"❌ [SnapTik] 下載異常: {e}")
            return False

    def download_via_tikmate(self, url: str, output_path: Path) -> bool:
        """
        使用 TikMate API 下載

        Args:
            url: TikTok視頻URL
            output_path: 輸出路徑

        Returns:
            是否成功
        """
        try:
            logger.info(f"📥 [TikMate] 正在下載: {url}")

            # TikMate API
            api_url = "https://tikmate.app/download"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Content-Type': 'application/x-www-form-urlencoded',
                'Origin': 'https://tikmate.app',
                'Referer': 'https://tikmate.app/'
            }

            data = {'url': url}

            response = self.session.post(api_url, data=data, headers=headers, timeout=30)

            if response.status_code != 200:
                logger.error(f"❌ [TikMate] API返回錯誤: {response.status_code}")
                return False

            # 解析JSON響應
            try:
                result = response.json()
                if 'token' in result:
                    download_url = f"https://tikmate.app/download/{result['token']}.mp4"
                else:
                    logger.error(f"❌ [TikMate] 無效響應")
                    return False
            except:
                # 嘗試從HTML解析
                html = response.text
                download_urls = re.findall(r'href="(/download/[^"]+\.mp4)"', html)

                if not download_urls:
                    logger.error(f"❌ [TikMate] 無法解析下載鏈接")
                    return False

                download_url = "https://tikmate.app" + download_urls[0]

            logger.info(f"✅ [TikMate] 找到下載鏈接")

            # 下載視頻
            video_response = self.session.get(download_url, stream=True, timeout=120)

            if video_response.status_code != 200:
                logger.error(f"❌ [TikMate] 視頻下載失敗: {video_response.status_code}")
                return False

            with open(output_path, 'wb') as f:
                for chunk in video_response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ [TikMate] 下載成功: {file_size:.2f} MB")
            return True

        except Exception as e:
            logger.error(f"❌ [TikMate] 下載異常: {e}")
            return False

    def download_via_savett(self, url: str, output_path: Path) -> bool:
        """
        使用 SaveTT API 下載

        Args:
            url: TikTok視頻URL
            output_path: 輸出路徑

        Returns:
            是否成功
        """
        try:
            logger.info(f"📥 [SaveTT] 正在下載: {url}")

            # SaveTT使用簡單的GET請求
            api_url = f"https://www.savett.cc/api/ajaxSearch"

            data = {
                'q': url,
                'lang': 'en'
            }

            response = self.session.post(api_url, data=data, timeout=30)

            if response.status_code != 200:
                logger.error(f"❌ [SaveTT] API返回錯誤: {response.status_code}")
                return False

            result = response.json()

            if result.get('status') != 'ok':
                logger.error(f"❌ [SaveTT] API返回失敗狀態")
                return False

            # 查找視頻下載鏈接
            html_data = result.get('data', '')
            download_urls = re.findall(r'href="(https://[^"]*\.mp4[^"]*)"', html_data)

            if not download_urls:
                logger.error(f"❌ [SaveTT] 無法解析下載鏈接")
                return False

            download_url = download_urls[0]
            logger.info(f"✅ [SaveTT] 找到下載鏈接")

            # 下載視頻
            video_response = self.session.get(download_url, stream=True, timeout=120)

            if video_response.status_code != 200:
                logger.error(f"❌ [SaveTT] 視頻下載失敗: {video_response.status_code}")
                return False

            with open(output_path, 'wb') as f:
                for chunk in video_response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ [SaveTT] 下載成功: {file_size:.2f} MB")
            return True

        except Exception as e:
            logger.error(f"❌ [SaveTT] 下載異常: {e}")
            return False

    def download(self, url: str, output_path: Path) -> bool:
        """
        使用多個服務嘗試下載（優先級順序）

        Args:
            url: TikTok視頻URL
            output_path: 輸出路徑

        Returns:
            是否成功
        """
        services = [
            ('SnapTik', self.download_via_snaptik),
            ('TikMate', self.download_via_tikmate),
            ('SaveTT', self.download_via_savett),
        ]

        for service_name, download_func in services:
            try:
                logger.info(f"🔄 嘗試使用 {service_name}...")

                if download_func(url, output_path):
                    return True

                logger.warning(f"⚠️  {service_name} 失敗，嘗試下一個...")
                time.sleep(2)  # 延遲避免觸發限制

            except Exception as e:
                logger.error(f"❌ {service_name} 異常: {e}")
                continue

        logger.error(f"❌ 所有服務都失敗")
        return False


def test():
    """測試下載器"""
    downloader = ThirdPartyDownloader()

    test_url = "https://www.tiktok.com/@mrbeast/video/7145811890956569899"
    output = Path("test_third_party.mp4")

    print("=" * 80)
    print("第三方API下載器測試")
    print("=" * 80)
    print()

    success = downloader.download(test_url, output)

    if success:
        print()
        print("✅ 測試成功！")
        output.unlink()  # 刪除測試文件
    else:
        print()
        print("❌ 測試失敗")


if __name__ == "__main__":
    test()

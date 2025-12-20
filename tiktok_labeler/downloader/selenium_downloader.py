#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Selenium真實瀏覽器下載器 - 最終方案
使用真實Chrome瀏覽器訪問TikTok並攔截網絡請求獲取視頻URL
"""
import sys
import io
import time
import logging
import requests
from pathlib import Path
from typing import Optional, Tuple
from seleniumwire import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeleniumDownloader:
    """使用Selenium真實瀏覽器下載TikTok視頻"""

    def __init__(self, headless: bool = True):
        """
        Args:
            headless: 是否使用無頭模式（不顯示瀏覽器窗口）
        """
        self.headless = headless
        self.driver = None

    def _init_driver(self):
        """初始化Chrome瀏覽器"""
        if self.driver:
            return

        logger.info("🔧 初始化Chrome瀏覽器...")

        chrome_options = Options()

        if self.headless:
            chrome_options.add_argument('--headless=new')

        # 完整瀏覽器模擬 + 穩定性增強
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

        # 防止崩潰的關鍵選項
        chrome_options.add_argument('--disable-software-rasterizer')
        chrome_options.add_argument('--disable-extensions')
        chrome_options.add_argument('--disable-infobars')
        chrome_options.add_argument('--disable-notifications')
        chrome_options.add_argument('--disable-popup-blocking')
        chrome_options.add_argument('--ignore-certificate-errors')
        chrome_options.add_argument('--ignore-ssl-errors')
        chrome_options.add_argument('--disable-background-timer-throttling')
        chrome_options.add_argument('--disable-backgrounding-occluded-windows')
        chrome_options.add_argument('--disable-renderer-backgrounding')
        chrome_options.add_argument('--enable-features=NetworkService,NetworkServiceInProcess')
        chrome_options.add_argument('--force-color-profile=srgb')

        # 增加穩定性
        chrome_options.add_argument('--disable-features=VizDisplayCompositor')
        chrome_options.add_argument('--disable-crash-reporter')
        chrome_options.add_argument('--disable-in-process-stack-traces')

        # 內存優化
        chrome_options.add_argument('--js-flags=--max-old-space-size=4096')

        # 排除自動化特徵
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        # 禁用圖片和CSS加速加載
        prefs = {
            "profile.managed_default_content_settings.images": 2,
            "profile.default_content_setting_values.notifications": 2,
        }
        chrome_options.add_experimental_option("prefs", prefs)

        # Selenium Wire配置（用於攔截網絡請求）- 優化內存
        seleniumwire_options = {
            'disable_encoding': True,
            'verify_ssl': False,
            'connection_timeout': None,  # 無超時限制
            'suppress_connection_errors': True,  # 忽略連接錯誤
        }

        try:
            service = Service(ChromeDriverManager().install())
            service.creation_flags = 0x08000000  # CREATE_NO_WINDOW flag

            self.driver = webdriver.Chrome(
                service=service,
                options=chrome_options,
                seleniumwire_options=seleniumwire_options
            )

            # 設置頁面加載超時
            self.driver.set_page_load_timeout(60)
            self.driver.set_script_timeout(60)

            # 移除webdriver標誌
            self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': '''
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    })
                '''
            })

            logger.info("✅ 瀏覽器初始化成功")

        except Exception as e:
            logger.error(f"❌ 瀏覽器初始化失敗: {e}")
            raise

    def _extract_video_url(self, page_url: str, timeout: int = 30) -> Optional[str]:
        """
        訪問TikTok頁面並從網絡請求中提取視頻URL

        Args:
            page_url: TikTok視頻頁面URL
            timeout: 超時時間(秒)

        Returns:
            視頻下載URL，如果失敗返回None
        """
        try:
            logger.info(f"🌐 訪問TikTok頁面: {page_url}")

            # 清空之前的請求記錄
            del self.driver.requests

            # 訪問頁面
            self.driver.get(page_url)

            # 等待頁面加載完成
            logger.info("⏳ 等待頁面加載...")
            time.sleep(3)

            # 嘗試滾動頁面觸發視頻加載
            try:
                self.driver.execute_script("window.scrollTo(0, 500);")
                time.sleep(2)
                self.driver.execute_script("window.scrollTo(0, 0);")
            except:
                pass

            # 等待視頻請求完成
            logger.info("⏳ 等待視頻請求...")
            time.sleep(15)  # 增加等待時間確保視頻URL請求完成

            # 從網絡請求中查找視頻URL
            logger.info("🔍 分析網絡請求...")
            logger.info(f"📊 總共捕獲 {len(self.driver.requests)} 個請求")

            # 調試：打印包含"video"或"mp4"的所有URL
            video_related = []
            for request in self.driver.requests:
                try:
                    if request.url and ('video' in request.url.lower() or 'mp4' in request.url.lower() or 'webapp' in request.url):
                        video_related.append(request.url[:150])
                except:
                    pass

            if video_related:
                logger.info(f"🎬 發現 {len(video_related)} 個視頻相關請求:")
                for i, url in enumerate(video_related[:5], 1):  # 只顯示前5個
                    logger.info(f"  [{i}] {url}...")
            else:
                logger.warning("⚠️  未發現任何視頻相關請求")

            video_urls = []

            # 遍歷所有請求
            for request in self.driver.requests:
                try:
                    if not request.url:
                        continue

                    url = request.url

                    # TikTok視頻URL特徵：
                    # 1. 包含 v16-webapp, v19-webapp, v26-webapp 等
                    # 2. 包含 /video/tos/ 路徑
                    # 3. 包含 .mp4 擴展名或參數
                    is_video_url = (
                        ('v16-webapp' in url or 'v19-webapp' in url or 'v26-webapp' in url) or
                        ('/video/tos/' in url) or
                        ('.mp4' in url and 'tiktok' in url.lower())
                    )

                    if is_video_url:
                        # 檢查是否有響應（不強制要求）
                        if request.response:
                            try:
                                content_type = request.response.headers.get('Content-Type', '')
                                # 只接受視頻內容
                                if 'video' not in content_type and 'mp4' not in url:
                                    continue
                            except:
                                # 無法獲取headers，但URL看起來像視頻，仍然添加
                                pass

                        video_urls.append(url)
                        logger.info(f"✅ 找到視頻URL: {url[:100]}...")

                except Exception as e:
                    # 跳過有問題的請求
                    continue

            if video_urls:
                # 優先選擇包含最多TikTok特徵的URL
                best_url = video_urls[0]
                for url in video_urls:
                    if 'v16-webapp' in url or 'v19-webapp' in url or 'v26-webapp' in url:
                        if '/video/tos/' in url:
                            best_url = url
                            break

                logger.info(f"🎯 選擇視頻URL: {best_url[:150]}...")
                return best_url
            else:
                logger.error("❌ 未找到視頻URL")
                logger.error(f"💡 請檢查網絡請求日誌")
                return None

        except Exception as e:
            logger.error(f"❌ 提取視頻URL失敗: {e}")
            import traceback
            traceback.print_exc()
            return None

    def download(self, url: str, output_path: Path) -> Tuple[bool, str]:
        """
        下載TikTok視頻

        Args:
            url: TikTok視頻頁面URL
            output_path: 輸出路徑

        Returns:
            (成功, 錯誤訊息)
        """
        try:
            # 初始化瀏覽器
            self._init_driver()

            # 提取視頻URL
            video_url = self._extract_video_url(url)

            if not video_url:
                return False, "無法從頁面提取視頻URL"

            # 下載視頻
            logger.info(f"📥 下載視頻...")

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Referer': 'https://www.tiktok.com/'
            }

            response = requests.get(video_url, headers=headers, stream=True, timeout=120)

            if response.status_code != 200:
                return False, f"視頻下載失敗: HTTP {response.status_code}"

            # 保存文件
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            if not output_path.exists() or output_path.stat().st_size == 0:
                return False, "文件保存失敗"

            size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ 下載成功: {size_mb:.2f} MB")

            return True, ""

        except Exception as e:
            logger.error(f"❌ 下載異常: {e}")
            import traceback
            traceback.print_exc()
            return False, str(e)

    def close(self):
        """關閉瀏覽器"""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("🔒 瀏覽器已關閉")
            except:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def test():
    """測試Selenium下載器"""
    print("=" * 80)
    print("Selenium真實瀏覽器下載器測試")
    print("=" * 80)
    print()

    with SeleniumDownloader(headless=True) as downloader:
        # 測試舊視頻
        print("測試1: 舊視頻 (2020)")
        old_url = "https://www.tiktok.com/@bellapoarch/video/6862153058223197445"
        old_output = Path("test_selenium_old.mp4")

        success, error = downloader.download(old_url, old_output)

        if success and old_output.exists():
            print(f"✅ 舊視頻測試成功: {old_output.stat().st_size / (1024*1024):.2f} MB")
            old_output.unlink()
        else:
            print(f"❌ 舊視頻測試失敗: {error}")

        print()
        print("=" * 80)
        print()

        # 測試新視頻
        print("測試2: 新視頻 (2024)")
        new_url = "https://www.tiktok.com/@mrbeast/video/7145811890956569899"
        new_output = Path("test_selenium_new.mp4")

        success, error = downloader.download(new_url, new_output)

        if success and new_output.exists():
            print(f"✅ 新視頻測試成功: {new_output.stat().st_size / (1024*1024):.2f} MB")
            new_output.unlink()
        else:
            print(f"❌ 新視頻測試失敗: {error}")

    print()
    print("=" * 80)


if __name__ == "__main__":
    test()

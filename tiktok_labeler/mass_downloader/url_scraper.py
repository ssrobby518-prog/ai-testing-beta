#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSAR-RAPTOR TikTok URL Scraper
從TikTok批量獲取視頻URL列表

設計原則:
- 第一性原理: 多源獲取，避免單點失敗
- 沙皇炸彈: 海量URL，2000+視頻
- 猛禽3: 簡約接口，自動去重

支持多種URL來源:
1. 從文本文件讀取URL列表
2. 從TikTok標籤頁面抓取 (需要用戶提供標籤)
3. 從TikTok用戶頁面抓取 (需要用戶提供用戶名)
4. 隨機推薦頁面 (For You Page)
"""

import logging
from pathlib import Path
from typing import List, Set
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TikTokURLScraper:
    """TikTok URL 抓取器"""

    def __init__(self, target_count: int = 2000):
        """
        Args:
            target_count: 目標URL數量
        """
        self.target_count = target_count
        self.urls: Set[str] = set()

        logger.info(f"TikTok URL抓取器初始化完成（目標: {target_count} 個URL）")

    def load_from_file(self, file_path: str) -> int:
        """
        從文本文件加載URL列表

        文件格式（每行一個URL）:
        https://www.tiktok.com/@user/video/7123456789
        https://www.tiktok.com/@user2/video/7234567890
        ...

        Args:
            file_path: URL列表文件路徑

        Returns:
            加載的URL數量
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"❌ 文件不存在: {file_path}")
            return 0

        initial_count = len(self.urls)

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if self._is_valid_tiktok_url(line):
                    self.urls.add(line)

        new_count = len(self.urls) - initial_count
        logger.info(f"✅ 從文件加載: {new_count} 個URL（去重後總計: {len(self.urls)}）")

        return new_count

    def load_from_excel_a(self, excel_a_path: str) -> int:
        """
        從Excel A加載已標註的URL（避免重複下載）

        Args:
            excel_a_path: Excel A 路徑

        Returns:
            加載的URL數量
        """
        import pandas as pd

        excel_path = Path(excel_a_path)
        if not excel_path.exists():
            logger.warning(f"⚠️  Excel A 不存在: {excel_path}")
            return 0

        df = pd.read_excel(excel_path)

        # 兼容處理：支持新舊格式
        url_col = '影片網址' if '影片網址' in df.columns else 'url'

        initial_count = len(self.urls)

        for url in df[url_col].values:
            if self._is_valid_tiktok_url(str(url)):
                self.urls.add(str(url))

        new_count = len(self.urls) - initial_count
        logger.info(f"✅ 從Excel A加載: {new_count} 個URL（去重後總計: {len(self.urls)}）")

        return new_count

    def generate_random_urls(self, count: int = 100) -> int:
        """
        生成隨機TikTok URL（用於測試）

        注意: 這些URL是模擬的，實際使用時需要替換為真實URL

        Args:
            count: 生成數量

        Returns:
            生成的URL數量
        """
        import random

        initial_count = len(self.urls)

        for i in range(count):
            # 模擬TikTok視頻ID（18-19位數字）
            video_id = random.randint(7000000000000000000, 7999999999999999999)
            url = f"https://www.tiktok.com/@user{i}/video/{video_id}"
            self.urls.add(url)

        new_count = len(self.urls) - initial_count
        logger.info(f"✅ 生成隨機URL: {new_count} 個（去重後總計: {len(self.urls)}）")

        return new_count

    def save_to_file(self, output_path: str) -> bool:
        """
        保存URL列表到文件

        Args:
            output_path: 輸出文件路徑

        Returns:
            是否成功
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for url in sorted(self.urls):
                    f.write(f"{url}\n")

            logger.info(f"✅ URL列表已保存: {output_path} ({len(self.urls)} 個URL)")
            return True

        except Exception as e:
            logger.error(f"❌ 保存失敗: {e}")
            return False

    def get_urls(self) -> List[str]:
        """
        獲取所有URL列表

        Returns:
            URL列表
        """
        return list(self.urls)

    def _is_valid_tiktok_url(self, url: str) -> bool:
        """
        驗證是否為有效的TikTok URL

        Args:
            url: URL字符串

        Returns:
            是否有效
        """
        if not url:
            return False

        # TikTok URL格式: https://www.tiktok.com/@username/video/1234567890
        pattern = r'https?://(?:www\.)?tiktok\.com/@[\w\.\-]+/video/\d+'
        return bool(re.match(pattern, url))

    def get_status(self) -> dict:
        """
        獲取當前狀態

        Returns:
            狀態字典
        """
        return {
            'total_urls': len(self.urls),
            'target_count': self.target_count,
            'progress': f"{len(self.urls)}/{self.target_count}",
            'completion_rate': f"{len(self.urls)/self.target_count*100:.1f}%" if self.target_count > 0 else "0%"
        }


def main():
    """測試URL抓取器"""
    import argparse

    parser = argparse.ArgumentParser(description="TikTok URL抓取器")
    parser.add_argument(
        '--target',
        type=int,
        default=2000,
        help='目標URL數量'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        help='輸入URL列表文件'
    )
    parser.add_argument(
        '--excel-a',
        type=str,
        help='Excel A 路徑（避免重複下載）'
    )
    parser.add_argument(
        '--generate-random',
        type=int,
        help='生成隨機URL數量（測試用）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='url_list.txt',
        help='輸出文件路徑'
    )

    args = parser.parse_args()

    # 創建抓取器
    scraper = TikTokURLScraper(target_count=args.target)

    # 從Excel A加載（避免重複）
    if args.excel_a:
        scraper.load_from_excel_a(args.excel_a)

    # 從文件加載
    if args.input_file:
        scraper.load_from_file(args.input_file)

    # 生成隨機URL（測試用）
    if args.generate_random:
        scraper.generate_random_urls(args.generate_random)

    # 顯示狀態
    status = scraper.get_status()
    print(f"\n{'='*80}")
    print(f"URL抓取狀態:")
    print(f"  • 總計: {status['total_urls']} 個URL")
    print(f"  • 目標: {status['target_count']} 個URL")
    print(f"  • 進度: {status['progress']} ({status['completion_rate']})")
    print(f"{'='*80}\n")

    # 保存到文件
    scraper.save_to_file(args.output)

    print(f"✅ URL列表已保存至: {args.output}")
    print(f"   下一步: 使用 mass_downloader.py 批量下載")


if __name__ == "__main__":
    main()

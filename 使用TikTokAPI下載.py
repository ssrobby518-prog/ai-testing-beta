#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 TikTokApi + Playwright 真實瀏覽器下載
"""
import asyncio
from TikTokApi import TikTokApi
from pathlib import Path
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

async def download_video(video_url: str, output_path: Path):
    """使用TikTokApi下載視頻"""
    print(f"正在下載: {video_url}")

    async with TikTokApi() as api:
        await api.create_sessions(num_sessions=1, sleep_after=3, headless=True)

        # 從URL提取視頻ID
        video_id = video_url.split('/')[-1].split('?')[0]

        try:
            # 使用完整URL
            video = api.video(url=video_url)
            video_data = await video.info()

            # 獲取視頻下載URL
            video_bytes = await video.bytes()

            # 保存到文件
            with open(output_path, 'wb') as f:
                f.write(video_bytes)

            print(f"✅ 下載成功: {output_path}")
            print(f"   大小: {len(video_bytes) / (1024*1024):.2f} MB")
            return True

        except Exception as e:
            print(f"❌ 下載失敗: {e}")
            return False

async def main():
    print("=" * 80)
    print("TikTokApi 真實瀏覽器下載測試")
    print("=" * 80)
    print()

    # 測試視頻
    test_videos = [
        "https://www.tiktok.com/@mrbeast/video/7145811890956569899",
        "https://www.tiktok.com/@gordonramsayofficial/video/7285043558775836971",
        "https://www.tiktok.com/@therock/video/7283845742701112619",
    ]

    output_dir = Path(__file__).parent / "tiktok_api_downloads"
    output_dir.mkdir(exist_ok=True)

    success_count = 0
    for url in test_videos:
        video_id = url.split('/')[-1].split('?')[0]
        output_file = output_dir / f"{video_id}.mp4"

        if await download_video(url, output_file):
            success_count += 1

        print()
        await asyncio.sleep(5)  # 延遲避免封鎖

    print("=" * 80)
    print(f"完成: {success_count}/{len(test_videos)} 個視頻下載成功")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())

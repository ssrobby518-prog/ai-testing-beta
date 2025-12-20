@echo off
chcp 65001 >nul
echo ========================================
echo TikTok 下載器 - 使用 Chrome Cookies
echo ========================================
echo.

REM 設置環境變量
set YTDLP_COOKIES_FROM_BROWSER=chrome

echo [1/3] 配置完成
echo   - 使用瀏覽器: Chrome
echo   - Cookies: 已啟用
echo.

echo [2/3] 開始下載...
echo.

REM 執行下載
python tiktok_downloader_classified.py --workers 4

echo.
echo [3/3] 下載完成
echo.
echo 按任意鍵退出...
pause >nul

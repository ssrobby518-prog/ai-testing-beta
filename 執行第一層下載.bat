@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo 執行第一層下載
echo ========================================
echo.

cd tiktok_labeler\pipeline
python layer1_pipeline.py

echo.
echo 按任意鍵退出...
pause >nul

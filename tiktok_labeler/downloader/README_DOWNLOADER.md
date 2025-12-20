# TikTok 下載器使用指南

## 📥 功能概述

自動從 Excel A 批量下載 TikTok 視頻並分類到對應文件夾：
- **real** → `tiktok tinder videos/real/`
- **ai** → `tiktok tinder videos/ai/`
- **uncertain** → `tiktok tinder videos/not sure/`
- **exclude** → `tiktok tinder videos/movies/`

---

## 🚀 快速開始

### 基本用法

```bash
cd tiktok_labeler/downloader
python tiktok_downloader_classified.py
```

### 指定 Excel A 路徑

```bash
python tiktok_downloader_classified.py --excel-a "path/to/excel_a.xlsx"
```

### 調整並行下載數

```bash
python tiktok_downloader_classified.py --workers 4
```

---

## ⚠️ 常見問題與解決方案

### 問題 1: IP 被封鎖

**錯誤信息**:
```
ERROR: [TikTok] Your IP address is blocked from accessing this post
```

**原因**: TikTok 檢測到機器人行為並封鎖了你的 IP。

**解決方案**:

#### 方案 A: 使用瀏覽器 Cookies（推薦）

1. **安裝瀏覽器擴展**（如果還沒有）:
   - Chrome: 安裝 "Get cookies.txt LOCALLY"
   - Firefox: 安裝 "cookies.txt"

2. **導出 cookies**:
   - 訪問 tiktok.com 並登錄
   - 點擊擴展圖標，導出 cookies.txt
   - 保存到安全位置，例如: `C:\cookies\tiktok_cookies.txt`

3. **配置環境變量**:

   **Windows (CMD)**:
   ```cmd
   set YTDLP_COOKIES_FROM_BROWSER=chrome
   ```

   **Windows (PowerShell)**:
   ```powershell
   $env:YTDLP_COOKIES_FROM_BROWSER="chrome"
   ```

   **Linux/Mac**:
   ```bash
   export YTDLP_COOKIES_FROM_BROWSER=chrome
   ```

   支持的瀏覽器: `chrome`, `firefox`, `edge`, `safari`, `opera`, `brave`

4. **運行下載器**:
   ```bash
   python tiktok_downloader_classified.py
   ```

#### 方案 B: 使用代理

1. **配置代理環境變量**:

   **Windows (CMD)**:
   ```cmd
   set YTDLP_PROXY=socks5://127.0.0.1:1080
   ```

   **Windows (PowerShell)**:
   ```powershell
   $env:YTDLP_PROXY="socks5://127.0.0.1:1080"
   ```

   **Linux/Mac**:
   ```bash
   export YTDLP_PROXY=socks5://127.0.0.1:1080
   ```

   支持的代理格式:
   - HTTP: `http://proxy.example.com:8080`
   - HTTPS: `https://proxy.example.com:8443`
   - SOCKS5: `socks5://127.0.0.1:1080`

2. **運行下載器**:
   ```bash
   python tiktok_downloader_classified.py
   ```

#### 方案 C: 同時使用 Cookies + 代理

```bash
# Windows
set YTDLP_COOKIES_FROM_BROWSER=chrome
set YTDLP_PROXY=socks5://127.0.0.1:1080
python tiktok_downloader_classified.py

# Linux/Mac
export YTDLP_COOKIES_FROM_BROWSER=chrome
export YTDLP_PROXY=socks5://127.0.0.1:1080
python tiktok_downloader_classified.py
```

---

### 問題 2: 視頻私密或不可用

**錯誤信息**:
```
視頻私密或不可用
```

**原因**: 視頻已被刪除、設為私密或僅限特定地區訪問。

**解決方案**:
- 檢查 URL 是否正確
- 確認視頻是否仍然存在
- 使用不同地區的代理

---

### 問題 3: Excel A 格式錯誤

**錯誤信息**:
```
❌ Excel A 不存在
```

**解決方案**:

確保 Excel A 包含以下列：

| 必需列 | 說明 | 示例 |
|--------|------|------|
| 影片網址 | TikTok URL | https://www.tiktok.com/@user/video/123456 |
| 判定結果 | 分類標籤 | real / ai / uncertain / exclude |
| 視頻ID | 視頻唯一ID | 123456789 |
| 作者 | 作者用戶名 | @username |

**Excel A 路徑**: `tiktok_labeler/tiktok tinder videos/data/excel_a_labels_raw.xlsx`

---

## 📊 輸出結果

### 成功下載

視頻會自動保存到對應文件夾：

```
tiktok tinder videos/
├── real/
│   └── real_123456789.mp4
├── ai/
│   └── ai_987654321.mp4
├── not sure/
│   └── uncertain_555555555.mp4
└── movies/
    └── exclude_111111111.mp4
```

### 下載報告

```
================================================================================
下載完成:
  ✅ 成功: 45
  ❌ 失敗: 5
  分類統計:
    - Real: 20
    - AI: 15
    - Uncertain: 8
    - Movies: 2
  失敗列表: 123456789, 987654321...
================================================================================
```

---

## 🔧 進階配置

### 自定義重試次數

修改 `tiktok_downloader_classified.py`:

```python
downloader = TikTokDownloaderClassified(
    max_workers=8,
    retry_times=5  # 默認為 3
)
```

### 自定義超時時間

修改 line 345:

```python
timeout=300  # 默認為 180 秒 (3 分鐘)
```

---

## 💡 最佳實踐

### 1. 避免 IP 封鎖

- **使用登錄狀態的瀏覽器 cookies** (最有效)
- **限制並行下載數**: `--workers 2` (避免過於激進)
- **添加延遲**: 修改 line 380 的 `time.sleep(3)` 增加到 5-10 秒

### 2. 提高成功率

- **使用穩定的網絡連接**
- **配置可靠的代理** (如果在受限地區)
- **定期更新 yt-dlp**: `pip install -U yt-dlp`

### 3. 批量下載大量視頻

```bash
# 分批下載，每批 50 個
python tiktok_downloader_classified.py --workers 2
```

---

## 🛠️ 依賴要求

```bash
pip install yt-dlp pandas openpyxl
```

### 檢查 yt-dlp 版本

```bash
python -m yt_dlp --version
```

建議版本: **2024.12.08** 或更新

---

## 📝 故障排除檢查清單

- [ ] yt-dlp 已安裝且為最新版本
- [ ] Excel A 路徑正確且文件存在
- [ ] Excel A 包含必需的列（影片網址、判定結果、視頻ID）
- [ ] TikTok URL 格式正確
- [ ] 網絡連接正常
- [ ] 已配置瀏覽器 cookies 或代理（如果遇到 IP 封鎖）
- [ ] 目標文件夾有寫入權限

---

## 🆘 獲取幫助

如果問題仍未解決：

1. 查看完整錯誤日誌
2. 檢查 yt-dlp 是否支持該 URL: `python -m yt_dlp [URL] --verbose`
3. 查看 TikTok 下載限制說明: https://github.com/yt-dlp/yt-dlp#tiktok

---

**最後更新**: 2025-12-17

**設計原則**: 第一性原理 × 增量下載 × 自動分類

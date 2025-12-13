# TSAR-RAPTOR Layer 2: AI主導自動化系統

## 🎯 概述

Layer 2 是完全自動化的AI檢測系統，能夠：
1. **批量下載** 2000個TikTok視頻
2. **AI自動檢測** 分類為 real/ai/not sure/電影動畫
3. **生成 Excel D** 記錄分類結果和15+特徵
4. **自動分類文件** 移動到對應文件夾
5. **Tinder復審** 人工復審不確定視頻
6. **持續優化** 自我學習，準確率提升至99%

---

## 📂 文件結構

```
tiktok_labeler/
├── mass_downloader/              ← 批量下載器
│   ├── url_scraper.py           ← URL抓取/生成
│   └── mass_downloader.py       ← 批量下載（並行）
├── ai_classifier/                ← AI檢測分類
│   ├── ai_detector.py           ← AI檢測器（整合現有系統）
│   └── excel_d_generator.py     ← Excel D生成器
├── file_organizer/               ← 文件自動分類
│   └── auto_classifier.py       ← 自動移動文件
├── local_reviewer/               ← 本地復審（擴展版）
│   └── review_interface.py      ← Tinder復審 + 文件移動 + Excel D更新
└── pipeline/                     ← 流水線整合
    └── layer2_pipeline.py       ← Layer 2 完整流水線

輸出文件夾結構:
C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok videos download\
├── data/
│   ├── excel_d_detection_results.xlsx  ← Excel D（AI檢測結果+特徵）
│   └── layer2_review_results.csv       ← 人工復審結果
├── real/                                ← 真實視頻
│   └── real_7123456789.mp4
├── ai/                                  ← AI生成視頻
│   └── ai_7234567890.mp4
├── not sure/                            ← 不確定視頻（待復審）
│   └── not_sure_7345678901.mp4
└── 電影動畫/                             ← 電影/動畫視頻
    └── movie_7456789012.mp4
```

---

## 🚀 快速開始

### 前置準備

確保已安裝依賴：
```bash
pip install flask flask-cors pandas openpyxl yt-dlp opencv-python scipy
```

### Step 1: 準備URL列表

有三種方式獲取TikTok URL：

#### 方式A: 從文本文件加載
創建 `url_list.txt`，每行一個URL：
```
https://www.tiktok.com/@user/video/7123456789
https://www.tiktok.com/@user2/video/7234567890
...
```

#### 方式B: 從 Excel A 導出（避免重複）
```bash
cd tiktok_labeler/mass_downloader
python url_scraper.py \
  --excel-a "../../data/tiktok_labels/excel_a_labels_raw.xlsx" \
  --output url_list.txt
```

#### 方式C: 生成測試URL（測試用）
```bash
python url_scraper.py \
  --generate-random 100 \
  --output url_list.txt
```

### Step 2: 運行完整 Layer 2 流水線

```bash
cd tiktok_labeler/pipeline
python layer2_pipeline.py \
  --url-list ../mass_downloader/url_list.txt \
  --download-dir "../../tiktok videos download" \
  --target 2000
```

這會自動執行：
1. ✅ 批量下載 2000 個視頻
2. ✅ AI檢測分類
3. ✅ 生成 Excel D
4. ✅ 自動移動文件到分類文件夾
5. ✅ 提示人工復審不確定視頻

### Step 3: 人工復審不確定視頻

流水線會自動進入Tinder復審模式：
```
================================================================================
Tinder式復審 - 視頻 1/50
================================================================================
📹 視頻: not_sure_7345678901.mp4
🤖 AI預測: 45.2% (不確定)

操作指南:
  ← (l) - Real（真實）
  → (r) - AI（生成）
  ↓ (m) - Movie/Anime（電影/動畫）
  s - Skip（跳過）
  p - Play（播放視頻）
  t - Thumbnail（顯示縮略圖）
  q - Quit（退出）
────────────────────────────────────────────────────────────────────────────────

你的判斷 (l/r/m/s/p/t/q): _
```

復審後會自動：
- ✅ 移動視頻到正確文件夾（real/ai/電影動畫）
- ✅ 更新 Excel D 人工復審結果

---

## 🎮 分步執行（高級用法）

### 僅下載視頻
```bash
cd mass_downloader
python mass_downloader.py \
  --url-list url_list.txt \
  --output "../../tiktok videos download" \
  --workers 8 \
  --target 2000
```

### 僅AI檢測
```bash
cd ai_classifier
python ai_detector.py \
  --video-dir "../../tiktok videos download" \
  --workers 4
```

### 僅生成 Excel D
```bash
cd ai_classifier
python excel_d_generator.py \
  --video-dir "../../tiktok videos download" \
  --output "../../tiktok videos download/data/excel_d_detection_results.xlsx"
```

### 僅文件分類
```bash
cd file_organizer
python auto_classifier.py \
  --source-dir "../../tiktok videos download" \
  --excel-d "../../tiktok videos download/data/excel_d_detection_results.xlsx"
```

### 僅人工復審
```bash
cd local_reviewer
python review_interface.py \
  --videos "../../tiktok videos download/not sure/*.mp4"
```

---

## 📊 Excel D 格式說明

Excel D 包含以下列：

### 基本信息
- **序號**: 自動編號
- **影片網址**: TikTok URL
- **AI檢測分類**: REAL / AI / NOT_SURE / 電影動畫
- **信心度**: 0-100
- **視頻ID**: TikTok視頻ID
- **檔案路徑**: 本地文件路徑
- **分析時間**: 檢測時間戳

### 關鍵特徵（自我訓練用）
- **fps**: 幀率
- **width/height**: 分辨率
- **duration**: 時長
- **avg_brightness**: 平均亮度
- **avg_contrast**: 平均對比度
- **avg_saturation**: 平均飽和度
- **avg_blur**: 平均模糊度
- **avg_optical_flow**: 平均光流（運動強度）
- **scene_changes**: 場景變化次數
- **dct_energy**: DCT高頻能量
- **spectral_entropy**: 頻譜熵
- **audio_sample_rate**: 音頻採樣率
- **audio_channels**: 音頻聲道數
- **bitrate**: 視頻碼率

### 復審信息
- **人工復審結果**: 人工標註（初始為空）
- **復審時間**: 復審時間戳
- **備註**: 備註信息

詳細格式說明：[EXCEL_D_FORMAT.md](./EXCEL_D_FORMAT.md)

---

## 🔄 工作流程圖

```
┌──────────────────────────────────────────────────┐
│ Step 1: 準備URL列表（2000個）                    │
│   - 從文本文件                                    │
│   - 從Excel A導出                                │
│   - 生成測試URL                                   │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ Step 2: 批量下載 + AI檢測 + 生成 Excel D        │
│   - yt-dlp並行下載（8線程）                      │
│   - AI檢測分類（物理違規/頻域/面部剛性分析）     │
│   - 生成 Excel D 記錄所有判定結果和影片信息      │
│     • AI檢測分類結果 (real/ai/not sure/電影動畫) │
│     • 信心度 (0-100)                              │
│     • 提取15+特徵值                               │
│     • 影片路徑和基本信息                          │
│     • 預留人工復審欄位                            │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ Step 3: 自動文件分類                             │
│   - 根據Excel D的分類結果移動視頻                │
│   - real/ ← 真實視頻                             │
│   - ai/ ← AI生成視頻                             │
│   - not sure/ ← 不確定視頻                       │
│   - 電影動畫/ ← 電影/動畫視頻                    │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ Step 4: 準備人工復審                             │
│   - 統計 not sure 文件夾中的視頻數量             │
│   - 加載待復審視頻列表                           │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ Step 5: 人工復審 + 更新 Excel D                  │
│   - 從 "not sure" 文件夾加載視頻                 │
│   - Tinder式快速復審（← Real | → AI | ↓ Movie） │
│   - 收集肉眼判定完的數據後更新 Excel D           │
│     • 記錄人工復審結果                            │
│     • 記錄復審時間                                │
│     • 記錄備註信息                                │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ Step 6: 根據 Excel D 重新分類 not sure 影片      │
│   - 讀取 Excel D 中的人工復審結果                │
│   - 自動移動到正確的分類文件夾                   │
│     • real/ ← 復審判定為真實                     │
│     • ai/ ← 復審判定為AI生成                     │
│     • 電影動畫/ ← 復審判定為電影/動畫            │
│   - 更新 Excel D 中的檔案路徑                    │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ Step 7: 循環優化與自我學習                       │
│   - 根據Excel D完整數據分析                      │
│   - 對比AI判定 vs 人工復審的差異                 │
│   - 識別誤判模式和特徵                           │
│   - 優化AI檢測模組參數                           │
│   - 提升準確率至99%                              │
└──────────────────────────────────────────────────┘
                    ↓
          ↻ 返回 Step 1（持續學習）
```

---

## ⚙️ 配置參數

### 下載器參數
- `--url-list`: URL列表文件路徑
- `--download-dir`: 下載目錄（默認: ../../tiktok videos download）
- `--target`: 目標下載數量（默認: 2000）
- `--download-workers`: 並行下載數（默認: 8）

### 檢測器參數
- `--detect-workers`: 並行檢測數（默認: 4）

### 流程控制參數
- `--skip-download`: 跳過下載步驟
- `--skip-detection`: 跳過AI檢測步驟
- `--skip-classification`: 跳過文件分類步驟
- `--skip-review`: 跳過人工復審步驟

---

## 📈 性能優化

### 下載速度優化
- 調整 `--download-workers` 參數（建議: CPU核心數的1-2倍）
- 使用更快的網絡連接
- 確保 yt-dlp 為最新版本

### 檢測速度優化
- 調整 `--detect-workers` 參數（建議: GPU數量或CPU核心數）
- 減少採樣幀數（在 ai_detector.py 中配置）
- 使用GPU加速（如果有CUDA支持）

### 預期性能（AMD Ryzen 9 9950X + RTX 5090）
- 下載: ~100-150 視頻/小時（8並行）
- 檢測: ~200-300 視頻/小時（4並行）
- 總流程: ~2000 視頻約需 12-15小時

---

## 🛠️ 故障排除

### 問題1: 下載失敗率高
**解決**:
- 檢查網絡連接
- 更新 yt-dlp: `pip install --upgrade yt-dlp`
- 降低並行數: `--download-workers 4`
- 檢查URL列表是否有效

### 問題2: AI檢測卡住
**解決**:
- 降低並行數: `--detect-workers 2`
- 檢查視頻文件是否損壞
- 查看錯誤日誌

### 問題3: Excel D 生成失敗
**解決**:
- 確保有足夠的磁盤空間
- 檢查 openpyxl 是否安裝: `pip install openpyxl`
- 查看特徵提取是否成功

### 問題4: 文件移動失敗
**解決**:
- 檢查文件夾權限
- 確保目標文件夾存在
- 檢查磁盤空間

---

## 📊 統計信息示例

```
================================================================================
Layer 2 流水線執行完畢！
================================================================================
  • 下載視頻: 1950 成功
  • AI檢測: 1950 個視頻
    - REAL: 850 (43.6%)
    - AI: 750 (38.5%)
    - NOT_SURE: 300 (15.4%)
    - 電影動畫: 50 (2.6%)
  • 文件分類: 1950 個已移動
  • 人工復審: 280 個已復審
================================================================================
```

---

## 🔗 相關文檔

- [QUICKSTART.md](./QUICKSTART.md) - Layer 1 快速開始
- [EXCEL_A_FORMAT.md](./EXCEL_A_FORMAT.md) - Excel A 格式說明
- [EXCEL_D_FORMAT.md](./EXCEL_D_FORMAT.md) - Excel D 格式說明
- [README.md](./README.md) - 完整系統文檔

---

## 🎯 下一步

完成 Layer 2 後，可以：
1. **持續訓練**: 使用 Excel D 的特徵數據優化AI模組
2. **擴展數據集**: 繼續下載更多視頻，建立海量數據庫
3. **提升準確率**: 根據人工復審結果調整檢測閾值
4. **自動化循環**: 設置定時任務，完全自動化運行

目標：**99% 準確率 + 海量數據 + 零人工干預**

---

**設計哲學**: 沙皇炸彈（海量數據） × 猛禽3（極速自動化） × 第一性原理（物理不可偽造）

**最後更新**: 2025-12-12

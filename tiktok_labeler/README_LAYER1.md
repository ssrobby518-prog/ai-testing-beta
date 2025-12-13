# TSAR-RAPTOR Layer 1: 人工主導標註系統

## 🎯 系統概述

Layer 1 是人工主導的Tinder式標註系統，用於快速建立高質量訓練數據。

### 核心功能
1. **Chrome Tinder式標註** - 左右上下鍵快速標註
2. **實時存儲** - 標註即時保存到 Excel A
3. **自動下載分類** - 根據標籤自動下載到對應文件夾
4. **特徵提取** - 15+特徵提取到 Excel B
5. **大數據分析** - 統計分析生成 Excel C
6. **模組優化** - 自動優化AI檢測模組

---

## 📂 文件結構（新路徑）

```
C:\Users\s_robby518\Documents\trae_projects\ai testing\
└── tiktok_labeler\
    ├── tiktok tinder videos\           ← Layer 1 基礎目錄
    │   ├── data\
    │   │   ├── excel a\                ← Excel A（人工標註原始數據）
    │   │   ├── excel b\                ← Excel B（視頻特徵統計）
    │   │   └── excel c\                ← Excel C（AI vs 真實差異分析）
    │   ├── real\                       ← 真實視頻文件夾
    │   │   └── real_7123456789.mp4
    │   ├── ai\                         ← AI生成視頻文件夾
    │   │   └── ai_7234567890.mp4
    │   ├── not sure\                   ← 不確定視頻文件夾
    │   │   └── uncertain_7345678901.mp4
    │   └── movies\                     ← 電影/動畫視頻文件夾
    │       └── exclude_7456789012.mp4
    │
    ├── chrome_extension\               ← Chrome擴展
    │   ├── manifest.json
    │   ├── content.js
    │   └── styles.css
    ├── backend\                        ← Flask後端
    │   └── server.py
    ├── downloader\                     ← 下載器
    │   └── tiktok_downloader_classified.py
    ├── analyzer\                       ← 分析器
    │   ├── feature_extractor_layer1.py
    │   └── big_data_analyzer.py
    ├── auto_reconstructor\             ← 模組優化
    │   └── module_optimizer.py
    ├── pipeline\                       ← 流水線
    │   └── layer1_pipeline.py
    └── config.py                       ← 路徑配置文件
```

---

## 🚀 快速開始

### Step 1: 啟動後端服務器

```bash
cd tiktok_labeler/backend
python server.py
```

輸出:
```
🚀 TSAR-RAPTOR TikTok Labeler Backend Server
================================================================================
📊 Excel A 路徑: C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok tinder videos\data\excel a
🌐 API 端點: http://127.0.0.1:5000/api/label
================================================================================
🟢 服務器啟動中...
```

**保持這個終端運行！**

### Step 2: 安裝 Chrome 擴展

1. 打開 Chrome 瀏覽器
2. 訪問 `chrome://extensions/`
3. 右上角開啟「**開發者模式**」
4. 點擊「**加載已解壓的擴展程序**」
5. 選擇文件夾: `tiktok_labeler/chrome_extension`

### Step 3: 開始 Tinder 式標註

1. 訪問 https://www.tiktok.com/foryou
2. 等待3秒，會出現歡迎橫幅
3. 使用鍵盤快速標註：

   | 按鍵 | 判定 | 說明 | 存儲位置 |
   |------|------|------|----------|
   | **← 左箭頭** | Real | 真實視頻 | `tiktok tinder videos/real/` |
   | **→ 右箭頭** | AI | AI生成視頻 | `tiktok tinder videos/ai/` |
   | **↑ 上箭頭** | Uncertain | 不確定 | `tiktok tinder videos/not sure/` |
   | **↓ 下箭頭** | Movie/Anime | 電影/動畫（排除） | `tiktok tinder videos/movies/` |

4. 每標註一個視頻：
   - ✅ 網址自動記錄到 Excel A
   - ✅ 螢幕中央顯示大emoji反饋
   - ✅ 自動滾動到下一個視頻
   - ✅ 右上角統計面板實時更新

**目標：標註至少 100-200 個視頻**

### Step 4: 運行完整 Layer 1 流水線

標註完成後，一鍵執行完整流水線：

```bash
cd tiktok_labeler/pipeline
python layer1_pipeline.py
```

自動執行：
1. ✅ 從 Excel A 批量下載視頻
2. ✅ **自動分類到對應文件夾（real/ai/not sure/movies）**
3. ✅ 提取15+特徵 → Excel B
4. ✅ 大數據統計分析 → Excel C
5. ✅ 自動優化AI檢測模組

---

## 📊 數據文件說明

### Excel A（人工標註原始數據）
- **路徑**: `C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok tinder videos\data\excel a`
- **格式**: 中文列名
- **列**: 序號, 影片網址, 判定結果, 標註時間, 視頻ID, 作者, 標題, 點贊數, 來源, 版本
- **判定結果**: REAL / AI / UNCERTAIN / EXCLUDE（大寫）

### Excel B（視頻特徵統計）
- **路徑**: `C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok tinder videos\data\excel b`
- **格式**: 15+特徵列 + 標籤列
- **特徵**: fps, width, height, duration, avg_brightness, avg_contrast, avg_saturation, avg_blur, avg_optical_flow, scene_changes, dct_energy, spectral_entropy, audio_sample_rate, audio_channels, bitrate
- **額外列**: label（英文）, label_cn（中文）

### Excel C（AI vs 真實差異分析）
- **路徑**: `C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok tinder videos\data\excel c`
- **格式**: 多個工作表
  - Feature_Ranking - 特徵排名
  - Descriptive_Stats - 描述性統計
  - Significance_Testing - 顯著性檢驗
  - Insights - 優化建議

---

## 🎮 分步執行（高級用法）

### 僅下載並分類視頻
```bash
cd downloader
python tiktok_downloader_classified.py
```

輸出:
```
下載完成:
  ✅ 成功: 95
  ❌ 失敗: 5
  分類統計:
    - Real: 45
    - AI: 35
    - Uncertain: 10
    - Movies: 5
```

### 僅提取特徵
```bash
cd analyzer
python feature_extractor_layer1.py
```

### 僅大數據分析
```bash
cd analyzer
python big_data_analyzer.py \
  --excel-b "C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok tinder videos\data\excel b" \
  --output "C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok tinder videos\data\excel c"
```

### 僅模組優化
```bash
cd auto_reconstructor
python module_optimizer.py \
  --excel-c "C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok tinder videos\data\excel c"
```

---

## 🔄 完整工作流程

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: Chrome Tinder式標註                              │
│   在TikTok上使用鍵盤快速標註（← → ↑ ↓）                 │
│   目標: 100-200個視頻                                    │
└─────────────────────────────────────────────────────────┘
                    ↓ 自動存儲
┌─────────────────────────────────────────────────────────┐
│ Excel A: 人工標註原始數據                                │
│   序號, 影片網址, 判定結果, 標註時間, ...               │
└─────────────────────────────────────────────────────────┘
                    ↓ 批量下載
┌─────────────────────────────────────────────────────────┐
│ Step 2: 自動下載並分類到文件夾                           │
│   • Real → real/                                        │
│   • AI → ai/                                            │
│   • Uncertain → not sure/                               │
│   • Movies → movies/                                    │
└─────────────────────────────────────────────────────────┘
                    ↓ 特徵提取
┌─────────────────────────────────────────────────────────┐
│ Excel B: 視頻特徵統計                                    │
│   15+特徵 + 標籤                                         │
└─────────────────────────────────────────────────────────┘
                    ↓ 統計分析
┌─────────────────────────────────────────────────────────┐
│ Excel C: AI vs 真實差異分析                              │
│   • 特徵排名（Cohen's d）                                │
│   • 顯著性檢驗（t-test）                                 │
│   • 優化建議                                             │
└─────────────────────────────────────────────────────────┘
                    ↓ 模組優化
┌─────────────────────────────────────────────────────────┐
│ 輸出: 優化的AI檢測模組配置                               │
│   • 特徵權重調整                                         │
│   • 閾值優化                                             │
└─────────────────────────────────────────────────────────┘
```

---

## ⚙️ 配置文件

所有路徑配置都在 `config.py` 中：

```python
# Layer 1 路徑配置
LAYER1_BASE_DIR = Path(r"C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok tinder videos")

# Excel 文件路徑
EXCEL_A_PATH = LAYER1_DATA_DIR / "excel a"
EXCEL_B_PATH = LAYER1_DATA_DIR / "excel b"
EXCEL_C_PATH = LAYER1_DATA_DIR / "excel c"

# 視頻分類文件夾
LAYER1_VIDEO_FOLDERS = {
    'real': LAYER1_BASE_DIR / 'real',
    'ai': LAYER1_BASE_DIR / 'ai',
    'uncertain': LAYER1_BASE_DIR / 'not sure',
    'exclude': LAYER1_BASE_DIR / 'movies'
}
```

---

## 📈 預期成果

### 第1天
- ✅ 標註 100 個視頻
- ✅ Excel A 生成
- ✅ 視頻自動分類到文件夾
- ✅ Excel B/C 生成
- ✅ 初步優化建議

### 第7天（累積）
- ✅ 標註 700 個視頻
- ✅ 精確的統計分析
- ✅ 顯著的特徵區分能力
- ✅ AI檢測準確率提升 5-10%

---

## 🛠️ 故障排除

### 問題1: Chrome擴展無反應
**解決**:
```bash
# 檢查後端是否運行
curl http://127.0.0.1:5000/api/stats

# 重新加載擴展
chrome://extensions/ → 點擊刷新按鈕
```

### 問題2: 視頻下載到錯誤文件夾
**解決**:
```bash
# 檢查路徑配置
cd pipeline
python layer1_pipeline.py --check-paths
```

### 問題3: Excel 文件路徑錯誤
**解決**:
所有路徑都在 `config.py` 中配置，確保路徑正確。

---

## 💡 使用技巧

### 快速標註
- **連續Real**: 左左左左左（快速判定明顯真實視頻）
- **連續AI**: 右右右右右（快速判定明顯AI視頻）
- **跳過廣告**: 下下下（快速跳過電影/動畫）

### 提升效率
- **目標**: 每天標註 100-200 個視頻（約 10-20 分鐘）
- **策略**: 憑第一直覺快速判斷，不要過度思考
- **休息**: 每標註 50 個休息一下

### 數據質量
- **平衡**: 保持 Real 和 AI 樣本大致平衡
- **多樣性**: 標註不同類型的視頻（舞蹈、美妝、搞笑...）
- **清晰**: 電影/動畫一定要排除，避免污染數據

---

## 🔗 相關文檔

- [LAYER2_README.md](./LAYER2_README.md) - Layer 2 AI主導自動化
- [EXCEL_A_FORMAT.md](./EXCEL_A_FORMAT.md) - Excel A 格式說明
- [SYSTEM_OVERVIEW.md](./SYSTEM_OVERVIEW.md) - 完整系統總覽

---

**設計哲學**: 第一性原理 × 沙皇炸彈 × 猛禽3

**目標**: 快速建立高質量訓練數據 → 優化AI檢測模組 → 99%準確率

**最後更新**: 2025-12-12

# TSAR-RAPTOR Layer 1: 人工主導標註系統

## 🎯 系統概述

Layer 1 是人工主導的Tinder式標註系統，用於快速建立高質量訓練數據。

### 核心功能
1. **Chrome Tinder式標註** - 左右上下鍵快速標註
2. **智能URL解析** - 自動將短網址轉換為真實網址
3. **實時存儲** - 標註即時保存到 Excel A
4. **防IP封鎖下載** - 自動分類下載到對應文件夾
5. **下載狀態追蹤** - Excel A/B/C 自動排除未下載影片
6. **特徵提取** - 15+特徵提取到 Excel B
7. **大數據分析** - 統計分析生成 Excel C
8. **模組優化** - 自動優化AI檢測模組

---

## 📂 文件結構（新路徑）

```
C:\Users\s_robby518\Documents\trae_projects\ai testing\
└── tiktok_labeler\
    ├── tiktok tinder videos\           ← Layer 1 基礎目錄
    │   ├── data\
    │   │   ├── excel_a_labels_raw.xlsx ← Excel A（人工標註原始數據）
    │   │   ├── excel_b_features.xlsx   ← Excel B（視頻特徵統計）
    │   │   └── excel_c_analysis.xlsx   ← Excel C（AI vs 真實差異分析）
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
    │   ├── content_test.js
    │   └── styles.css
    ├── backend\                        ← Flask後端
    │   └── server.py
    ├── downloader\                     ← 下載器
    │   ├── tiktok_downloader_classified.py
    │   └── run_with_cookies.py
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
📊 Excel A 路徑: ...\tiktok tinder videos\data\excel_a_labels_raw.xlsx
🌐 API 端點: http://127.0.0.1:5000/api/label
================================================================================
🟢 服務器啟動中...
✅ 已添加「下載狀態」列到 Excel A
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
2. 看到彈窗提示：「🚀 TSAR-RAPTOR 標註系統已啟動！」
3. 使用鍵盤快速標註：

   | 按鍵 | 判定 | 說明 | 存儲位置 |
   |------|------|------|----------|
   | **← 左箭頭** | Real | 真實視頻 | `tiktok tinder videos/real/` |
   | **→ 右箭頭** | AI | AI生成視頻 | `tiktok tinder videos/ai/` |
   | **↑ 上箭頭** | Uncertain | 不確定 | `tiktok tinder videos/not sure/` |
   | **↓ 下箭頭** | Movie/Anime | 電影/動畫（排除） | `tiktok tinder videos/movies/` |

4. 每標註一個視頻：
   - ✅ **短網址自動解析為真實網址**（vm.tiktok.com → www.tiktok.com/@user/video/xxx）
   - ✅ 網址自動記錄到 Excel A（下載狀態: 未下載）
   - ✅ 螢幕中央顯示大emoji反饋
   - ✅ 後端控制台顯示標註成功

**目標：標註至少 100-200 個視頻**

### Step 4: 運行完整 Layer 1 流水線

標註完成後，一鍵執行完整流水線：

```bash
cd tiktok_labeler/pipeline
python layer1_pipeline.py
```

或使用快捷方式：
```bash
python 執行第一層下載.py
```

自動執行：
1. ✅ 從 Excel A 批量下載視頻（**智能防IP封鎖**）
2. ✅ 自動更新 Excel A 下載狀態（已下載/下載失敗）
3. ✅ **自動分類到對應文件夾（real/ai/not sure/movies）**
4. ✅ 提取15+特徵 → Excel B（**自動跳過未下載影片**）
5. ✅ 大數據統計分析 → Excel C（**只分析已下載影片**）
6. ✅ 自動優化AI檢測模組

---

## 🛡️ 智能防IP封鎖系統（第一性原理）

### 問題根源
TikTok 對高頻請求和特定視頻實施IP封鎖，導致下載失敗。

### 解決方案（三層防護）

#### 1. URL層防護
- **短網址自動解析**: vm.tiktok.com → www.tiktok.com/@user/video/xxx
- **真實URL存儲**: Excel A 只記錄完整可下載URL
- **去重機制**: 避免重複下載同一視頻

#### 2. 網絡層防護
- **自定義User-Agent**: 模擬真實瀏覽器訪問
- **請求延遲**: 每次請求間隔2秒（--sleep-requests 2）
- **下載片段延遲**: 片段間隔1-3秒（--sleep-interval 1）
- **單線程下載**: max_workers=1，避免並行觸發封鎖

#### 3. 重試層防護
- **漸進式重試**: 失敗後等待 5s → 8s → 11s 後重試
- **重試次數**: 最多3次
- **錯誤分類**:
  - IP封鎖 → 標註「下載失敗: IP被封鎖」
  - 私密視頻 → 標註「下載失敗: 視頻私密或不可用」

### 下載狀態追蹤

Excel A 新增「下載狀態」欄位：
- **未下載**: 初始狀態，等待下載
- **已下載**: 下載成功，視頻已存儲
- **下載失敗: 原因**: 下載失敗，註明具體原因

**關鍵保證**: Excel B/C 自動跳過「未下載」和「下載失敗」的影片，確保分析數據完整性。

---

## 📊 數據文件說明

### Excel A（人工標註原始數據）
- **路徑**: `tiktok_labeler/tiktok tinder videos/data/excel_a_labels_raw.xlsx`
- **格式**: 中文列名
- **列**:
  - 序號, 影片網址（真實URL）, 判定結果, **下載狀態**, 標註時間
  - 視頻ID, 作者, 標題, 點贊數, 來源, 版本
- **判定結果**: REAL / AI / UNCERTAIN / EXCLUDE（大寫）
- **下載狀態**: 未下載 / 已下載 / 下載失敗: 原因

### Excel B（視頻特徵統計）
- **路徑**: `tiktok_labeler/tiktok tinder videos/data/excel_b_features.xlsx`
- **格式**: 15+特徵列 + 標籤列
- **特徵**: fps, width, height, duration, avg_brightness, avg_contrast, avg_saturation, avg_blur, avg_optical_flow, scene_changes, dct_energy, spectral_entropy, audio_sample_rate, audio_channels, bitrate
- **額外列**: label（英文）, label_cn（中文）
- **自動過濾**: 只包含「已下載」的影片特徵

### Excel C（AI vs 真實差異分析）
- **路徑**: `tiktok_labeler/tiktok tinder videos/data/excel_c_analysis.xlsx`
- **格式**: 多個工作表
  - Feature_Ranking - 特徵排名
  - Descriptive_Stats - 描述性統計
  - Significance_Testing - 顯著性檢驗
  - Insights - 優化建議
- **數據來源**: 僅基於 Excel B（已下載影片）

---

## 🎮 分步執行（高級用法）

### 僅下載並分類視頻
```bash
cd downloader
python run_with_cookies.py
```

輸出:
```
================================================================================
TikTok 下載器
================================================================================

下載完成:
  ✅ 成功: 2
  ❌ 失敗: 1
  分類統計:
    - Real: 2
    - AI: 0
    - Uncertain: 0
    - Movies: 0
  失敗列表: 6897974718560423173...
✅ 已更新 Excel A 下載狀態
```

### 僅提取特徵
```bash
cd analyzer
python feature_extractor_layer1.py
```

### 僅大數據分析
```bash
cd analyzer
python big_data_analyzer.py --excel-b "..." --output "..."
```

### 僅模組優化
```bash
cd auto_reconstructor
python module_optimizer.py --excel-c "..."
```

---

## 🔄 完整工作流程

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: Chrome Tinder式標註                              │
│   在TikTok上使用鍵盤快速標註（← → ↑ ↓）                 │
│   ✅ 短網址自動解析為真實URL                             │
│   目標: 100-200個視頻                                    │
└─────────────────────────────────────────────────────────┘
                    ↓ 自動存儲 + URL解析
┌─────────────────────────────────────────────────────────┐
│ Excel A: 人工標註原始數據                                │
│   序號, 影片網址（真實URL）, 判定結果, 下載狀態, ...    │
│   下載狀態: 未下載（初始）                               │
└─────────────────────────────────────────────────────────┘
                    ↓ 智能防IP封鎖下載
┌─────────────────────────────────────────────────────────┐
│ Step 2: 批量下載並更新狀態                               │
│   • 自定義User-Agent + 請求延遲                          │
│   • 漸進式重試（5s → 8s → 11s）                         │
│   • 成功 → 下載狀態: 已下載                             │
│   • 失敗 → 下載狀態: 下載失敗: 原因                     │
└─────────────────────────────────────────────────────────┘
                    ↓ 自動分類到文件夾
┌─────────────────────────────────────────────────────────┐
│ 視頻文件自動分類                                         │
│   • Real → real/                                        │
│   • AI → ai/                                            │
│   • Uncertain → not sure/                               │
│   • Movies → movies/                                    │
└─────────────────────────────────────────────────────────┘
                    ↓ 特徵提取（跳過未下載）
┌─────────────────────────────────────────────────────────┐
│ Excel B: 視頻特徵統計                                    │
│   15+特徵 + 標籤（僅包含已下載影片）                     │
└─────────────────────────────────────────────────────────┘
                    ↓ 統計分析（跳過未下載）
┌─────────────────────────────────────────────────────────┐
│ Excel C: AI vs 真實差異分析                              │
│   • 特徵排名（Cohen's d）                                │
│   • 顯著性檢驗（t-test）                                 │
│   • 優化建議（基於已下載影片）                           │
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
EXCEL_A_PATH = LAYER1_DATA_DIR / "excel_a_labels_raw.xlsx"
EXCEL_B_PATH = LAYER1_DATA_DIR / "excel_b_features.xlsx"
EXCEL_C_PATH = LAYER1_DATA_DIR / "excel_c_analysis.xlsx"

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
- ✅ Excel A 生成（真實URL + 下載狀態）
- ✅ 視頻自動分類到文件夾（IP封鎖率 < 30%）
- ✅ Excel B/C 生成（僅基於成功下載）
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

# 檢查控制台
F12 → Console 查看錯誤
```

### 問題2: Excel A 記錄短網址
**解決**:
- ✅ 已自動解析！後端會將 `vm.tiktok.com/xxx` 自動轉換為 `www.tiktok.com/@user/video/xxx`
- 如果發現短網址，檢查後端日誌：`🔗 解析短網址: ...`

### 問題3: 視頻下載失敗（IP封鎖）
**解決**:
```bash
# 查看 Excel A 下載狀態列
# 失敗原因:
# - "IP被封鎖" → TikTok平台限制，無法解決，跳過該視頻
# - "視頻私密或不可用" → 視頻已刪除或設為私密

# 下載成功率預期: 70-80%（部分視頻確實會被封鎖）
```

### 問題4: Excel B/C 數據缺失
**原因**:
- Excel B/C 只包含「已下載」的影片
- 檢查 Excel A「下載狀態」列，確認有足夠的「已下載」樣本

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
- **容錯**: 部分視頻下載失敗是正常的（IP封鎖），系統會自動跳過

---

## 🔗 相關文檔

- [LAYER2_README.md](./LAYER2_README.md) - Layer 2 AI主導自動化
- [EXCEL_A_FORMAT.md](./EXCEL_A_FORMAT.md) - Excel A 格式說明
- [SYSTEM_OVERVIEW.md](./SYSTEM_OVERVIEW.md) - 完整系統總覽

---

**設計哲學**: 第一性原理 × 沙皇炸彈 × 猛禽3

**核心創新**:
- **智能URL解析**: 短網址 → 真實URL（100%成功率）
- **三層防IP封鎖**: User-Agent + 延遲 + 漸進重試（70-80%成功率）
- **狀態追蹤閉環**: Excel A/B/C 自動排除未下載影片（數據完整性100%）

**目標**: 快速建立高質量訓練數據 → 優化AI檢測模組 → 99%準確率

**最後更新**: 2025-12-19

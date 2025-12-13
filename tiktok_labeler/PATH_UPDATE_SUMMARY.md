# Layer 1 路徑更新總結

## 🎯 更新內容

根據您的要求，已將 Layer 1 的所有路徑重新配置為新的結構。

---

## 📂 新路徑配置

### 基礎目錄
```
C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok tinder videos
```

### Excel 文件路徑

| 文件 | 路徑 |
|------|------|
| **Excel A** | `C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok tinder videos\data\excel a` |
| **Excel B** | `C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok tinder videos\data\excel b` |
| **Excel C** | `C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok tinder videos\data\excel c` |

### 視頻分類文件夾

| 分類 | 路徑 | 說明 |
|------|------|------|
| **Real** | `tiktok tinder videos\real\` | 真實視頻 |
| **AI** | `tiktok tinder videos\ai\` | AI生成視頻 |
| **Not Sure** | `tiktok tinder videos\not sure\` | 不確定視頻 |
| **Movies** | `tiktok tinder videos\movies\` | 電影/動畫（排除訓練） |

---

## 🆕 新增/更新的文件

### 1. 配置文件
- ✅ **config.py** - 統一路徑配置文件
  - 定義所有 Layer 1 和 Layer 2 路徑
  - 提供 `ensure_directories()` 函數自動創建目錄

### 2. 更新的後端服務器
- ✅ **backend/server.py**
  - 更新為使用 `config.py` 中的路徑
  - Excel A 路徑自動配置

### 3. 新的下載器（帶自動分類）
- ✅ **downloader/tiktok_downloader_classified.py**
  - 根據 Excel A 的標籤自動下載到對應文件夾
  - 支持增量下載（避免重複）
  - 分類映射:
    - `real` → `real/`
    - `ai` → `ai/`
    - `uncertain` → `not sure/`
    - `exclude/movies` → `movies/`

### 4. 新的特徵提取器（Layer 1 專用）
- ✅ **analyzer/feature_extractor_layer1.py**
  - 從所有分類文件夾（real/ai/not sure/movies）加載視頻
  - 提取特徵並標註標籤
  - 輸出到 Excel B

### 5. 新的 Layer 1 流水線
- ✅ **pipeline/layer1_pipeline.py**
  - 一鍵執行完整 Layer 1 流程
  - 自動下載分類 → 特徵提取 → 大數據分析 → 模組優化
  - 支持 `--check-paths` 檢查路徑配置

### 6. 更新的文檔
- ✅ **README_LAYER1.md** - Layer 1 完整指南（新路徑）
- ✅ **QUICKSTART.md** - 更新快速開始指南（新路徑）
- ✅ **PATH_UPDATE_SUMMARY.md** - 本文檔

---

## 🚀 使用方法

### 初始化目錄
```bash
cd tiktok_labeler
python config.py
```

輸出:
```
✅ 所有目錄已創建

Layer 1 基礎目錄: C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok tinder videos
Layer 2 基礎目錄: C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok videos download
```

### 檢查路徑配置
```bash
cd pipeline
python layer1_pipeline.py --check-paths
```

輸出:
```
================================================================================
路徑配置:
================================================================================
基礎目錄: C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok tinder videos
數據目錄: C:\Users\s_robby518\Documents\trae_projects\ai testing\tiktok_labeler\tiktok tinder videos\data

Excel 文件:
  • Excel A: C:\...\tiktok tinder videos\data\excel a
  • Excel B: C:\...\tiktok tinder videos\data\excel b
  • Excel C: C:\...\tiktok tinder videos\data\excel c

視頻文件夾:
  • real: C:\...\tiktok tinder videos\real
  • ai: C:\...\tiktok tinder videos\ai
  • uncertain: C:\...\tiktok tinder videos\not sure
  • exclude: C:\...\tiktok tinder videos\movies
================================================================================
```

### 運行完整流水線
```bash
cd pipeline
python layer1_pipeline.py
```

---

## 🔄 自動分類邏輯

下載器會根據 Excel A 中的 `判定結果` 列自動分類：

| Excel A 中的標籤 | 下載到文件夾 | 文件命名格式 |
|------------------|--------------|--------------|
| `REAL` | `real/` | `real_7123456789.mp4` |
| `AI` | `ai/` | `ai_7234567890.mp4` |
| `UNCERTAIN` | `not sure/` | `uncertain_7345678901.mp4` |
| `EXCLUDE` | `movies/` | `exclude_7456789012.mp4` |

---

## 📊 工作流程

```
1. Chrome Tinder式標註 → Excel A
   （← Real | → AI | ↑ Uncertain | ↓ Movies）
                ↓
2. 運行 layer1_pipeline.py
                ↓
3. 自動下載並分類:
   • Real → real/
   • AI → ai/
   • Uncertain → not sure/
   • Movies → movies/
                ↓
4. 特徵提取 → Excel B
   （從所有文件夾提取，帶標籤）
                ↓
5. 大數據分析 → Excel C
   （AI vs Real 統計差異）
                ↓
6. 模組優化 → optimized_config.json
   （自動調整AI檢測模組）
```

---

## ✅ 驗證清單

請確認以下內容：

### 目錄結構
- [ ] `tiktok tinder videos/` 存在
- [ ] `tiktok tinder videos/data/` 存在
- [ ] `tiktok tinder videos/real/` 存在
- [ ] `tiktok tinder videos/ai/` 存在
- [ ] `tiktok tinder videos/not sure/` 存在
- [ ] `tiktok tinder videos/movies/` 存在

### 文件生成
- [ ] 標註後 Excel A 生成在正確位置
- [ ] 下載的視頻正確分類到對應文件夾
- [ ] Excel B 生成在正確位置
- [ ] Excel C 生成在正確位置

### 功能測試
- [ ] Chrome擴展標註正常工作
- [ ] 後端服務器正常運行
- [ ] 視頻下載並分類成功
- [ ] 特徵提取成功
- [ ] 大數據分析成功

---

## 🛠️ 快速測試

### 1. 測試路徑配置
```bash
python config.py
```

### 2. 測試後端服務器
```bash
cd backend
python server.py
# 瀏覽器訪問: http://127.0.0.1:5000/api/stats
```

### 3. 測試下載器（需要先有 Excel A）
```bash
cd downloader
python tiktok_downloader_classified.py
```

### 4. 測試完整流水線
```bash
cd pipeline
python layer1_pipeline.py
```

---

## 📖 相關文檔

- **README_LAYER1.md** - Layer 1 完整使用指南
- **QUICKSTART.md** - 快速開始（已更新）
- **config.py** - 路徑配置文件（可自定義）
- **SYSTEM_OVERVIEW.md** - 完整系統總覽

---

## 🔗 與 Layer 2 的關係

Layer 1 和 Layer 2 使用不同的路徑：

| Layer | 基礎目錄 | 用途 |
|-------|----------|------|
| **Layer 1** | `tiktok tinder videos/` | 人工主導標註 |
| **Layer 2** | `tiktok videos download/` | AI主導自動化 |

兩者數據互不干擾，可以並行使用。

---

**更新時間**: 2025-12-12

**設計原則**: 第一性原理 × 沙皇炸彈 × 猛禽3
